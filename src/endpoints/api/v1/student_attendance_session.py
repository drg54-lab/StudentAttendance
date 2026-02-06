import uuid
import os
from fastapi import APIRouter, Depends, HTTPException, Query, status, Form, UploadFile, File
from sqlalchemy.orm import Session, joinedload
from datetime import date, datetime, time, timedelta
from typing import List, Optional
import calendar
from pathlib import Path
import json
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import numpy as np
from pydantic import BaseModel
import shutil
from src.core.utils.media_manager import media_manager
from src.database import get_db
from src.models import (
    User, Class, AttendanceSession, AttendanceCaptureLog,
    Enrollment, Student, UserDetails, PeriodTimetable,
    Subject, AttendanceStatus, StudentAttendance
)
from sqlalchemy import desc, and_, or_, func
from src.core.utils.helpers import get_current_user
router = APIRouter(prefix="/api/attendance/session", tags=["Student Attendance Session"])


def create_annotated_image(
    image_path: str,
    face_locations: List[tuple],
    student_names: List[str],
    output_path: str
) -> str:
    """
    Create an annotated image with face detection boxes and names.
    Only saves the annotated image, not the original.
    
    Args:
        image_path: Path to the original image
        face_locations: List of face locations (top, right, bottom, left)
        student_names: List of student names for each face
        output_path: Path to save the annotated image
    
    Returns:
        Path to the annotated image
    """
    try:
        # Load the original image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to load image: {image_path}")
            # If can't load, copy the file as is
            shutil.copy2(image_path, output_path)
            return output_path
        
        # Calculate image dimensions for font scaling
        image_height, image_width = image.shape[:2]
        
        # Calculate dynamic font scale based on image size
        # Smaller images get smaller text, larger images get larger text
        base_font_scale = 0.6  # Smaller base font size
        scale_factor = min(image_width, image_height) / 800  # Normalize to 800px reference
        font_scale = max(0.3, min(0.8, base_font_scale * scale_factor))  # Clamp between 0.3 and 0.8
        
        # Thicker border for face boxes
        border_thickness = 4  # Increased from 3 to 4 for thicker border
        
        # Define font properties
        font = cv2.FONT_HERSHEY_SIMPLEX  # Simpler font for smaller text
        font_thickness = 1  # Thinner text for smaller font
        
        # Draw rectangles and labels for each face
        for (top, right, bottom, left), name in zip(face_locations, student_names):
            # Draw face rectangle with thicker border (bright green color)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), border_thickness)
            
            # Get text size with smaller font
            (text_width, text_height), baseline = cv2.getTextSize(
                name, font, font_scale, font_thickness
            )
            
            # Add padding to text background
            text_padding = 5
            text_padding_vertical = 3
            
            # Calculate background rectangle position
            # Place label above the face box
            bg_top = top - text_height - text_padding_vertical * 2
            bg_bottom = top
            bg_left = left
            bg_right = left + text_width + text_padding * 2
            
            # If label would go above image, place it below the face
            if bg_top < 0:
                bg_top = bottom
                bg_bottom = bottom + text_height + text_padding_vertical * 2
            
            # Ensure background stays within image bounds horizontally
            if bg_right > image_width:
                bg_left = image_width - text_width - text_padding * 2
                bg_right = image_width
            
            # Draw label background (semi-transparent dark green)
            # Create a copy for the background
            bg_color = (0, 100, 0)  # Darker green for background
            
            # Draw filled rectangle for background
            cv2.rectangle(
                image,
                (bg_left, bg_top),
                (bg_right, bg_bottom),
                bg_color,
                -1  # Filled rectangle
            )
            
            # Draw border around label background (optional)
            cv2.rectangle(
                image,
                (bg_left, bg_top),
                (bg_right, bg_bottom),
                (0, 255, 0),  # Bright green border
                1  # Thin border
            )
            
            # Draw name text (white with shadow for better visibility)
            text_x = bg_left + text_padding
            text_y = bg_bottom - text_padding_vertical
            
            # Add text shadow (dark color slightly offset)
            shadow_offset = 1
            cv2.putText(
                image,
                name,
                (text_x + shadow_offset, text_y + shadow_offset),
                font,
                font_scale,
                (0, 0, 0),  # Black shadow
                font_thickness,
                cv2.LINE_AA  # Anti-aliased text
            )
            
            # Draw main text (white)
            cv2.putText(
                image,
                name,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),  # White color
                font_thickness,
                cv2.LINE_AA  # Anti-aliased text
            )
        
        # Save the annotated image (this is the only image saved)
        success = cv2.imwrite(output_path, image)
        
        if not success:
            print(f"Failed to save annotated image: {output_path}")
            # If can't save annotated, copy original
            shutil.copy2(image_path, output_path)
        
        return output_path
        
    except Exception as e:
        print(f"Error creating annotated image: {e}")
        import traceback
        traceback.print_exc()
        # If annotation fails, copy original image as annotated
        shutil.copy2(image_path, output_path)
        return output_path

def get_student_name_for_face(
    face_idx: int,
    matched_faces: List[bool],
    known_student_ids: List[str],
    best_match_indices: List[int],
    student_info: dict
) -> str:
    """
    Get student name for a face based on matching results.
    
    Args:
        face_idx: Index of the face
        matched_faces: List of which faces were matched
        known_student_ids: List of student IDs in known faces
        best_match_indices: List of best match indices for each face
        student_info: Dictionary mapping student_id to student info
    
    Returns:
        Student name or "Unknown"
    """
    if matched_faces[face_idx] and face_idx < len(best_match_indices):
        best_match_idx = best_match_indices[face_idx]
        if best_match_idx < len(known_student_ids):
            student_id = known_student_ids[best_match_idx]
            if student_id in student_info:
                return student_info[student_id]["full_name"]
    
    return "Unknown"

class SaveAttendanceRequest(BaseModel):
    session_id: str
    students: List[dict]


@router.get("/sessions/", response_model=dict)
async def get_attendance_sessions(
    start_date: Optional[date] = Query(None, description="Start date for filtering"),
    end_date: Optional[date] = Query(None, description="End date for filtering"),
    class_id: Optional[str] = Query(None, description="Filter by class ID"),
    teacher_id: Optional[str] = Query(None, description="Filter by teacher ID"),
    status: Optional[str] = Query(None, description="Filter by session status"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get attendance sessions with school-based filtering.
    Users can only see sessions from their own school.
    Defaults to last month's sessions ordered by date.
    """
    try:
        # Get current user's school_id
        user_id = current_user["user_id"]
        user_school_id = current_user.get("school_id")
        
        if not user_school_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any school"
            )
        
        # Default to last month if no date filters provided
        if not start_date:
            start_date = datetime.now().date() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now().date()
        
        # Build base query with school-based filtering
        query = db.query(AttendanceSession).options(
            joinedload(AttendanceSession.class_),
            joinedload(AttendanceSession.teacher),
            joinedload(AttendanceSession.subject)
        ).join(
            Class, AttendanceSession.class_id == Class.id
        ).filter(
            and_(
                AttendanceSession.session_date.between(start_date, end_date),
                Class.school_id == user_school_id  # School-based filter
            )
        )
        
        # Apply additional filters
        if class_id:
            query = query.filter(AttendanceSession.class_id == class_id)
        
        if teacher_id:
            query = query.filter(AttendanceSession.teacher_id == teacher_id)
        
        if status:
            query = query.filter(AttendanceSession.status == status)
        
        # Get user's role for additional filtering
        user_role = current_user.get("role", "")
        
        # If user is teacher, only show their sessions
        if user_role == "teacher":
            query = query.filter(AttendanceSession.teacher_id == user_id)
        
        # If user is class teacher, only show their class sessions
        elif user_role == "class_teacher":
            # Get classes where user is class teacher
            teacher_classes = db.query(Class.id).filter(
                Class.class_teacher_id == user_id,
                Class.school_id == user_school_id
            ).all()
            class_ids = [c[0] for c in teacher_classes]
            
            if class_ids:
                query = query.filter(AttendanceSession.class_id.in_(class_ids))
            else:
                # If not a class teacher of any class, return empty
                return {
                    "message": "No sessions found",
                    "data": [],
                    "pagination": {
                        "page": page,
                        "limit": limit,
                        "total_records": 0,
                        "total_pages": 0,
                        "has_next": False,
                        "has_previous": False
                    }
                }
        
        # Count total records
        total_records = query.count()
        
        # Apply pagination and ordering
        sessions = query.order_by(
            desc(AttendanceSession.session_date),
            desc(AttendanceSession.created_at)
        ).offset((page - 1) * limit).limit(limit).all()
        
        # Prepare response data
        sessions_data = []
        for session in sessions:
            # Get class name
            class_name = session.class_.full_name if session.class_ else f"{session.class_.name} - {session.class_.section}"
            
            # Get teacher name
            teacher_name = session.teacher.fullname if session.teacher else "Unknown"
            
            # Get subject name
            subject_name = session.subject.name if session.subject else None
            
            sessions_data.append({
                "session_id": session.id,
                "session_date": session.session_date.isoformat(),
                "class_id": session.class_id,
                "class_name": class_name,
                "teacher_id": session.teacher_id,
                "teacher_name": teacher_name,
                "subject_id": session.subject_id,
                "subject_name": subject_name,
                "period_timetable_id": session.period_timetable_id,
                "status": session.status,
                "attendance_method": session.attendance_method,
                "total_students": session.total_students,
                "present_count": session.present_count,
                "absent_count": session.absent_count,
                "late_count": session.late_count,
                "attendance_percentage": round((session.present_count / session.total_students * 100), 2) if session.total_students > 0 else 0,
                "scheduled_start": session.scheduled_start.isoformat() if session.scheduled_start else None,
                "scheduled_end": session.scheduled_end.isoformat() if session.scheduled_end else None,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "capture_logs_count": len(session.capture_logs) if session.capture_logs else 0
            })
        
        return {
            "message": "Sessions retrieved successfully",
            "data": sessions_data,
            "pagination": {
                "page": page,
                "limit": limit,
                "total_records": total_records,
                "total_pages": (total_records + limit - 1) // limit,
                "has_next": (page * limit) < total_records,
                "has_previous": page > 1
            },
            "filters": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "class_id": class_id,
                "teacher_id": teacher_id,
                "status": status,
                "school_id": user_school_id
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve sessions: {str(e)}"
        )


@router.post("/create_session_multiple/")
async def create_attendance_session_multiple(
    class_id: str = Form(...),
    subject_id: Optional[str] = Form(None),
    period_timetable_id: Optional[str] = Form(None),
    capture_images: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Create an attendance session with multiple capture images.
    User can only create sessions for classes in their own school.
    """
    try:
        # Validate at least one image is provided
        if not capture_images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one capture image is required"
            )
        
        # =============================
        # 1️⃣ VALIDATIONS with school check
        # =============================
        user_id = current_user["user_id"]
        user_school_id = current_user.get("school_id")
        
        if not user_school_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any school"
            )
        
        # Get teacher with school validation
        teacher = db.query(User).filter(
            User.id == user_id,
            User.is_active == True,
            User.school_id == user_school_id  # Ensure teacher belongs to this school
        ).first()
        
        if not teacher:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Teacher not found or not authorized for this school"
            )

        # Get class with school validation
        class_obj = db.query(Class).filter(
            Class.id == class_id,
            Class.is_active == True,
            Class.school_id == user_school_id  # Class must belong to user's school
        ).first()
        
        if not class_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Class not found or not in your school"
            )

        # Validate subject with school check
        if subject_id:
            subject = db.query(Subject).filter(
                Subject.id == subject_id,
                Subject.is_active == True,
                Subject.school_id == user_school_id  # Subject must belong to user's school
            ).first()
            
            if not subject:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Subject not found or not in your school"
                )

        # Validate period timetable with school check
        if period_timetable_id:
            period_timetable = db.query(PeriodTimetable).filter(
                PeriodTimetable.id == period_timetable_id,
                PeriodTimetable.is_active == True
            ).first()
            
            # Join with class to check school
            if period_timetable:
                timetable_class = db.query(Class).filter(
                    Class.id == period_timetable.class_id,
                    Class.school_id == user_school_id
                ).first()
                
                if not timetable_class:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Period timetable not found in your school"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Period timetable not found"
                )

        # =============================
        # 2️⃣ CREATE OR GET ATTENDANCE SESSION
        # =============================
        
        session_date = datetime.now().date()
        current_time = datetime.now().time()
        
        # Check if session already exists for today (within same school)
        session = db.query(AttendanceSession).join(
            Class, AttendanceSession.class_id == Class.id
        ).filter(
            and_(
                AttendanceSession.class_id == class_id,
                AttendanceSession.session_date == session_date,
                AttendanceSession.teacher_id == user_id,
                Class.school_id == user_school_id  # Ensure session belongs to user's school
            )
        ).first()
        
        if not session:
            session = AttendanceSession(
                id=str(uuid.uuid4()),
                class_id=class_id,
                teacher_id=user_id,
                subject_id=subject_id,
                period_timetable_id=period_timetable_id,
                session_date=session_date,
                scheduled_start=current_time,
                status="active",
                attendance_method="face_recognition"
            )
            db.add(session)
            db.commit()
            db.refresh(session)

        # =============================
        # 3️⃣ GET ENROLLED STUDENTS WITH SCHOOL VALIDATION
        # =============================
        
        enrollments = db.query(Enrollment).options(
            joinedload(Enrollment.student)
        ).filter(
            Enrollment.class_id == class_id,
            Enrollment.status == "active",
            Enrollment.academic_year == class_obj.academic_year
        ).all()
        
        if not enrollments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active students found in this class"
            )
        
        # Prepare student data structures
        student_info = {}
        known_face_encodings = []
        known_student_ids = []
        student_attendance_tracker = {}
        
        for enrollment in enrollments:
            student = enrollment.student
            
            # Verify student belongs to same school
            if student.school_id != user_school_id:
                continue  # Skip students from other schools
            
            # Store student info
            student_info[student.id] = {
                "full_name": student.full_name,
                "roll_number": student.roll_number,
                "first_name": student.first_name,
                "last_name": student.last_name,
                "email": student.email
            }
            
            # Initialize attendance tracker
            student_attendance_tracker[student.id] = {
                "detected_count": 0,
                "best_confidence": 0.0,
                "face_detected": False,
                "has_face_data": False
            }
            
            # Load face encodings if available
            if student.face_encoding_data and "encoding" in student.face_encoding_data:
                try:
                    encoding_list = student.face_encoding_data["encoding"]
                    if isinstance(encoding_list, list) and len(encoding_list) > 0:
                        face_encoding = np.array(encoding_list, dtype=np.float64)
                        known_face_encodings.append(face_encoding)
                        known_student_ids.append(student.id)
                        student_attendance_tracker[student.id]["has_face_data"] = True
                except Exception as e:
                    print(f"Error loading face encoding for student {student.id}: {e}")

        # =============================
        # 4️⃣ PROCESS MULTIPLE IMAGES
        # =============================
        
        capture_logs = []
        all_face_locations = []
        all_student_names = []
        total_faces_detected = 0
        total_faces_recognized = 0
        
        for idx, capture_image in enumerate(capture_images):
            try:
                # Read image bytes
                capture_image_bytes = await capture_image.read()
                
                # Create capture log entry
                timestamp = datetime.now()
                capture_log_id = str(uuid.uuid4())
                
                # Process face recognition for this image
                image_faces_detected = 0
                image_faces_recognized = 0
                face_locations = []
                student_names = []
                
                # Open image
                original_image = Image.open(io.BytesIO(capture_image_bytes))
                
                if known_face_encodings:
                    try:
                        # Convert to numpy array
                        image_array = np.array(original_image)
                        
                        # Detect faces
                        face_locations = face_recognition.face_locations(image_array)
                        face_encodings = face_recognition.face_encodings(image_array, face_locations)
                        
                        image_faces_detected = len(face_encodings)
                        total_faces_detected += image_faces_detected
                        
                        # Process each face
                        for face_idx, face_encoding in enumerate(face_encodings):
                            if known_face_encodings:
                                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                
                                if len(face_distances) > 0:
                                    best_match_index = np.argmin(face_distances)
                                    face_match_threshold = 0.6
                                    
                                    if face_distances[best_match_index] < face_match_threshold:
                                        matched_student_id = known_student_ids[best_match_index]
                                        confidence = 1 - face_distances[best_match_index]
                                        
                                        # Update attendance tracker
                                        if matched_student_id in student_attendance_tracker:
                                            student_attendance_tracker[matched_student_id]["detected_count"] += 1
                                            student_attendance_tracker[matched_student_id]["best_confidence"] = max(
                                                student_attendance_tracker[matched_student_id]["best_confidence"],
                                                confidence
                                            )
                                            student_attendance_tracker[matched_student_id]["face_detected"] = True
                                            image_faces_recognized += 1
                                            total_faces_recognized += 1
                                        
                                        # Get student name for annotation
                                        student_name = f"{student_info[matched_student_id]['first_name']} {student_info[matched_student_id]['last_name'][0]}."
                                        student_names.append(student_name)
                                    else:
                                        student_names.append("Unknown")
                                else:
                                    student_names.append("Unknown")
                            else:
                                student_names.append("Unknown")
                    except Exception as e:
                        print(f"Face recognition error in image {idx}: {e}")
                
                # Store for combined annotation
                all_face_locations.extend(face_locations)
                all_student_names.extend(student_names)
                
                # Create annotated image
                annotated_image = original_image.copy()
                
                if face_locations:
                    draw = ImageDraw.Draw(annotated_image)
                    
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                    
                    # Draw annotations for this image
                    for i, (top, right, bottom, left) in enumerate(face_locations):
                        draw.rectangle([left, top, right, bottom], outline="red", width=3)
                        
                        name = student_names[i] if i < len(student_names) else "Unknown"
                        text_bbox = draw.textbbox((0, 0), name, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        draw.rectangle([left, bottom, left + text_width + 10, bottom + text_height + 10], fill="red")
                        draw.text((left + 5, bottom + 5), name, fill="white", font=font)
                
                # Convert to bytes and save
                img_byte_arr = io.BytesIO()
                annotated_image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                annotated_image_bytes = img_byte_arr.getvalue()
                
                # Save annotated image
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                filename_prefix = f"attendance_{class_obj.name}_{class_obj.section}_{timestamp_str}_img{idx+1}"
                
                annotated_image_result = media_manager.save_bytes_as_file(
                    file_bytes=annotated_image_bytes,
                    media_type="attendance_sessions",
                    filename_prefix=filename_prefix,
                    subdirectory=f"session_{session.id}",
                    extension=".jpg"
                )
                
                annotated_image_url = annotated_image_result['relative_path']
                
                # Create capture log
                capture_log = AttendanceCaptureLog(
                    id=capture_log_id,
                    attendance_session_id=session.id,
                    image_url=annotated_image_url,
                    timestamp=timestamp,
                    detected_faces=image_faces_detected,
                    recognition_confidence=round((image_faces_recognized / image_faces_detected * 100), 2) if image_faces_detected > 0 else 0
                )
                db.add(capture_log)
                capture_logs.append({
                    "id": capture_log_id,
                    "image_url": media_manager.get_file_url(annotated_image_url),
                    "timestamp": timestamp.isoformat(),
                    "detected_faces": image_faces_detected,
                    "recognized_faces": image_faces_recognized
                })
                
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
        
        # Commit all capture logs
        db.commit()
        
        # =============================
        # 5️⃣ DETERMINE FINAL ATTENDANCE STATUS
        # =============================
        
        students_data = []
        present_count = 0
        
        for student_id, tracker in student_attendance_tracker.items():
            student = student_info.get(student_id)
            if not student:
                continue
            
            # Determine attendance status
            attendance_status = "present" if tracker["face_detected"] else "absent"
            
            if attendance_status == "present":
                present_count += 1
            
            students_data.append({
                "student_id": student_id,
                "roll_number": student["roll_number"],
                "full_name": student["full_name"],
                "attendance_status": attendance_status,
                "confidence": round(tracker["best_confidence"] * 100, 2) if tracker["best_confidence"] > 0 else 0.0,
                "detected_in_images": tracker["detected_count"],
                "is_face_detected": tracker["face_detected"],
                "has_face_data": tracker["has_face_data"]
            })
        
        # =============================
        # 6️⃣ UPDATE SESSION STATISTICS
        # =============================
        
        total_students = len(students_data)
        absent_count = total_students - present_count
        
        session.total_students = total_students
        session.present_count = present_count
        session.absent_count = absent_count
        session.late_count = 0
        
        db.commit()
        
        # =============================
        # 7️⃣ RESPONSE
        # =============================
        
        return {
            "message": "Attendance session created successfully with multiple captures",
            "session_id": session.id,
            "school_id": user_school_id,
            "session_details": {
                "session_date": session.session_date.isoformat(),
                "class_id": session.class_id,
                "class_name": class_obj.full_name,
                "teacher_id": session.teacher_id,
                "teacher_name": teacher.fullname,
                "subject_id": session.subject_id,
                "status": session.status,
                "attendance_method": session.attendance_method
            },
            "capture_summary": {
                "total_images": len(capture_images),
                "successfully_processed": len(capture_logs),
                "total_faces_detected": total_faces_detected,
                "total_faces_recognized": total_faces_recognized,
                "recognition_rate": round((total_faces_recognized / total_faces_detected * 100), 2) if total_faces_detected > 0 else 0
            },
            "attendance_summary": {
                "total_students": total_students,
                "present_count": present_count,
                "absent_count": absent_count,
                "attendance_percentage": round((present_count / total_students * 100), 2) if total_students > 0 else 0
            },
            "capture_logs": capture_logs,
            "students": students_data,
            "recommendation": "Please review the recognition results and call /save_attendance/ to save records"
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create attendance session: {str(e)}"
        )


@router.get("/session/{session_id}/", response_model=dict)
async def get_session_details(
    session_id: str,
    include_absent: bool = Query(True, description="Include absent students in response"),
    include_present: bool = Query(True, description="Include present students in response"),
    include_face_detection: bool = Query(True, description="Include face detection details"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get detailed session information with school-based access control.
    Users can only access sessions from their own school.
    """
    try:
        # Get current user's school_id
        user_school_id = current_user.get("school_id")
        
        if not user_school_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User is not associated with any school"
            )
        
        # =============================
        # 1️⃣ GET SESSION WITH SCHOOL VALIDATION
        # =============================
        
        session = db.query(AttendanceSession).options(
            joinedload(AttendanceSession.class_),
            joinedload(AttendanceSession.teacher),
            joinedload(AttendanceSession.subject),
            joinedload(AttendanceSession.capture_logs),
            joinedload(AttendanceSession.student_attendance).joinedload(StudentAttendance.student)
        ).join(
            Class, AttendanceSession.class_id == Class.id
        ).filter(
            and_(
                AttendanceSession.id == session_id,
                Class.school_id == user_school_id  # School-based access control
            )
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Attendance session not found or not accessible"
            )
        
        # =============================
        # 2️⃣ GET ALL ENROLLED STUDENTS FROM SAME SCHOOL
        # =============================
        
        class_obj = session.class_
        enrollments = db.query(Enrollment).options(
            joinedload(Enrollment.student)
        ).filter(
            Enrollment.class_id == session.class_id,
            Enrollment.status == "active",
            Enrollment.academic_year == class_obj.academic_year
        ).all()
        
        # Create map of all enrolled students (only from same school)
        all_students_map = {}
        for enrollment in enrollments:
            student = enrollment.student
            
            # Filter by school
            if student.school_id != user_school_id:
                continue
                
            all_students_map[student.id] = {
                "student_id": student.id,
                "roll_number": student.roll_number,
                "full_name": student.full_name,
                "email": student.email,
                "phone": student.phone,
                "has_face_data": bool(student.face_encoding_data and "encoding" in student.face_encoding_data),
                "attendance_status": "absent",  # Default
                "is_face_detected": False,
                "confidence": 0.0,
                "is_late": False,
                "late_minutes": 0,
                "remarks": None,
                "recorded_by": None,
                "recorded_at": None
            }
        
        # =============================
        # 3️⃣ GET ATTENDANCE RECORDS AND UPDATE STATUS
        # =============================
        
        attendance_records = session.student_attendance
        present_students = []
        absent_students = []
        
        # Update student status from attendance records
        for record in attendance_records:
            if record.student_id in all_students_map:
                all_students_map[record.student_id].update({
                    "attendance_status": record.status.value,
                    "is_face_detected": record.is_face_detected,
                    "is_late": record.is_late,
                    "late_minutes": record.late_minutes,
                    "remarks": record.remarks,
                    "recorded_by": record.recorded_by,
                    "recorded_at": record.created_at.isoformat() if record.created_at else None
                })
                
                if record.status == AttendanceStatus.present:
                    present_students.append(all_students_map[record.student_id])
                elif record.status == AttendanceStatus.absent:
                    absent_students.append(all_students_map[record.student_id])
        
        # Get students without attendance records (absent by default)
        for student_id, student_data in all_students_map.items():
            if student_data["attendance_status"] == "absent" and student_id not in [s["student_id"] for s in absent_students]:
                absent_students.append(student_data)
        
        # Sort students
        present_students.sort(key=lambda x: x["roll_number"])
        absent_students.sort(key=lambda x: x["roll_number"])
        
        # =============================
        # 4️⃣ GET CAPTURE LOGS DETAILS
        # =============================
        
        capture_logs_details = []
        for log in session.capture_logs:
            capture_logs_details.append({
                "id": log.id,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "image_url": media_manager.get_file_url(log.image_url) if log.image_url else None,
                "detected_faces": log.detected_faces,
                "recognition_confidence": log.recognition_confidence,
                "created_at": log.created_at.isoformat() if log.created_at else None
            })
        
        # =============================
        # 5️⃣ PREPARE RESPONSE
        # =============================
        
        response_data = {
            "session_id": session.id,
            "school_id": user_school_id,
            "session_date": session.session_date.isoformat(),
            "class_details": {
                "class_id": session.class_id,
                "class_name": class_obj.full_name if class_obj else "Unknown",
                "grade": class_obj.grade if class_obj else None,
                "section": class_obj.section if class_obj else None,
                "academic_year": class_obj.academic_year if class_obj else None,
                "school_id": class_obj.school_id if class_obj else None
            },
            "teacher_details": {
                "teacher_id": session.teacher_id,
                "teacher_name": session.teacher.fullname if session.teacher else "Unknown",
                "email": session.teacher.email if session.teacher else None,
                "school_id": session.teacher.school_id if session.teacher else None
            },
            "subject_details": {
                "subject_id": session.subject_id,
                "subject_name": session.subject.name if session.subject else None,
                "subject_code": session.subject.code if session.subject else None,
                "school_id": session.subject.school_id if session.subject else None
            } if session.subject_id else None,
            "timetable_details": {
                "period_timetable_id": session.period_timetable_id,
                "scheduled_start": session.scheduled_start.isoformat() if session.scheduled_start else None,
                "scheduled_end": session.scheduled_end.isoformat() if session.scheduled_end else None
            },
            "session_status": {
                "status": session.status,
                "attendance_method": session.attendance_method,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None
            },
            "statistics": {
                "total_students": session.total_students,
                "present_count": session.present_count,
                "absent_count": session.absent_count,
                "late_count": session.late_count,
                "attendance_percentage": round((session.present_count / session.total_students * 100), 2) if session.total_students > 0 else 0,
                "expected_students": len(all_students_map),
                "students_from_same_school": len(all_students_map)
            },
            "capture_logs": {
                "total_captures": len(capture_logs_details),
                "logs": capture_logs_details
            }
        }
        
        # Add student lists based on filters
        if include_present:
            response_data["present_students"] = {
                "count": len(present_students),
                "students": present_students
            }
        
        if include_absent:
            response_data["absent_students"] = {
                "count": len(absent_students),
                "students": absent_students
            }
        
        # Add face detection summary if requested
        if include_face_detection:
            students_with_face_data = sum(1 for s in all_students_map.values() if s["has_face_data"])
            students_detected = sum(1 for s in all_students_map.values() if s["is_face_detected"])
            
            response_data["face_detection_summary"] = {
                "students_with_face_data": students_with_face_data,
                "students_without_face_data": len(all_students_map) - students_with_face_data,
                "students_detected": students_detected,
                "detection_rate": round((students_detected / len(all_students_map) * 100), 2) if all_students_map else 0
            }
        
        return {
            "message": "Session details retrieved successfully",
            "school_id": user_school_id,
            "data": response_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve session details: {str(e)}"
        )
        
        
        
        
        
        
        
        
        
        