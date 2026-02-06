import os
import uuid
from fastapi import APIRouter, Depends, Form, File, UploadFile, HTTPException, status, Query, Request
from src.core.utils.helpers import oauth2_scheme
from src.core.utils.customize_response import success_response
from pydantic import EmailStr
from src.database import get_db
from sqlalchemy.orm import Session, joinedload
from src.models import User, UserDetails, School, UserRole, Gender, UserAttendance
from src.core.utils.helpers import password_hasher, face_recognize_
from datetime import datetime, date
from src.config import settings
from typing import Optional
from src.core.utils.media_manager import media_manager
from src.core.utils.helpers import PaginationParams, error_paginated_response, create_paginated_response, apply_pagination_and_sorting
from src.core.utils.helpers import get_current_user
from sqlalchemy import or_
from pydantic import BaseModel

router = APIRouter(prefix="/api/user", tags=["User Management"])

@router.post("/register/", status_code=status.HTTP_201_CREATED)
async def register_user(
    fullname: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    phone: Optional[str] = Form(None),
    dob: Optional[date] = Form(),
    gender: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    face_image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Register a new user with face recognition data
    """
    try:
        # print(current_user)
        user_id = current_user["user_id"]
        user = db.query(User).filter(User.id==user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"User Not Valid"
            )
        school_id = user.school_id
    except Exception as e:
        raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"School Not found"
            )
    try:
        # =============================
        # 1️⃣ VALIDATIONS
        # =============================
        # Check if email already exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Validate role
        try:
            user_role = UserRole(role.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Must be one of: {', '.join([r.value for r in UserRole])}"
            )

        # Validate gender if provided
        user_gender = None
        if gender:
            try:
                user_gender = Gender(gender.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid gender. Must be one of: {', '.join([g.value for g in Gender])}"
                )

        # Validate school exists
        school = db.query(School).filter(
            School.id == school_id,
            School.is_active == True
        ).first()
        
        if not school:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="School not found or inactive"
            )

        # =============================
        # 2️⃣ FACE PROCESSING
        # =============================
        face_bytes = await face_image.read()
        
        # Use MediaManager to save the face image
        try:
            # Save the face image to user_profile directory
            face_image_result = media_manager.save_file(
                file=face_image,
                media_type="user_profile",
                filename_prefix=f"face_{email}",
                optimize_image=True
            )
            
            # Get the relative path for database storage
            face_image_url = face_image_result['relative_path']
            
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save face image: {str(e)}"
            )
        
        # Extract face encoding from the bytes
        face_encoding = face_recognize_.encode_face_image(face_bytes)
        if face_encoding is None:
            # Delete the saved image since face recognition failed
            try:
                media_manager.delete_file(face_image_url)
            except:
                pass
                
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the image. Please upload a clear frontal face image."
            )

        # Convert encoding to list for storage
        encoding_list = face_encoding.tolist() if hasattr(face_encoding, "tolist") else list(face_encoding)

        # =============================
        # 3️⃣ CREATE USER
        # =============================
        user = User(
            id=str(uuid.uuid4()),
            school_id=school_id,
            fullname=fullname.strip(),
            email=email,
            phone=phone,
            password=password_hasher.hash_password(password),
            role=user_role,
            department=department,
            is_active=True
        )

        db.add(user)
        db.flush()

        # =============================
        # 4️⃣ CREATE USER DETAILS
        # =============================
        user_details = UserDetails(
            id=str(uuid.uuid4()),
            user_id=user.id,
            dob=dob,
            gender=user_gender,
            address=address,
            photo_url=face_image_url,
            face_encoding_data={"encoding": encoding_list},
            max_periods_per_day=6,
            max_periods_per_week=30
        )

        db.add(user_details)
        db.commit()

        # =============================
        # 5️⃣ RESPONSE
        # =============================
        return success_response(
            message="User registered successfully",
            data={
                "id": user.id,
                "fullname": user.fullname,
                "email": user.email,
                "role": user.role.value,
                "school_id": user.school_id,
                "department": user.department,
                "is_active": user.is_active,
                "profile_photo_url": media_manager.get_file_url(face_image_url)
            }
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.get("/list/")
async def get_user_list(
    request: Request,   # 👈 ADD THIS
    text: Optional[str] = Query(None, description="Search by name, email, or phone"),
    role: Optional[str] = Query(None, description="Filter by role"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    department: Optional[str] = Query(None, description="Filter by department"),
    
    # Pagination parameters
    pagination: PaginationParams = Depends(),
    
    # Authentication - use get_current_admin for admin-only access
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)  # Changed to admin only
):
    """
    Get paginated list of users with search and filtering (Admin only)
    """
    try:
        # print(current_user["role"], "role")
        if current_user["role"] not in ["hod", "admin"]:
            return error_paginated_response(code=401, message="UNAUTHORIZED USER")
        user_id = current_user["user_id"]
        user = db.query(User).filter(User.id==user_id).first()
        if not user:
            return error_paginated_response(code=401, message="UNAUTHORIZED USER")
    except Exception as e:
        return error_paginated_response(code=500, message=str(e))
        
    try:
        school_id = user.school_id
        # Build base query with eager loading
        query = db.query(User).options(
            joinedload(User.user_details),
            joinedload(User.school)
        )
        
        # Apply text search filter
        if text:
            search_text = f"%{text}%"
            query = query.filter(
                or_(
                    User.fullname.ilike(search_text),
                    User.email.ilike(search_text),
                    User.phone.ilike(search_text)
                )
            )
        
        # Apply role filter
        if role:
            query = query.filter(User.role == role)
        
        # Apply school filter
        if school_id:
            query = query.filter(User.school_id == school_id)
        
        # Apply active status filter
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        # Apply department filter
        if department:
            query = query.filter(User.department == department)
        
        # Get total count before pagination
        total_count = query.count()
        
        # Define allowed sort fields
        allowed_sort_fields = [
            "fullname", "email", "role", "created_at", 
            "updated_at", "is_active", "department"
        ]
        
        # Apply pagination and sorting
        query = apply_pagination_and_sorting(
            query=query,
            pagination=pagination,
            default_sort_field="created_at",
            allowed_sort_fields=allowed_sort_fields
        )
        
        # Execute query
        users = query.all()
        base_url = str(request.base_url).rstrip("/")
        # Prepare response data
        user_list = []
        for user in users:
            photo_url = None

            if user.user_details and user.user_details.photo_url:
                photo_url = f"{base_url}/media/{user.user_details.photo_url}"
                # print(photo_url)
            user_data = {
                "id": user.id,
                "fullname": user.fullname,
                "contact": user.phone,
                "role": user.role,
                "department": user.department,
                "school_id": user.school_id,
                "school_name": user.school.name if user.school else None,
                "gender": user.user_details.gender.value if user.user_details and user.user_details.gender else None,
                "photo_url": photo_url,
                    
            }
            user_list.append(user_data)
        
        # Create paginated response
        response = create_paginated_response(
            items=user_list,
            total_count=total_count,
            page=pagination.page,
            per_page=pagination.per_page,
            message="User list retrieved successfully"
        )
        
        return response
        
    
    except Exception as e:
        return error_paginated_response(code=500, message=str(e))

class AttendanceHistoryResponse(BaseModel):
    id: int
    user_id: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True  # SQLAlchemy -> Pydantic

@router.get("/get_my_attendance_history/")
async def get_user_attendance_history(
    target_date: Optional[date] = Query(None),
    my_history: Optional[bool] = Query(True),

    # Pagination
    pagination: PaginationParams = Depends(),

    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    try:
        # ------------------------------------------------------------------
        # AUTH
        # ------------------------------------------------------------------
        user_id = current_user["user_id"]

        user = (
            db.query(User)
            .options(joinedload(User.school))
            .filter(User.id == user_id)
            .first()
        )

        if not user:
            return error_paginated_response(code=401, message="UNAUTHORIZED USER")

        school_id = user.school_id

        # Teachers & staff can only see their own history
        if user.role in ["teacher", "stuff"]:
            my_history = True

        # ------------------------------------------------------------------
        # BASE QUERY (School-based)
        # ------------------------------------------------------------------
        query = (
            db.query(UserAttendance)
            .join(User)
            .filter(User.school_id == school_id)
        )

        # ------------------------------------------------------------------
        # FILTERS
        # ------------------------------------------------------------------
        if my_history:
            query = query.filter(UserAttendance.user_id == user_id)

        if target_date:
            query = query.filter(UserAttendance.attendance_date == target_date)

        # ------------------------------------------------------------------
        # TOTAL COUNT (before pagination)
        # ------------------------------------------------------------------
        total_count = query.count()

        # ------------------------------------------------------------------
        # SORT + PAGINATION
        # ------------------------------------------------------------------
        allowed_sort_fields = [
            "attendance_date",
            "created_at",
            "status",
            "login_time",
            "logout_time"
        ]

        query = apply_pagination_and_sorting(
            query=query,
            pagination=pagination,
            default_sort_field="created_at",
            allowed_sort_fields=allowed_sort_fields
        )

        attendance_records = query.all()

        # ------------------------------------------------------------------
        # RESPONSE DATA
        # ------------------------------------------------------------------
        attendance_list = []
        for record in attendance_records:
            attendance_list.append({
                "id": record.id,
                "user_id": record.user_id,
                "attendance_date": record.attendance_date,
                "login_time": record.login_time,
                "logout_time": record.logout_time,
                "status": record.status,
                "login_type": record.login_type.value if record.login_type else None,
                "late_minutes": record.late_minutes,
                "remarks": record.remarks,
                "created_at": record.created_at
            })

        # ------------------------------------------------------------------
        # PAGINATED RESPONSE
        # ------------------------------------------------------------------
        return create_paginated_response(
            items=attendance_list,
            total_count=total_count,
            page=pagination.page,
            per_page=pagination.per_page,
            message="Attendance history retrieved successfully"
        )

    except Exception as e:
        return error_paginated_response(code=500, message=str(e))


@router.post("/attendance/logout")
async def logout_from_school(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    try:
        user_id = current_user["user_id"]

        # ----------------------------------------------------
        # AUTH
        # ----------------------------------------------------
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="UNAUTHORIZED USER")

        today = date.today()
        current_time = datetime.now().time()

        # ----------------------------------------------------
        # FETCH TODAY'S ATTENDANCE
        # ----------------------------------------------------
        attendance = (
            db.query(UserAttendance)
            .filter(
                UserAttendance.user_id == user_id,
                UserAttendance.attendance_date == today
            )
            .first()
        )

        if not attendance:
            raise HTTPException(
                status_code=400,
                detail="No login record found for today"
            )

        if attendance.logout_time:
            raise HTTPException(
                status_code=400,
                detail="User already logged out"
            )

        # ----------------------------------------------------
        # UPDATE LOGOUT TIME
        # ----------------------------------------------------
        attendance.logout_time = current_time
        attendance.updated_at = datetime.now()

        # OPTIONAL: Mark absent if login exists but logout is very early
        if attendance.login_time and current_time < attendance.login_time:
            attendance.status = "invalid"

        db.commit()
        db.refresh(attendance)

        return {
            "success": True,
            "message": "Logged out successfully",
            "data": {
                "attendance_id": attendance.id,
                "attendance_date": attendance.attendance_date,
                "login_time": attendance.login_time,
                "logout_time": attendance.logout_time,
                "status": attendance.status
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





  