from fastapi import APIRouter, Depends, UploadFile, File, Query
from src.core.utils.helpers import oauth2_scheme
from src.core.utils.customize_response import success_response, error_response
from pydantic import BaseModel, EmailStr
from src.database import get_db
from sqlalchemy.orm import Session
from src.models import User, UserAttendance
from typing import Optional
from src.core.utils.helpers import password_hasher, face_recognize_, mark_attendance, get_current_user
from datetime import datetime, timedelta, date
from src.config import settings


router = APIRouter(prefix="/api/auth", tags=["Self Attendance"])


@router.post("/self-attendance/")
async def self_attendance(
    face_image: UploadFile = File(...),
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    """Mark self attendance using face recognition"""
    try:
        user = password_hasher.get_current_user(db, token)
        
        # Validate face image
        if face_image.content_type not in settings.ALLOWED_IMAGE_TYPES:
            return error_response(
                message="Attendance failed",
                details="Invalid image type",
                code=400
            )
        
        face_image_bytes = await face_image.read()
        if len(face_image_bytes) > settings.MAX_FILE_SIZE:
            return error_response(
                message="Attendance failed",
                details="Image too large",
                code=400
            )
        
        # Get user's face encodings
        
        user_encodings = face_recognize_.get_user_face_encodings(db, user.id)
        
        if not user_encodings:
            return error_response(
                message="Attendance failed",
                details="No face data registered",
                code=400
            )
        
        # Verify face
        is_match, confidence, error_msg = face_recognize_.verify_user_face(
            face_image_bytes, user_encodings
        )
        
        if not is_match:
            return error_response(
                message="Attendance failed",
                details=error_msg or "Face verification failed",
                code=400
            )
        
        # Mark attendance
        attendance, existing = mark_attendance(db, user.id)
        
        if existing:
            return success_response(
                message="Already marked your Attendance",
                data={
                    "user_id": user.id,
                    "fullname": user.fullname,
                    "login_time": attendance.login_time,
                    "confidence_score": confidence,
                    "attendance_id": attendance.id
                }
            )
        else:
            return success_response(
                message="Attendance marked successfully",
                data={
                    "user_id": user.id,
                    "fullname": user.fullname,
                    "login_time": attendance.login_time,
                    "confidence_score": confidence,
                    "attendance_id": attendance.id
                }
            )
        
    except Exception as e:
        return error_response(
            message="Attendance failed",
            details=str(e),
            code=500
        )
        
        
@router.get("/list/")
async def get_user_list(
    target_date: Optional[date] = Query(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)  # Changed to admin only
):    
    res_ = {
        "message":None,
        "status_code": None,
        "data": None,
        "attend_status": False
    }
    try:
        if not target_date:
            target_date = datetime.today()
        user_id = current_user["user_id"]
        user = db.query(User).filter(
            User.id == user_id
        ).first()
        if not user:
            res_["message"], res_["status_code"] = "User Not Found", 401
            return res_
        user_attendance = db.query(UserAttendance).filter(
            UserAttendance.user_id == user_id,
            UserAttendance.created_at == target_date
        ).first()
        data = {
            "time": None,
            "login_time": None
        }
        if user_attendance:
            data["login_time"], data["time"] = user_attendance.created_at, user_attendance.login_time
            res_["attend_status"], res_["data"], res_["status_code"] = True, data, 200
        else:
            res_["attend_status"], res_["status_code"] = True, 200
        return res_ 
    except Exception as e:
        res_["message"], res_["status_code"] = str(e), 500
        return res_
