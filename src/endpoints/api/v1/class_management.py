import uuid
from typing import Optional
from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException, status, Form, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, validator
from src.core.utils.helpers import get_current_user, get_current_academic_year
from src.database import get_db
from src.models import Class, School, User, AttendanceSession
from src.core.utils.customize_response import success_response, error_response


router = APIRouter(
    prefix="/api/class",
    tags=["Class Management"]
)


# =========================================================
# 🔹 REQUEST SCHEMA
# =========================================================

class CreateClassRequest(BaseModel):
    school_id: str
    name: str
    section: str
    room_number: Optional[str] = None
    academic_year: str
    created_by: str

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Class name cannot be empty')
        if len(v.strip()) < 2:
            raise ValueError('Class name must be at least 2 characters')
        return v.strip()

    @validator('section')
    def validate_section(cls, v):
        if not v.strip():
            raise ValueError('Section cannot be empty')
        return v.strip().upper()

    @validator('academic_year')
    def validate_academic_year(cls, v):
        import re
        if not re.match(r'^\d{4}-\d{4}$', v):
            raise ValueError('Academic year must be in format: YYYY-YYYY')
        return v


# =========================================================
# 🔹 CREATE CLASS API
# =========================================================

@router.post("/")
def create_class(
    school_id: str = Form(...),
    name: str = Form(...),
    section: str = Form(...),
    room_number: Optional[str] = Form(None),
    academic_year: str = Form(...),
    created_by: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Create a new class in the system.
    
    This endpoint creates a new class with the specified details.
    Validates that the school and creator exist and are active.
    Ensures unique class name within the same school and academic year.
    """
    try:
        # =============================
        # 1️⃣ VALIDATIONS
        # =============================
        
        # Validate school exists and is active
        school = db.query(School).filter(
            School.id == school_id,
            School.is_active == True
        ).first()
        
        if not school:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="School not found or inactive"
            )

        # Validate creator exists and is active
        creator = db.query(User).filter(
            User.id == created_by,
            User.is_active == True
        ).first()
        
        if not creator:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Creator user not found or inactive"
            )

        # Check for duplicate class in same school and academic year
        existing_class = db.query(Class).filter(
            Class.school_id == school_id,
            Class.name == name.strip(),
            Class.section == section.strip().upper(),
            Class.academic_year == academic_year,
            Class.is_active == True
        ).first()
        
        if existing_class:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Class '{name} - {section}' already exists in this school for academic year {academic_year}"
            )

        # =============================
        # 2️⃣ CREATE CLASS
        # =============================
        
        class_obj = Class(
            id=str(uuid.uuid4()),
            school_id=school_id,
            name=name.strip(),
            section=section.strip().upper(),
            room_number=room_number,
            current_strength=0,
            is_active=True,
            academic_year=academic_year,
            created_by=created_by
        )

        db.add(class_obj)
        db.commit()
        db.refresh(class_obj)

        # =============================
        # 3️⃣ RESPONSE
        # =============================
        
        return success_response(
            message="Class created successfully",
            data={
                "id": class_obj.id,
                "name": class_obj.name,
                "section": class_obj.section,
                "full_name": class_obj.full_name,
                "room_number": class_obj.room_number,
                "school_id": class_obj.school_id,
                "school_name": school.name,
                "academic_year": class_obj.academic_year,
                "current_strength": class_obj.current_strength,
                "is_active": class_obj.is_active,
                "created_by": class_obj.created_by,
                "creator_name": creator.fullname,
                "created_at": class_obj.created_at.isoformat() if class_obj.created_at else None
            }
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create class: {str(e)}"
        )
        

@router.get("/list/")
async def get_class_list(
    target_date : Optional[date] = Query(None),
    academic_year: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # ------------------------------------------------------------------
    # AUTH
    # ------------------------------------------------------------------
    try:
        res_ = {
            "message": None,
            "item":[],
            "status_code": None
        }
        if not target_date:
            target_date = datetime.today()
        if not academic_year:
            academic_year = get_current_academic_year()
        # print(academic_year)
        user_id = current_user["user_id"]
        user = db.query(User).filter(User.id==user_id).first()
        if not user:
            res_["message"], res_["status_code"] = 'UNAUTHORIZED USER', 400
            return res_
        school_id = user.school_id
        all_class = db.query(Class).filter(
            Class.school_id == school_id,
            Class.academic_year == academic_year
        ).all()
        resonse_data = []
        # print(target_date)
        for cls in all_class:
            today_class_session = db.query(AttendanceSession).filter(
                AttendanceSession.class_id == cls.id,
                AttendanceSession.session_date == target_date.date()
            ).first()
            # print(today_class_session)
            if today_class_session:
                total_students = today_class_session.total_students
                present_count = today_class_session.present_count
                absent_count = today_class_session.absent_count
            else:
                total_students = 0
                present_count = 0
                absent_count = 0
            resonse_data.append(
                {
                    "id":cls.id,
                    "name": cls.name,
                    "section": cls.section,
                    "total_students":total_students,
                    "present_count":present_count,
                    "absent_count": absent_count
                }
            )
                
        res_["item"], res_["status_code"] = resonse_data, 200
        return res_

    except Exception as e:
        res_["message"], res_["status_code"] = str(e), 500
        return res_





        
        