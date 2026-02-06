import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel, validator

from src.database import get_db
from src.models import Subject, School
from src.core.utils.customize_response import success_response, error_response

router = APIRouter(
    prefix="/api/subject",
    tags=["Subject Management"]
)


# =========================================================
# 🔹 REQUEST SCHEMA
# =========================================================

class CreateSubjectRequest(BaseModel):
    school_id: str
    name: str
    color_code: Optional[str] = "#3498db"

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Subject name cannot be empty')
        if len(v.strip()) < 2:
            raise ValueError('Subject name must be at least 2 characters')
        return v.strip()

    @validator('color_code')
    def validate_color_code(cls, v):
        import re
        if v and not re.match(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$', v):
            raise ValueError('Color code must be a valid hex color (e.g., #3498db)')
        return v


# =========================================================
# 🔹 CREATE SUBJECT API
# =========================================================

@router.post("/")
def create_subject(
    school_id: str = Form(...),
    name: str = Form(...),
    color_code: Optional[str] = Form("#3498db"),
    db: Session = Depends(get_db)
):
    """
    Create a new subject in the system.
    
    This endpoint creates a new subject for a specific school.
    Validates that the school exists and is active.
    Ensures unique subject name within the same school.
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

        # Check for duplicate subject in same school
        existing_subject = db.query(Subject).filter(
            Subject.school_id == school_id,
            Subject.name == name.strip(),
            Subject.is_active == True
        ).first()
        
        if existing_subject:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Subject '{name}' already exists in this school"
            )

        # =============================
        # 2️⃣ CREATE SUBJECT
        # =============================
        
        subject = Subject(
            id=str(uuid.uuid4()),
            school_id=school_id,
            name=name.strip(),
            color_code=color_code,
            is_active=True
        )

        db.add(subject)
        db.commit()
        db.refresh(subject)

        # =============================
        # 3️⃣ RESPONSE
        # =============================
        
        return success_response(
            message="Subject created successfully",
            data={
                "id": subject.id,
                "name": subject.name,
                "color_code": subject.color_code,
                "school_id": subject.school_id,
                "school_name": school.name,
                "is_active": subject.is_active,
                "created_at": subject.created_at.isoformat() if subject.created_at else None,
                "updated_at": subject.updated_at.isoformat() if subject.updated_at else None
            }
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create subject: {str(e)}"
        )
        
