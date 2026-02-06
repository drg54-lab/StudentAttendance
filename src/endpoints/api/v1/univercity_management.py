from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import time
from typing import Optional, List
from pydantic import BaseModel, EmailStr, validator
import uuid

from src.database import get_db
from src.models import University, School, SchoolConfig
from src.core.utils.customize_response import success_response, error_response

router = APIRouter(
    prefix="/api/university",
    tags=["University & School"]
)

# =========================================================
# 🔹 SCHEMAS
# =========================================================

class UniversityCreate(BaseModel):
    name: str
    email: EmailStr
    logo_url: Optional[str] = None

    @validator('name')
    def validate_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('University name must be at least 2 characters')
        return v.strip()



class SchoolCreate(BaseModel):
    university_id: str
    name: str
    description: str  # Changed from 'type' to 'description' to match model
    code: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    lat: Optional[str] = None
    long: Optional[str] = None

    @validator('name')
    def validate_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('School name must be at least 2 characters')
        return v.strip()

    @validator('description')
    def validate_description(cls, v):
        if len(v.strip()) < 5:
            raise ValueError('Description must be at least 5 characters')
        return v.strip()


class SchoolConfigCreate(BaseModel):
    academic_year: str
    schedule_type: str = "weekly"  # Default value
    default_periods_per_day: int = 8
    period_duration: int = 45
    break_duration: int = 15
    school_start_time: time = time(8, 0)
    school_end_time: time = time(14, 0)
    is_half_day_allowed: bool = True
    half_day_periods: int = 4
    enable_automatic_schedule: bool = True
    lat: str = None
    long: str = None

    # @validator('academic_year')
    # def validate_academic_year(cls, v):
    #     # Validate format like "2024-2025"
    #     import re
    #     if not re.match(r'^\d{4}-\d{4}$', v):
    #         raise ValueError('Academic year must be in format: YYYY-YYYY')
    #     return v

    @validator('schedule_type')
    def validate_schedule_type(cls, v):
        valid_types = ["daily", "weekly", "monthly", "alternate", "custom"]
        if v not in valid_types:
            raise ValueError(f'Schedule type must be one of: {", ".join(valid_types)}')
        return v


class SchoolWithConfigCreate(BaseModel):
    school: SchoolCreate
    config: SchoolConfigCreate


# =========================================================
# 🔹 CREATE UNIVERSITY
# =========================================================

@router.post("/register/")
def create_university(
    payload: UniversityCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new university
    """
    try:
        # Check if university name already exists
        existing_university = db.query(University).filter(
            University.name == payload.name
        ).first()
        
        if existing_university:
            return error_response(
                    message = "University name already exists", 
                    code=400, 
                    details=None
                )
            

        # Check if email already exists
        existing_email = db.query(University).filter(
            University.email == payload.email
        ).first()
        
        if existing_email:
            return error_response(
                    message = "University email already exists", 
                    code=400, 
                    details=None
                )
            


        # Create university
        university = University(
            id=str(uuid.uuid4()),
            name=payload.name,
            email=payload.email,
            logo_url=payload.logo_url,
            is_active=True
        )

        db.add(university)
        db.commit()
        db.refresh(university)

        return success_response(
            message="University created successfully",
            data={
                "id": university.id,
                "name": university.name,
                "email": university.email,
                "is_active": university.is_active
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        return error_response(
                message = "Failed to create university", 
                code=400, 
                details=str(e)
            )


# =========================================================
# 🔹 CREATE SCHOOL WITH CONFIG (ATOMIC)
# =========================================================

@router.post("/schools/")
def create_school_with_config(
    payload: SchoolWithConfigCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new school with its configuration
    """
    try:
        # Validate university
        university = db.query(University).filter(
            University.id == payload.school.university_id,
            University.is_active == True
        ).first()

        if not university:
            return error_response(
                    message = "University not found or inactive", 
                    code=400, 
                    details=None
            )
            

        # Check if school name already exists for this university
        existing_school = db.query(School).filter(
            School.university_id == payload.school.university_id,
            School.name == payload.school.name
        ).first()
        
        if existing_school:
            return error_response(
                    message = "School name already exists in this university", 
                    code=400, 
                    details=None
            )
            

        # Create School
        school = School(
            id=str(uuid.uuid4()),
            university_id=payload.school.university_id,
            name=payload.school.name,
            description=payload.school.description,  # Changed from type to description
            email=payload.school.email,
            phone=payload.school.phone,
            address=payload.school.address,
            lat=payload.school.lat,
            long=payload.school.long,
            is_active=True
        )

        db.add(school)
        db.flush()  # Get school.id without committing

        # Create School Config
        config = SchoolConfig(
            id=str(uuid.uuid4()),
            school_id=school.id,
            academic_year=payload.config.academic_year,
            schedule_type=payload.config.schedule_type,
            default_periods_per_day=payload.config.default_periods_per_day,
            period_duration=payload.config.period_duration,
            break_duration=payload.config.break_duration,
            school_start_time=payload.config.school_start_time,
            school_end_time=payload.config.school_end_time,
            is_half_day_allowed=payload.config.is_half_day_allowed,
            half_day_periods=payload.config.half_day_periods,
            enable_automatic_schedule=payload.config.enable_automatic_schedule,
            lat=payload.config.lat,
            long=payload.config.long
        )

        db.add(config)
        db.commit()
        
        # Refresh both objects
        db.refresh(school)
        db.refresh(config)

        return success_response(
            message="School created successfully with configuration",
            data={
                "school": {
                    "id": school.id,
                    "name": school.name,
                    "description": school.description,
                    "university_id": school.university_id,
                    "email": school.email
                },
                "config": {
                    "academic_year": config.academic_year,
                    "schedule_type": config.schedule_type,
                    "periods_per_day": config.default_periods_per_day,
                    "school_start_time": config.school_start_time.strftime("%H:%M"),
                    "school_end_time": config.school_end_time.strftime("%H:%M")
                }
            }
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        return error_response(
            message = "Failed to create school", 
            code=400, 
            details=str(e)
        )
       
        
        
        
        
        