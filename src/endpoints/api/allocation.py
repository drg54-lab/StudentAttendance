import uuid
from typing import Optional, List, Dict
from datetime import time, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Form, BackgroundTasks
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel, validator
from sqlalchemy import and_, or_, func

from src.database import get_db
from src.models import (
    ClassTeacherAllocation, Class, User, Subject, 
    ClassSchedule, School, SchoolConfig, ScheduleType,
    PeriodTimetable, DayOfWeek
)
from src.core.utils.customize_response import success_response, error_response

router = APIRouter(
    prefix="/api/class-teacher",
    tags=["Class Teacher Allocation"]
)


# =========================================================
# 🔹 REQUEST SCHEMAS
# =========================================================

class CreateClassTeacherAllocationRequest(BaseModel):
    class_id: str
    teacher_id: str
    subject_id: str
    academic_year: str

    @validator('academic_year')
    def validate_academic_year(cls, v):
        import re
        if not re.match(r'^\d{4}-\d{4}$', v):
            raise ValueError('Academic year must be in format: YYYY-YYYY')
        return v


class CreateClassScheduleRequest(BaseModel):
    class_id: str
    schedule_type: str
    periods_per_day: int
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    
    # Day-specific periods (optional, used for custom schedules)
    monday_periods: Optional[int] = None
    tuesday_periods: Optional[int] = None
    wednesday_periods: Optional[int] = None
    thursday_periods: Optional[int] = None
    friday_periods: Optional[int] = None
    saturday_periods: Optional[int] = None
    sunday_periods: Optional[int] = None

    @validator('schedule_type')
    def validate_schedule_type(cls, v):
        valid_types = ["daily", "weekly", "monthly", "alternate", "custom"]
        if v not in valid_types:
            raise ValueError(f'Schedule type must be one of: {", ".join(valid_types)}')
        return v

    @validator('periods_per_day')
    def validate_periods_per_day(cls, v):
        if v <= 0 or v > 12:
            raise ValueError('Periods per day must be between 1 and 12')
        return v


# =========================================================
# 🔹 CREATE CLASS TEACHER ALLOCATION API
# =========================================================

@router.post("/")
def create_class_teacher_allocation(
    class_id: Optional[str] = Form(None),
    teacher_id: str = Form(...),
    subject_id: str = Form(...),
    academic_year: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Allocate a teacher to teach a subject in a class.
    
    This endpoint assigns a teacher to teach a specific subject in a class
    for a given academic year. Validates all entities exist and are active.
    Ensures no duplicate allocations for the same class-subject-teacher combination.
    """
    try:
        # =============================
        # 1️⃣ VALIDATIONS
        # =============================
        
        # Validate class exists and is active
        print(class_id)
        if class_id:
            class_obj = db.query(Class).filter(
                Class.id == class_id,
                Class.is_active == True
            ).first()
            
            if not class_obj:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Class not found or inactive"
                )
        class_obj = None
        # Validate teacher exists, is active, and has teacher role
        teacher = db.query(User).filter(
            User.id == teacher_id,
            User.is_active == True
        ).first()
        
        if not teacher:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Teacher not found or inactive"
            )
        
        # Check if user has teacher role
        from src.models import UserRole
        if teacher.role != UserRole.TEACHER and teacher.role != UserRole.HOD:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User must have teacher or HOD role to be allocated to a class"
            )

        # Validate subject exists and is active
        subject = db.query(Subject).filter(
            Subject.id == subject_id,
            Subject.is_active == True
        ).first()
        
        if not subject:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subject not found or inactive"
            )

        # Check if subject belongs to the same school as class
        if subject.school_id != teacher.school_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Subject does not belong to the same school as the class"
            )

        # # Check if teacher belongs to the same school as class
        # if teacher.school_id != class_obj.school_id:
        #     raise HTTPException(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         detail="Teacher does not belong to the same school as the class"
        #     )

        # Check for duplicate class-subject allocation for the academic year
        existing_subject_allocation = db.query(ClassTeacherAllocation).filter(
            ClassTeacherAllocation.teacher_id == teacher_id,
            ClassTeacherAllocation.subject_id == subject_id,
            ClassTeacherAllocation.academic_year == academic_year
        ).first()
        
        if existing_subject_allocation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Subject '{subject.name}' is already allocated to another teacher for academic year {academic_year}"
            )

        # Check for duplicate class-teacher allocation for the academic year
        # existing_teacher_allocation = db.query(ClassTeacherAllocation).filter(
        #     ClassTeacherAllocation.teacher_id == teacher_id,
        #     ClassTeacherAllocation.academic_year == academic_year
        # ).first()
        
        # if existing_teacher_allocation:
        #     existing_subject = db.query(Subject).filter(Subject.id == existing_teacher_allocation.subject_id).first()
        #     subject_name = existing_subject.name if existing_subject else "Unknown Subject"
        #     raise HTTPException(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         detail=f"Teacher '{teacher.fullname}' is already allocated to teach '{subject_name}' in this class for academic year {academic_year}"
        #     )

        # =============================
        # 2️⃣ CREATE ALLOCATION
        # =============================
        
        allocation = ClassTeacherAllocation(
            id=str(uuid.uuid4()),
            class_id=class_id,
            teacher_id=teacher_id,
            subject_id=subject_id,
            academic_year=academic_year
        )

        db.add(allocation)
        db.commit()
        db.refresh(allocation)

        # =============================
        # 3️⃣ RESPONSE
        # =============================
        
        return success_response(
            message="Class teacher allocation created successfully",
            data={
                "allocation_id": allocation.id,
                "teacher": {
                    "id": teacher.id,
                    "fullname": teacher.fullname,
                    "email": teacher.email,
                    "role": teacher.role.value
                },
                "subject": {
                    "id": subject.id,
                    "name": subject.name,
                    "color_code": subject.color_code
                },
                "academic_year": allocation.academic_year,
                "created_at": allocation.created_at.isoformat() if allocation.created_at else None
            }
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create class teacher allocation: {str(e)}"
        )


# =========================================================
# 🔹 CREATE CLASS SCHEDULE API
# =========================================================

@router.post("/schedule/")
def create_class_schedule(
    class_id: str = Form(...),
    schedule_type: str = Form(...),
    periods_per_day: int = Form(...),
    effective_from: Optional[str] = Form(None),
    effective_to: Optional[str] = Form(None),
    monday_periods: Optional[int] = Form(None),
    tuesday_periods: Optional[int] = Form(None),
    wednesday_periods: Optional[int] = Form(None),
    thursday_periods: Optional[int] = Form(None),
    friday_periods: Optional[int] = Form(None),
    saturday_periods: Optional[int] = Form(None),
    sunday_periods: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Create a class schedule for organizing periods and timings.
    
    This endpoint creates a schedule template for a class that defines:
    - How many periods per day
    - Which days of the week have classes
    - Schedule type (daily, weekly, monthly, alternate, custom)
    - Effective date range
    
    The schedule serves as a template for generating actual period timetables.
    """
    try:
        # =============================
        # 1️⃣ VALIDATIONS
        # =============================
        
        # Validate class exists and is active
        class_obj = db.query(Class).filter(
            Class.id == class_id,
            Class.is_active == True
        ).first()
        
        if not class_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Class not found or inactive"
            )

        # Check if class already has an active schedule
        existing_schedule = db.query(ClassSchedule).filter(
            ClassSchedule.class_id == class_id,
            ClassSchedule.is_active == True
        ).first()
        
        if existing_schedule:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Class already has an active schedule. Deactivate the existing schedule first."
            )

        # Validate schedule type
        try:
            schedule_type_enum = ScheduleType(schedule_type)
        except ValueError:
            valid_types = [st.value for st in ScheduleType]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid schedule type. Must be one of: {', '.join(valid_types)}"
            )

        # Parse date strings
        from datetime import datetime as dt
        effective_from_date = None
        effective_to_date = None
        
        if effective_from:
            try:
                effective_from_date = dt.strptime(effective_from, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="effective_from must be in YYYY-MM-DD format"
                )
        
        if effective_to:
            try:
                effective_to_date = dt.strptime(effective_to, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="effective_to must be in YYYY-MM-DD format"
                )
        
        if effective_from_date and effective_to_date and effective_from_date > effective_to_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="effective_from cannot be after effective_to"
            )

        # Validate day-specific periods for custom schedules
        if schedule_type == "custom":
            day_periods = [
                monday_periods, tuesday_periods, wednesday_periods,
                thursday_periods, friday_periods, saturday_periods, sunday_periods
            ]
            
            if all(period is None for period in day_periods):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="For custom schedules, at least one day must have periods specified"
                )
            
            # Set None values to 0 for custom schedules
        monday_periods = monday_periods or 0
        tuesday_periods = tuesday_periods or 0
        wednesday_periods = wednesday_periods or 0
        thursday_periods = thursday_periods or 0
        friday_periods = friday_periods or 0
        saturday_periods = saturday_periods or 0
        sunday_periods = sunday_periods or 0

        # =============================
        # 2️⃣ CREATE CLASS SCHEDULE
        # =============================
        
        schedule = ClassSchedule(
            id=str(uuid.uuid4()),
            class_id=class_id,
            schedule_type=schedule_type_enum,
            periods_per_day=periods_per_day,
            effective_from=effective_from_date,
            effective_to=effective_to_date,
            is_active=True
        )
        
        # Set day-specific periods for custom schedules
        if schedule_type == "custom":
            schedule.monday_periods = monday_periods
            schedule.tuesday_periods = tuesday_periods
            schedule.wednesday_periods = wednesday_periods
            schedule.thursday_periods = thursday_periods
            schedule.friday_periods = friday_periods
            schedule.saturday_periods = saturday_periods
            schedule.sunday_periods = sunday_periods

        db.add(schedule)
        db.commit()
        db.refresh(schedule)

        # =============================
        # 3️⃣ RESPONSE
        # =============================
        
        response_data = {
            "schedule_id": schedule.id,
            "class": {
                "id": class_obj.id,
                "name": class_obj.full_name,
                "school_id": class_obj.school_id
            },
            "schedule_type": schedule.schedule_type.value,
            "periods_per_day": schedule.periods_per_day,
            "effective_from": schedule.effective_from.isoformat() if schedule.effective_from else None,
            "effective_to": schedule.effective_to.isoformat() if schedule.effective_to else None,
            "is_active": schedule.is_active,
            "created_at": schedule.created_at.isoformat() if schedule.created_at else None
        }
        
        # Include day-specific periods for custom schedules
        if schedule_type == "custom":
            response_data["day_specific_periods"] = {
                "monday": schedule.monday_periods,
                "tuesday": schedule.tuesday_periods,
                "wednesday": schedule.wednesday_periods,
                "thursday": schedule.thursday_periods,
                "friday": schedule.friday_periods,
                "saturday": schedule.saturday_periods,
                "sunday": schedule.sunday_periods
            }

        return success_response(
            message="Class schedule created successfully",
            data=response_data
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create class schedule: {str(e)}"
        )