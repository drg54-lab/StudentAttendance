import uuid
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel, validator

from src.database import get_db
from src.models import Hostel, School, User
from src.core.utils.customize_response import success_response, error_response

router = APIRouter(
    prefix="/api/hostel",
    tags=["Hostel Management"]
)


# =========================================================
# 🔹 REQUEST SCHEMA
# =========================================================

class CreateHostelRequest(BaseModel):
    school_id: str
    name: str
    capacity: int
    address: Optional[str] = None
    warden_id: Optional[str] = None
    room_data: Optional[Dict[str, Any]] = None

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Hostel name cannot be empty')
        if len(v.strip()) < 2:
            raise ValueError('Hostel name must be at least 2 characters')
        return v.strip()

    @validator('capacity')
    def validate_capacity(cls, v):
        if v <= 0:
            raise ValueError('Capacity must be greater than 0')
        if v > 1000:
            raise ValueError('Capacity cannot exceed 1000')
        return v


# =========================================================
# 🔹 CREATE HOSTEL API
# =========================================================

@router.post("/")
def create_hostel(
    school_id: str = Form(...),
    name: str = Form(...),
    capacity: int = Form(...),
    address: Optional[str] = Form(None),
    warden_id: Optional[str] = Form(None),
    room_data: Optional[str] = Form('{}'),  # JSON string
    db: Session = Depends(get_db)
):
    """
    Create a new hostel in the system.
    
    This endpoint creates a new hostel for a specific school.
    Validates that the school exists and is active.
    Optionally assigns a warden to the hostel.
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

        # Check for duplicate hostel name in same school
        existing_hostel = db.query(Hostel).filter(
            Hostel.school_id == school_id,
            Hostel.name == name.strip(),
            Hostel.is_active == True
        ).first()
        
        if existing_hostel:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Hostel '{name}' already exists in this school"
            )

        # Validate warden if provided
        warden = None
        if warden_id:
            warden = db.query(User).filter(
                User.id == warden_id,
                User.is_active == True,
                User.school_id == school_id
            ).first()
            
            if not warden:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Warden user not found, inactive, or doesn't belong to this school"
                )

        # Parse room_data from JSON string
        room_json = {}
        try:
            import json
            if room_data and room_data != '{}':
                room_json = json.loads(room_data)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid room data format. Must be valid JSON."
            )

        # =============================
        # 2️⃣ CREATE HOSTEL
        # =============================
        
        hostel = Hostel(
            id=str(uuid.uuid4()),
            school_id=school_id,
            name=name.strip(),
            capacity=capacity,
            address=address,
            warden_id=warden_id,
            room=room_json,
            is_active=True
        )

        db.add(hostel)
        db.commit()
        db.refresh(hostel)

        # =============================
        # 3️⃣ RESPONSE
        # =============================
        
        response_data = {
            "id": hostel.id,
            "name": hostel.name,
            "school_id": hostel.school_id,
            "school_name": school.name,
            "capacity": hostel.capacity,
            "address": hostel.address,
            "is_active": hostel.is_active,
            "created_at": hostel.created_at.isoformat() if hostel.created_at else None,
            "updated_at": hostel.updated_at.isoformat() if hostel.updated_at else None
        }
        
        # Add warden info if exists
        if warden_id and warden:
            response_data["warden"] = {
                "id": warden.id,
                "fullname": warden.fullname,
                "email": warden.email,
                "role": warden.role.value
            }
        
        # Add room info if exists
        if room_json:
            response_data["room_data"] = room_json

        return success_response(
            message="Hostel created successfully",
            data=response_data
           
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create hostel: {str(e)}"
        )