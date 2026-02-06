from fastapi import APIRouter, Depends, Request
from src.core.utils.helpers import oauth2_scheme
from src.core.utils.customize_response import success_response, error_response
from pydantic import BaseModel, EmailStr
from src.database import get_db
from sqlalchemy.orm import Session
from src.models import User, UserDetails
from src.core.utils.helpers import password_hasher
from datetime import datetime, timedelta
from src.config import settings


router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class UserLogin(BaseModel):
    email: EmailStr
    password: str


@router.post("/login/")
async def login_user(
    request : Request,
    payload: UserLogin,
    db: Session = Depends(get_db)
):
    """Login user"""
    try:
        # Find user
        user = db.query(User).filter(User.email == payload.email).first()
        
        if not user:
            return error_response(
                message="Login failed",
                details="Invalid email or password",
                code=401
            )
        
        # Verify password
        if not password_hasher.verify_password(payload.password, user.password):
            return error_response(
                message="Login failed",
                details="Invalid email or password",
                code=401
            )
        
        # Check if active
        if not user.is_active:
            return error_response(
                message="Login failed",
                details="Account is inactive",
                code=403
            )
        
        # Create token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = password_hasher.create_access_token(
            data={"user_id": user.id, "email": user.email, "role": user.role},
            expires_delta=access_token_expires
        )
        user_details = db.query(UserDetails).filter(
            UserDetails.user_id == user.id
        ).first()
        base_url = str(request.base_url).rstrip("/")
        photo_url = None
        if user_details.photo_url:
            photo_url = f"{base_url}/media/{user_details.photo_url}"
        # Prepare response
        user_data = {
            "id": user.id,
            "email": user.email,
            "fullname": user.fullname,
            "role": user.role,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "department": user.department,
            "phone": photo_url
        }
        
        return success_response(
            message="Login successful",
            data={
                "user": user_data,
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
            }
        )
        
    except Exception as e:
        return error_response(
            message="Login failed",
            details=str(e),
            code=500
        )

