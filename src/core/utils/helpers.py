import hashlib, cv2
import bcrypt
from datetime import datetime, timedelta
from src.config import settings
import uuid
from jose import jwt
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from src.models import User, UserDetails, UserAttendance
from fastapi import HTTPException, status, Query, Depends
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import face_recognition as fr
from sqlalchemy import func
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
import math
# Setup logger


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


from datetime import date

def get_current_academic_year() -> str:
    today = date.today()
    year = today.year
    return f"{year}"


class AuthenticationFun:
    @staticmethod
    def hash_password(password: str) -> str:
        """SHA-256 pre-hash → bcrypt hash"""
        prehashed = hashlib.sha256(password.encode("utf-8")).digest()
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(prehashed, salt)
        return hashed.decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password using SHA-256 + bcrypt"""
        try:
            prehashed = hashlib.sha256(plain_password.encode("utf-8")).digest()
            return bcrypt.checkpw(prehashed, hashed_password.encode("utf-8"))
        except Exception:
            return False

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now() + expires_delta
        else:
            expire = datetime.now() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "jti": str(uuid.uuid4()),
            "iat": datetime.now()
        })
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    @staticmethod
    def get_current_user(db: Session, token: str) -> User:
        """Get current user from token"""
        payload = AuthenticationFun.verify_token(token)
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )
        return user

    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

class FaceRecognitionFunc:
    FACE_MATCH_THRESHOLD = getattr(settings, 'FACE_MATCH_THRESHOLD', 0.6)
    face_detection_model = "hog"  # or "cnn" for better accuracy but slower
    
    @staticmethod
    def encode_image_to_array(image_bytes: bytes) -> Optional[np.ndarray]:
        """Convert image bytes to numpy array."""
        try:
            # Try OpenCV first
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                # Fallback to PIL
                image = Image.open(BytesIO(image_bytes))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                img = np.array(image)
            
            return img
        except Exception as e:
            logger.error(f"Error encoding image to array: {e}")
            return None
    
    @staticmethod
    def encode_face_image(image_bytes: bytes) -> Optional[np.ndarray]:
        """Extract face encoding from image bytes."""
        try:
            img = FaceRecognitionFunc.encode_image_to_array(image_bytes)
            if img is None:
                return None
            
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = img
            
            # Detect faces
            face_locations = fr.face_locations(rgb_img, model="hog")
            
            if not face_locations:
                return None
            
            # Get encoding for first face
            face_encodings = fr.face_encodings(rgb_img, face_locations)
            return face_encodings[0] if face_encodings else None
            
        except Exception as e:
            logger.error(f"Error encoding face image: {e}")
            return None
    
    @staticmethod
    def get_user_face_encodings(db: Session, user_id: str) -> List[np.ndarray]:
        """Get all face encodings for a user from UserDetails"""
        try:
            user_details = db.query(UserDetails).filter(
                UserDetails.user_id == user_id
            ).first()
            
            if not user_details or not user_details.face_encoding_data:
                return []
            
            # Check if encoding exists in face_encoding_data
            if isinstance(user_details.face_encoding_data, dict) and "encoding" in user_details.face_encoding_data:
                encoding = user_details.face_encoding_data["encoding"]
                if isinstance(encoding, list) and len(encoding) > 0:
                    return [np.array(encoding)]
            
            return []
        except Exception as e:
            logger.error(f"Error getting user face encodings: {e}")
            return []
    
    @staticmethod
    def verify_user_face(image_bytes: bytes, user_encodings: List[np.ndarray]) -> Tuple[bool, float, Optional[str]]:
        """
        Verify if face belongs to user
        Returns: (is_match, confidence, error_message)
        """
        try:
            new_encoding = FaceRecognitionFunc.encode_face_image(image_bytes)
            if new_encoding is None:
                return False, 0.0, "No face detected in image"
            
            if not user_encodings:
                return False, 0.0, "No face data registered"
            
            # Calculate distances
            face_distances = fr.face_distance(user_encodings, new_encoding)
            min_distance = np.min(face_distances)
            
            # Convert distance to confidence (0-100)
            confidence = max(0, 100 - (min_distance * 100))
            
            # Use threshold from settings or default
            threshold = getattr(settings, 'FACE_MATCH_THRESHOLD', 0.6)
            if min_distance <= threshold:
                return True, confidence, None
            else:
                return False, confidence, f"Face verification failed (confidence: {confidence:.1f}%)"
                
        except Exception as e:
            logger.error(f"Error verifying face: {e}")
            return False, 0.0, f"Face verification error: {str(e)}"

# src/core/utils/pagination.py



class PaginationParams:
    """Dependency for pagination parameters"""
    def __init__(
        self,
        page: int = Query(1, ge=1, description="Page number"),
        per_page: int = Query(10, ge=1, le=100, description="Items per page"),
        sort_by: Optional[str] = Query(None, description="Field to sort by"),
        sort_order: str = Query("asc")
    ):
        self.page = page
        self.per_page = per_page
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.offset = (page - 1) * per_page

class PaginatedResponse(BaseModel):
    """Base model for paginated responses"""
    message: str
    data: List[Dict[str, Any]]
    pagination: Dict[str, Any]
    
    class Config:
        from_attributes = True

def create_paginated_response(
    items: List[Any],
    total_count: int,
    page: int,
    per_page: int,
    message: str = "Data retrieved successfully"
) -> Dict[str, Any]:
    """
    Create standardized paginated response
    """
    total_pages = math.ceil(total_count / per_page) if total_count > 0 else 0
    
    return {
        "message": message,
        "data": items,
        "current_page": page,
        "per_page": per_page,
        "total_items": total_count,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1,
        "next_page": page + 1 if page < total_pages else None,
        "previous_page": page - 1 if page > 1 else None
    }
    
def error_paginated_response(
    code : int = 500,
    message: str = "Something is wrong"
) -> Dict[str, Any]:
    
    return {
        "message": message,
        "data": [],
        "current_page": None,
        "per_page": None,
        "total_items": None,
        "total_pages": None,
        "has_next": None,
        "has_previous": None,
        "next_page": None,
        "previous_page": None,
        "status_code": code
    }

def apply_pagination_and_sorting(
    query,
    pagination: PaginationParams,
    default_sort_field: str = "created_at",
    allowed_sort_fields: List[str] = None
):
    """
    Apply pagination and sorting to SQLAlchemy query
    """
    # Apply sorting
    if pagination.sort_by:
        if allowed_sort_fields and pagination.sort_by not in allowed_sort_fields:
            # If sort field is not allowed, use default
            sort_field = getattr(query.column_descriptions[0]['entity'], default_sort_field)
        else:
            sort_field = getattr(query.column_descriptions[0]['entity'], pagination.sort_by)
    else:
        sort_field = getattr(query.column_descriptions[0]['entity'], default_sort_field)
    
    # Apply sort order
    if pagination.sort_order == "desc":
        query = query.order_by(sort_field.desc())
    else:
        query = query.order_by(sort_field.asc())
    
    # Apply pagination
    query = query.offset(pagination.offset).limit(pagination.per_page)
    
    return query

   
from jose import jwt, JWTError

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        print(payload)
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )




def mark_attendance(db: Session, user_id: str, login_type: str = "face_recognition") -> UserAttendance:
    """Mark attendance for user"""
    try:
        today = datetime.now().date()
        current_time = datetime.now().time()
        
        # Check existing attendance for today
        existing = db.query(UserAttendance).filter(
            UserAttendance.user_id == user_id,
            UserAttendance.attendance_date == today
        ).first()
        
        if existing:
            return existing, True  # Return existing record and flag indicating it already existed
        
        # Create new attendance
        attendance = UserAttendance(
            id=str(uuid.uuid4()),
            user_id=user_id,
            attendance_date=today,
            login_time=current_time,
            status="present",
            login_type=login_type,
            is_face_detected=(login_type == "face_recognition")
        )
        
        db.add(attendance)
        db.commit()
        db.refresh(attendance)
        return attendance, False  # Return new record and flag indicating it was created
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error marking attendance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error marking attendance: {str(e)}"
        )

password_hasher = AuthenticationFun()
face_recognize_ = FaceRecognitionFunc()


        
