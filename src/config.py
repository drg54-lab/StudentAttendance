# src/config.py
import os
from dotenv import load_dotenv
from typing import List
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv()


def get_required(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def get_list(key: str) -> List[str]:
    return [item.strip() for item in get_required(key).split(",") if item.strip()]


class Settings:
    # ---------------------------------------------------------------------
    # DATABASE
    # ---------------------------------------------------------------------
    DATABASE_HOST: str = get_required("DB_HOST")
    DATABASE_PORT: int = int(get_required("DB_PORT"))
    DATABASE_USER: str = get_required("DB_USER")
    DATABASE_PASSWORD: str = get_required("DB_PASSWORD")
    DATABASE_NAME: str = get_required("DB_NAME")

    # Full SQLAlchemy database URL (PRIMARY)
    DATABASE_URL: str = f"mysql+pymysql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}?charset=utf8mb4"

    # ---------------------------------------------------------------------
    # JWT / SECURITY
    # ---------------------------------------------------------------------
    SECRET_KEY: str = get_required("SECRET_KEY")
    ALGORITHM: str = get_required("ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        get_required("ACCESS_TOKEN_EXPIRE_MINUTES")
    )
    JWT_DEBUG_MODE: bool = (
        get_required("JWT_DEBUG_MODE").lower() == "true"
    )

    # ---------------------------------------------------------------------
    # CORS
    # ---------------------------------------------------------------------
    CORS_ALLOW_ORIGINS: List[str] = get_list("CORS_ALLOW_ORIGINS")
    CORS_ALLOW_METHODS: List[str] = get_list("CORS_ALLOW_METHODS")
    CORS_ALLOW_HEADERS: List[str] = get_list("CORS_ALLOW_HEADERS")
    CORS_ALLOW_CREDENTIALS: bool = (
        get_required("CORS_ALLOW_CREDENTIALS").lower() == "true"
    )

    # ---------------------------------------------------------------------
    # FILE UPLOAD
    # ---------------------------------------------------------------------
    UPLOAD_DIR: str = get_required("UPLOAD_DIR")

    # ---------------------------------------------------------------------
    # FACE RECOGNITION
    # ---------------------------------------------------------------------
    FACE_ENCODING_MODEL: str = get_required("FACE_ENCODING_MODEL")
    FACE_MATCH_THRESHOLD: float = float(
        get_required("FACE_MATCH_THRESHOLD")
    )
    ALLOWED_IMAGE_TYPES = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


# Singleton instance
settings = Settings()
