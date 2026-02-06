import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, BinaryIO
import shutil
from fastapi import UploadFile, HTTPException
from PIL import Image
import io
from src.config import settings

class MediaManager:
    def __init__(self):
        self.base_path = settings.UPLOAD_DIR      # "media"
        self.base_url = settings.BASE_URL.rstrip("/")  # 

    def save_file(
        self,
        file: UploadFile,
        media_type: str,
        filename_prefix: str,
        optimize_image: bool = False
    ):
        # ---------------------------
        # SANITIZE FILENAME
        # ---------------------------
        safe_prefix = (
            filename_prefix
            .replace("@", "_")
            .replace(".", "_")
            .replace(" ", "_")
        )

        ext = os.path.splitext(file.filename)[1].lower()
        filename = f"{safe_prefix}_{uuid.uuid4().hex}{ext}"

        # ---------------------------
        # FILESYSTEM PATH (OS SAFE)
        # ---------------------------
        save_dir = os.path.join(self.base_path, media_type)
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, filename)

        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # ---------------------------
        # DB PATH (URL SAFE)
        # ---------------------------
        relative_path = f"{media_type}/{filename}"

        return {
            "relative_path": relative_path
        }

    def delete_file(self, relative_path: str):
        if not relative_path:
            return

        file_path = os.path.join(
            self.base_path,
            relative_path.replace("/", os.sep)
        )

        if os.path.exists(file_path):
            os.remove(file_path)

    def get_file_url(self, relative_path: str | None):
        if not relative_path:
            return None

        return f"{self.base_url}/media/{relative_path.lstrip('/')}"


media_manager = MediaManager()