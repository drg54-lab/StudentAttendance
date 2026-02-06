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
    """
    Media file management utility for handling all file uploads.
    """
    
    # Base directories
    BASE_MEDIA_DIR = Path(settings.UPLOAD_DIR)
    # print("upload_dir", settings.UPLOAD_DIR)
    # Media type directories
    USER_PROFILE_DIR = BASE_MEDIA_DIR / "user_profile"
    STUDENT_IMAGES_DIR = BASE_MEDIA_DIR / "student_images"
    ATTENDANCE_SESSIONS_DIR = BASE_MEDIA_DIR / "attendance_sessions"
    HOSTEL_IMAGES_DIR = BASE_MEDIA_DIR / "hostel_images"
    SCHOOL_LOGOS_DIR = BASE_MEDIA_DIR / "school_logos"
    TEMP_DIR = BASE_MEDIA_DIR / "temp"
    
    # Allowed file types
    ALLOWED_IMAGE_TYPES = {
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp'
    }
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        'user_profile': 2 * 1024 * 1024,  # 2MB
        'student_images': 3 * 1024 * 1024,  # 3MB
        'attendance_sessions': 5 * 1024 * 1024,  # 5MB
        'default': 5 * 1024 * 1024  # 5MB default
    }
    
    @classmethod
    def get_media_directory(cls, media_type: str) -> Path:
        """
        Get the appropriate directory for a media type.
        
        Args:
            media_type: Type of media (user_profile, student_images, etc.)
        
        Returns:
            Path object for the directory
        """
        directories = {
            'user_profile': cls.USER_PROFILE_DIR,
            'student_images': cls.STUDENT_IMAGES_DIR,
            'attendance_sessions': cls.ATTENDANCE_SESSIONS_DIR,
            'hostel_images': cls.HOSTEL_IMAGES_DIR,
            'school_logos': cls.SCHOOL_LOGOS_DIR,
            'temp': cls.TEMP_DIR
        }
        
        directory = directories.get(media_type)
        if not directory:
            # Use a default subdirectory based on media_type
            directory = cls.BASE_MEDIA_DIR / media_type
        
        # Create directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)
        
        return directory
    
    @classmethod
    def validate_file(
        cls,
        file: UploadFile,
        media_type: str = 'default',
        max_size: Optional[int] = None
    ) -> dict:
        """
        Validate uploaded file.
        
        Args:
            file: UploadFile object
            media_type: Type of media for size validation
            max_size: Custom max size in bytes (overrides default)
        
        Returns:
            Dictionary with validation results
        """
        # Get content type
        content_type = file.content_type
        
        # Validate file type
        if content_type not in cls.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(cls.ALLOWED_IMAGE_TYPES.keys())}"
            )
        
        # Get file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset pointer
        
        # Determine max size
        if max_size is None:
            max_size = cls.MAX_FILE_SIZES.get(media_type, cls.MAX_FILE_SIZES['default'])
        
        # Validate file size
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size // (1024 * 1024)}MB"
            )
        
        # Validate image content (try to open with PIL)
        try:
            image_bytes = file.file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # Verify it's a valid image
            file.file.seek(0)  # Reset pointer
            
            # Get image dimensions
            width, height = image.size
            
            return {
                'valid': True,
                'content_type': content_type,
                'file_size': file_size,
                'extension': cls.ALLOWED_IMAGE_TYPES[content_type],
                'dimensions': {'width': width, 'height': height}
            }
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
    
    @classmethod
    def save_file(
        cls,
        file: UploadFile,
        media_type: str,
        filename_prefix: str = "",
        subdirectory: str = "",
        optimize_image: bool = False,
        max_dimension: Optional[int] = None
    ) -> dict:
        """
        Save uploaded file to appropriate directory.
        
        Args:
            file: UploadFile object
            media_type: Type of media (user_profile, student_images, etc.)
            filename_prefix: Prefix for the filename
            subdirectory: Additional subdirectory within the media type directory
            optimize_image: Whether to optimize/compress the image
            max_dimension: Maximum dimension for image resizing (width/height)
        
        Returns:
            Dictionary with file information
        """
        # Validate file first
        validation = cls.validate_file(file, media_type)
        
        # Get directory
        base_dir = cls.get_media_directory(media_type)
        
        # Create subdirectory if specified
        if subdirectory:
            save_dir = base_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = base_dir
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        extension = validation['extension']
        
        if filename_prefix:
            filename = f"{filename_prefix}_{timestamp}_{unique_id}{extension}"
        else:
            filename = f"{media_type}_{timestamp}_{unique_id}{extension}"
        
        # Full file path
        file_path = save_dir / filename
        
        # Save the file
        try:
            if optimize_image or max_dimension:
                # Process image with PIL
                image_bytes = file.file.read()
                image = Image.open(io.BytesIO(image_bytes))
                
                # Resize if max_dimension specified
                if max_dimension:
                    width, height = image.size
                    if width > max_dimension or height > max_dimension:
                        if width > height:
                            new_width = max_dimension
                            new_height = int(height * (max_dimension / width))
                        else:
                            new_height = max_dimension
                            new_width = int(width * (max_dimension / height))
                        
                        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Optimize/compress if requested
                if optimize_image:
                    # Save with optimization
                    if extension.lower() in ['.jpg', '.jpeg']:
                        image.save(file_path, 'JPEG', quality=85, optimize=True)
                    elif extension.lower() == '.png':
                        image.save(file_path, 'PNG', optimize=True)
                    else:
                        image.save(file_path)
                else:
                    image.save(file_path)
                
                # Get new file size
                file_size = os.path.getsize(file_path)
            else:
                # Save directly
                with open(file_path, "wb") as f:
                    content = file.file.read()
                    f.write(content)
                    file_size = len(content)
            
            # Generate relative path for database storage
            relative_path = str(file_path.relative_to(cls.BASE_MEDIA_DIR))
            
            # Generate URL-friendly path (replace backslashes with forward slashes)
            url_path = relative_path.replace('\\', '/')
            
            return {
                'success': True,
                'original_filename': file.filename,
                'saved_filename': filename,
                'file_path': str(file_path),
                'relative_path': relative_path,
                'url_path': url_path,
                'file_size': file_size,
                'content_type': validation['content_type'],
                'extension': extension,
                'dimensions': validation['dimensions'],
                'media_type': media_type
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file: {str(e)}"
            )
    
    @classmethod
    def save_bytes_as_file(
        cls,
        file_bytes: bytes,
        media_type: str,
        filename_prefix: str = "",
        subdirectory: str = "",
        extension: str = ".jpg"
    ) -> dict:
        """
        Save bytes as file (useful for generated images).
        
        Args:
            file_bytes: Bytes of the file
            media_type: Type of media
            filename_prefix: Prefix for the filename
            subdirectory: Additional subdirectory
            extension: File extension
        
        Returns:
            Dictionary with file information
        """
        try:
            # Get directory
            base_dir = cls.get_media_directory(media_type)
            
            # Create subdirectory if specified
            if subdirectory:
                save_dir = base_dir / subdirectory
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = base_dir
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            
            if filename_prefix:
                filename = f"{filename_prefix}_{timestamp}_{unique_id}{extension}"
            else:
                filename = f"{media_type}_{timestamp}_{unique_id}{extension}"
            
            # Full file path
            file_path = save_dir / filename
            
            # Save bytes to file
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            
            # Generate relative path
            relative_path = str(file_path.relative_to(cls.BASE_MEDIA_DIR))
            url_path = relative_path.replace('\\', '/')
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            return {
                'success': True,
                'saved_filename': filename,
                'file_path': str(file_path),
                'relative_path': relative_path,
                'url_path': url_path,
                'file_size': file_size,
                'extension': extension,
                'media_type': media_type
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save bytes as file: {str(e)}"
            )
    
    @classmethod
    def get_file_url(cls, relative_path: str) -> str:
        """
        Generate full URL for a file based on relative path.
        
        Args:
            relative_path: Relative path stored in database
        
        Returns:
            Full URL for accessing the file
        """
        if not relative_path:
            return ""
        
        # Clean path (replace backslashes with forward slashes)
        clean_path = relative_path.replace('\\', '/')
        
        # Construct URL based on your application's base URL
        base_url = getattr(settings, 'MEDIA_BASE_URL', '/media')
        
        return f"{base_url}/{clean_path}"
    
    @classmethod
    def get_file_path(cls, relative_path: str) -> Path:
        """
        Get absolute file path from relative path.
        
        Args:
            relative_path: Relative path stored in database
        
        Returns:
            Absolute Path object
        """
        if not relative_path:
            raise ValueError("Relative path cannot be empty")
        
        return cls.BASE_MEDIA_DIR / relative_path
    
    @classmethod
    def delete_file(cls, relative_path: str) -> bool:
        """
        Delete a file.
        
        Args:
            relative_path: Relative path stored in database
        
        Returns:
            True if deleted, False if not found
        """
        try:
            file_path = cls.get_file_path(relative_path)
            
            if file_path.exists():
                file_path.unlink()
                return True
            return False
            
        except Exception as e:
            print(f"Error deleting file {relative_path}: {e}")
            return False
    
    @classmethod
    def cleanup_temp_files(cls, older_than_hours: int = 24):
        """
        Clean up temporary files older than specified hours.
        
        Args:
            older_than_hours: Delete files older than this many hours
        """
        temp_dir = cls.TEMP_DIR
        if not temp_dir.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)
        
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file():
                file_time = file_path.stat().st_mtime
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"Error deleting temp file {file_path}: {e}")
    
    @classmethod
    def get_storage_info(cls) -> dict:
        """
        Get information about media storage.
        
        Returns:
            Dictionary with storage information
        """
        info = {
            'base_directory': str(cls.BASE_MEDIA_DIR),
            'total_size': 0,
            'file_count': 0,
            'directories': {}
        }
        
        # Calculate size and count for each directory
        for media_type in ['user_profile', 'student_images', 'attendance_sessions', 
                          'hostel_images', 'school_logos']:
            directory = cls.get_media_directory(media_type)
            size = 0
            count = 0
            
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    size += file_path.stat().st_size
                    count += 1
            
            info['directories'][media_type] = {
                'path': str(directory),
                'size_bytes': size,
                'size_mb': size / (1024 * 1024),
                'file_count': count
            }
            info['total_size'] += size
            info['file_count'] += count
        
        info['total_size_mb'] = info['total_size'] / (1024 * 1024)
        
        return info


# Create global instance
media_manager = MediaManager()