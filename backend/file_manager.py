import os
import uuid
import aiofiles
from typing import Tuple
from fastapi import UploadFile


class FileManager:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

    async def save_file(self, file: UploadFile) -> Tuple[str, str]:
        """
        Save the file and return the paths
        Returns: (file_path, unique_filename)
        """
        # Generate unique file name
        file_extension = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join(self.upload_dir, unique_filename)

        # Save file
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        return file_path, unique_filename

    def get_file_size(self, file_path: str) -> int:
        """Rotate file size"""
        return os.path.getsize(file_path) if os.path.exists(file_path) else 0

    def delete_file(self, file_path: str) -> bool:
        """delete file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except:
            return False
