from pathlib import Path
import aiofiles


class FileService:
    """Сервис для работы с файлами: сохранение и удаление"""
    
    def __init__(self):
        self.storage_dir = Path("storage")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_file(
        self, 
        file_bytes: bytes, 
        filename: str, 
        subdirectory: str = None
    ) -> Path:
        """
        Сохраняет файл на диск асинхронно.
        
        Args:
            file_bytes: Байты файла для сохранения
            filename: Имя файла
            subdirectory: Поддиректория для сохранения (опционально)
            
        Returns:
            Path: Путь к сохраненному файлу
        """
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError("File is empty")
        
        # Определяем путь для сохранения
        if subdirectory:
            save_dir = self.storage_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.storage_dir
        
        file_path = save_dir / filename
        
        # Сохраняем файл асинхронно
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_bytes)
        
        return str(file_path)
    
    async def delete_file(self, file_path: str | Path) -> bool:
        """
        Удаляет файл с диска асинхронно.
        
        Args:
            file_path: Путь к файлу для удаления
            
        Returns:
            bool: True если файл был удален, False если файл не существует
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        
        if not path.exists():
            return False
        
        # Используем aiofiles.os для асинхронного удаления
        import aiofiles.os
        await aiofiles.os.remove(path)
        return True
    
    def get_file_path(self, filename: str, subdirectory: str = None) -> Path:
        """
        Получает путь к файлу без его сохранения.
        
        Args:
            filename: Имя файла
            subdirectory: Поддиректория (опционально)
            
        Returns:
            Path: Путь к файлу
        """
        if subdirectory:
            return self.storage_dir / subdirectory / filename
        return self.storage_dir / filename
    
    def file_exists(self, file_path: str | Path) -> bool:
        """
        Проверяет существование файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True если файл существует
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        return path.exists()

