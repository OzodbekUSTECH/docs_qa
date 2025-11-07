import hashlib
from app.dto.common import BaseModelResponse
from app.dto.documents import CreateDocumentRequest
from app.repositories.documents import DocumentsRepository
from app.repositories.uow import UnitOfWork
from app.entities.documents import Document
from app.services.file_service import FileService

from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from app.utils.constants import Storages


class CreateDocumentInteractor:
    
    
    def __init__(
        self,
        uow: UnitOfWork,
        documents_repository: DocumentsRepository,
        file_service: FileService,
    ):
        self.uow = uow
        self.documents_repository = documents_repository
        self.file_service = file_service
        
        
    async def execute(self, request: CreateDocumentRequest) -> BaseModelResponse:
        # Читаем файл один раз
        file_bytes = await request.file.read()
        
        if not file_bytes or len(file_bytes) == 0:
            raise AppError(status_code=400, message=ErrorMessages.DOCUMENT_EMPTY)
        
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        
        existing_document = await self.documents_repository.get_one(
            where=[Document.file_hash == file_hash]
        )
        
        if existing_document:
            raise AppError(status_code=400, message=ErrorMessages.DOCUMENT_ALREADY_EXISTS)
        
        # Сохраняем файл через сервис
        file_path = await self.file_service.save_file(file_bytes, request.file.filename, Storages.DOCUMENTS)
        
        document = Document(
            filename=request.file.filename,
            content_type=request.file.content_type,
            file_path=file_path,
            file_hash=file_hash,
            type=request.type
        )
        
        await self.documents_repository.create(document)
        
        await self.uow.commit()
        
        return BaseModelResponse.model_validate(document)