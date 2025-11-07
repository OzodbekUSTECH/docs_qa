from app.repositories.documents import DocumentsRepository
from app.dto.common import BaseResponse
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from app.entities.documents import Document
from app.repositories.uow import UnitOfWork
from app.services.file_service import FileService



class DeleteDocumentInteractor:
    
    def __init__(
        self,
        documents_repository: DocumentsRepository,
        uow: UnitOfWork,
        file_service: FileService,
    ):
        self.documents_repository = documents_repository
        self.uow = uow
        self.file_service = file_service
    async def execute(self, document_id: int) -> BaseResponse:
        document = await self.documents_repository.get_one(
            where=[Document.id == document_id]
        )
        if not document:
            raise AppError(status_code=404, message=ErrorMessages.DOCUMENT_NOT_FOUND)
        await self.documents_repository.delete(document_id)
        await self.file_service.delete_file(document.file_path)
        await self.uow.commit()
        
        return BaseResponse()