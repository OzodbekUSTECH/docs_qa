from app.repositories.documents import DocumentsRepository
from app.repositories.uow import UnitOfWork
from app.dto.documents import UpdateDocumentPartiallyRequest
from app.dto.common import BaseModelResponse
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages


class UpdateDocumentPartiallyInteractor:
    
    def __init__(
        self,
        uow: UnitOfWork,
        documents_repository: DocumentsRepository,
    ):
        self.uow = uow
        self.documents_repository = documents_repository
        
    async def execute(self, id: int, request: UpdateDocumentPartiallyRequest) -> BaseModelResponse:
        document = await self.documents_repository.get_one(id=id)
        if not document:
            raise AppError(status_code=404, message=ErrorMessages.DOCUMENT_NOT_FOUND)
        await self.documents_repository.update(id, request.model_dump(exclude_unset=True))
        await self.uow.commit()
        return BaseModelResponse.model_validate(document)