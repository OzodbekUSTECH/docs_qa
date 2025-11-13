from sqlalchemy.orm import joinedload, selectinload
from app.repositories.documents import DocumentsRepository
from app.dto.pagination import PaginatedResponse
from app.dto.documents import DocumentListResponse, DocumentResponse, GetDocumentsParams
from app.entities.documents import Document, DocumentFieldValue
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages




class GetAllDocumentsInteractor:
    
    
    def __init__(self, documents_repository: DocumentsRepository):
        self.documents_repository = documents_repository
        
        
    async def execute(self, request: GetDocumentsParams) -> PaginatedResponse[DocumentListResponse]:
        documents, total = await self.documents_repository.get_all_and_count(
            request_query=request,
        )
        return PaginatedResponse[DocumentListResponse](
            items=[DocumentListResponse.model_validate(document) for document in documents],
            total=total,
            page=request.page,
            size=request.size
        )
        
        
class GetDocumentByIdInteractor:
    
    def __init__(self, documents_repository: DocumentsRepository):
        self.documents_repository = documents_repository
        
        
    async def execute(self, id: int) -> DocumentResponse:
        document = await self.documents_repository.get_one(
            id,
            options=[
                selectinload(Document.field_values).options(
                    joinedload(DocumentFieldValue.field)
                )
            ]
        )
        if not document:
            raise AppError(status_code=404, message=ErrorMessages.DOCUMENT_NOT_FOUND)
        return DocumentResponse.model_validate(document)