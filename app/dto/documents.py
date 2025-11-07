from typing import Optional
from fastapi import UploadFile
from pydantic import BaseModel
from app.dto.common import BaseModelResponse, TimestampResponse
from app.entities.documents import Document
from app.utils.enums import DocumentStatus, DocumentType
from app.dto.pagination import PaginationRequest


class CreateDocumentRequest(BaseModel):
    file: UploadFile
    type: DocumentType
    
    
    
    
class BaseDocumentResponse(BaseModelResponse, TimestampResponse):
    filename: str
    content_type: str
    type: DocumentType
    status: DocumentStatus
    
class DocumentListResponse(BaseDocumentResponse):
    pass

class DocumentResponse(BaseDocumentResponse):
    content: str



class GetDocumentsParams(PaginationRequest):
    
    filename: Optional[str] = None
    
    
    class Constants:
        filter_map = {
            "filename": lambda value: Document.filename.ilike(f"%{value}%"),
        }