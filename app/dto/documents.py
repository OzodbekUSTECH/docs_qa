from typing import Optional
from fastapi import UploadFile
from pydantic import BaseModel, field_validator
from app.dto.common import BaseModelResponse, TimestampResponse
from app.entities.documents import Document
from app.utils.enums import DocumentStatus, DocumentType, ExtractionFieldType
from app.dto.pagination import PaginationRequest



class RerunExtractionRequest(BaseModel):
    extraction_field_ids: list[int]

class CreateDocumentRequest(BaseModel):
    file: UploadFile
    type: DocumentType
    extraction_field_ids: list[int]
    
    
class UpdateDocumentPartiallyRequest(BaseModel):
    type: Optional[DocumentType] = None
    
    
    
    
class BaseDocumentResponse(BaseModelResponse, TimestampResponse):
    filename: str
    file_path: str
    content_type: str
    type: DocumentType
    status: DocumentStatus
    
    @field_validator('file_path', mode='before')
    @classmethod
    def normalize_file_path(cls, v: str) -> str:
        """Нормализует путь к файлу: заменяет обратные слеши на прямые для веб-URL"""
        if isinstance(v, str):
            # Заменяем обратные слеши на прямые
            return v.replace('\\', '/')
        return v
    
class DocumentListResponse(BaseDocumentResponse):
    pass



class ExtractionFieldResponse(BaseModelResponse):
    name: str
    short_description: Optional[str] = None

class DocumentFieldValueResponse(BaseModelResponse):
    value_text: str
    confidence: Optional[float] = None
    page_num: Optional[int] = None
    bbox: Optional[list[float]] = None
    field: ExtractionFieldResponse

class DocumentResponse(BaseDocumentResponse):
    field_values: list[DocumentFieldValueResponse]



class GetDocumentsParams(PaginationRequest):
    
    filename: Optional[str] = None
    
    
    class Constants:
        filter_map = {
            "filename": lambda value: Document.filename.ilike(f"%{value}%"),
        }