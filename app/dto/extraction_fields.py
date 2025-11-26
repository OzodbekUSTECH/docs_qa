from pydantic import BaseModel
from typing import Optional, List
from app.entities.extracion_fields import ExtractionField
from app.utils.enums import ExtractionFieldType, DocumentType, FieldOccurrence, ExtractionBy
from app.dto.common import BaseModelResponse, TimestampResponse
from app.dto.pagination import PaginationRequest

class CreateExtractionFieldRequest(BaseModel):
    name: str
    short_description: Optional[str] = None
    type: ExtractionFieldType
    document_types: List[DocumentType]
    use_ai: bool = False
    keywords: Optional[List[str]] = None
    prompt: Optional[str] = None
    examples: Optional[List[str]] = None
    identifier: Optional[str] = None
    occurrence: FieldOccurrence = FieldOccurrence.OPTIONAL_ONCE
    extraction_by: ExtractionBy = ExtractionBy.DOCUMENT_AI
    


class UpdateExtractionFieldRequest(BaseModel):
    name: Optional[str] = None
    short_description: Optional[str] = None
    type: Optional[ExtractionFieldType] = None
    document_types: Optional[List[DocumentType]] = None
    use_ai: Optional[bool] = None
    keywords: Optional[List[str]] = None
    prompt: Optional[str] = None
    examples: Optional[List[str]] = None
    identifier: Optional[str] = None
    occurrence: Optional[FieldOccurrence] = None
    extraction_by: Optional[ExtractionBy] = None
    
class ExtractionFieldListResponse(BaseModelResponse, TimestampResponse):
    name: str
    short_description: Optional[str] = None
    type: ExtractionFieldType
    document_types: List[DocumentType]
    use_ai: bool
    occurrence: FieldOccurrence
    identifier: Optional[str] = None
    extraction_by: ExtractionBy
    
class ExtractionFieldResponse(ExtractionFieldListResponse):
    keywords: List[str]
    prompt: Optional[str]
    examples: List[str]
    
class GetExtractionFieldsParams(PaginationRequest):
    name: Optional[str] = None
    type: Optional[ExtractionFieldType] = None
    document_types: Optional[List[DocumentType]] = None
    use_ai: Optional[bool] = None
    identifier: Optional[str] = None
    occurrence: Optional[FieldOccurrence] = None
    
    class Constants:
        filter_map = {
            "name": lambda value: ExtractionField.name.ilike(f"%{value}%"),
            "type": lambda value: ExtractionField.type == value,
            "document_types": lambda value: ExtractionField.document_types.contains(value),
            "use_ai": lambda value: ExtractionField.use_ai == value,
            "occurrence": lambda value: ExtractionField.occurrence == value,
        }
        searchable_fields = ["name"]
        orderable_fields = {
            "name": ExtractionField.name,
            "type": ExtractionField.type,
            "document_types": ExtractionField.document_types,
        }
