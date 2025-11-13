from pydantic import BaseModel
from typing import Optional, List
from app.entities.extracion_fields import ExtractionField
from app.utils.enums import ExtractionFieldType, DocumentType
from app.dto.common import BaseModelResponse, TimestampResponse
from app.dto.pagination import PaginationRequest

class CreateExtractionFieldRequest(BaseModel):
    name: str
    identifier: str
    short_description: Optional[str] = None
    type: ExtractionFieldType
    document_types: List[DocumentType]
    prompt: str
    examples: List[str]
    

from pydantic import model_validator

class UpdateExtractionFieldRequest(BaseModel):
    name: Optional[str] = None
    identifier: Optional[str] = None
    short_description: Optional[str] = None
    type: Optional[ExtractionFieldType] = None
    document_types: Optional[List[DocumentType]] = None
    prompt: Optional[str] = None
    examples: Optional[List[str]] = None

    @model_validator(mode="before")
    def check_name_identifier(cls, values):
        name = values.get("name")
        identifier = values.get("identifier")
        if name is not None and identifier is None:
            raise ValueError("If 'name' is provided, 'identifier' must also be provided.")
        return values
    
class ExtractionFieldListResponse(BaseModelResponse, TimestampResponse):
    name: str
    short_description: Optional[str] = None
    type: ExtractionFieldType
    document_types: List[DocumentType]
    
class ExtractionFieldResponse(ExtractionFieldListResponse):
    prompt: str
    examples: List[str]
    
    
class GetExtractionFieldsParams(PaginationRequest):
    name: Optional[str] = None
    type: Optional[ExtractionFieldType] = None
    document_types: Optional[List[DocumentType]] = None
    
    class Constants:
        filter_map = {
            "name": lambda value: ExtractionField.name.ilike(f"%{value}%"),
            "type": lambda value: ExtractionField.type == value,
            "document_types": lambda value: ExtractionField.document_types.contains(value),
        }
        searchable_fields = ["name"]
        orderable_fields = {
            "name": ExtractionField.name,
            "type": ExtractionField.type,
            "document_types": ExtractionField.document_types,
        }
