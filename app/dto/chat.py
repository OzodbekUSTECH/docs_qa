from typing import Optional, List

from pydantic import BaseModel


class GenerateChatRequest(BaseModel):
    document_ids: list[int]
    prompt: str
    model: str = "gpt-4o"
    session_id: Optional[str] = None




# Новые DTO для поиска по field values
class MatchedFieldValueResponse(BaseModel):
    """DTO для найденного значения поля"""
    id: int
    field_id: int
    field_name: str
    value_text: str
    confidence: Optional[float]
    page_num: Optional[int]
    bbox: Optional[List[float]]
    text_rank: float
    vector_score: float
    hybrid_score: float


class OtherFieldValueResponse(BaseModel):
    """DTO для других полей документа (не найденных в поиске) - только метаданные для понимания доступных полей"""
    id: int
    field_id: int
    field_name: str
    short_description: Optional[str] = None


class DocumentSearchResultResponse(BaseModel):
    """DTO для результата поиска по документу"""
    document_id: int
    filename: str
    file_path: str
    content_type: str
    document_type: str
    status: str
    summary: Optional[str] = None
    matched_fields: List[MatchedFieldValueResponse]
    other_fields: List[OtherFieldValueResponse]
    max_hybrid_score: float  # Максимальный score среди найденных полей


class FieldValueSearchResponse(BaseModel):
    """DTO для ответа поиска по field values"""
    documents: List[DocumentSearchResultResponse]
    