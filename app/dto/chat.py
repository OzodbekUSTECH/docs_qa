from typing import Optional

from pydantic import BaseModel


class GenerateChatRequest(BaseModel):
    document_ids: list[int]
    prompt: str
    model: str = "gpt-4o"
    session_id: Optional[str] = None


class SearchChunkResponse(BaseModel):
    """DTO для chunk результата поиска"""
    id: int
    document_id: int
    chunk_index: int
    content: str
    text_rank: float
    vector_score: float
    hybrid_score: float


class HybridSearchResponse(BaseModel):
    """DTO для ответа гибридного поиска"""
    chunks: list[SearchChunkResponse]
    