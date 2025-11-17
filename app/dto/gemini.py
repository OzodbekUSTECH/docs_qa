from typing import Optional, List
from pydantic import BaseModel


class GeminiUploadFileRequest(BaseModel):
    """Request for uploading a file to Gemini File Search store."""
    display_name: Optional[str] = None
    chunking_config: Optional[dict] = None
    custom_metadata: Optional[List[dict]] = None


class GeminiChatMessageRequest(BaseModel):
    """Request for sending a chat message."""
    prompt: str
    model: str = "gemini-2.5-flash"
    session_id: Optional[str] = None
    document_ids: List[int] = []  # For compatibility, not used in File Search
    history: Optional[List[dict]] = None  # Chat history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]


class GeminiChatSessionResponse(BaseModel):
    """Response for chat session."""
    session_id: str
    messages: List[dict] = []


class GeminiFileResponse(BaseModel):
    """Response for file information."""
    name: str
    display_name: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    expiration_time: Optional[str] = None
    sha256_hash: Optional[str] = None
    uri: Optional[str] = None


class GeminiDocumentResponse(BaseModel):
    """Response for document in File Search store."""
    name: str
    display_name: Optional[str] = None
    mime_type: Optional[str] = None
    create_time: Optional[str] = None
    update_time: Optional[str] = None
    custom_metadata: Optional[List[dict]] = None


class GeminiChunkResponse(BaseModel):
    """Response for chunk content from File Search store."""
    name: str
    data: Optional[str] = None
    custom_metadata: Optional[List[dict]] = None

