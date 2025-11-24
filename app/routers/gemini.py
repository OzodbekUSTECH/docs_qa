from typing import List, Optional
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter, status, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import StreamingResponse
from dishka.integrations.fastapi import FromDishka
from pathlib import Path
import tempfile
import os
import urllib.parse
import logging

from app.interactors.documents.gemini_file_upload import GeminiFileUploadInteractor
from app.interactors.chat.gemini_chat import GeminiChatInteractor
from app.dto.gemini import (
    GeminiUploadFileRequest,
    GeminiChatMessageRequest,
    GeminiFileResponse,
    GeminiDocumentResponse,
    GeminiChunkResponse,
)
from app.dto.chat import GenerateChatRequest


router = APIRouter(
    prefix="/gemini",
    tags=["Gemini"],
    route_class=DishkaRoute,
)


# ==================== File Operations ====================

@router.post("/files/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(
    file_upload_interactor: FromDishka[GeminiFileUploadInteractor],
    file: UploadFile = File(...),
    display_name: str = Form(None),
    chunking_mode: str = Form("auto", description="auto or custom"),
    max_tokens_per_chunk: Optional[int] = Form(None),
    max_overlap_tokens: Optional[int] = Form(None),
):
    """
    Upload a file to Gemini File Search store.
    The file will be indexed and available for RAG queries.
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    # Build chunking configuration if requested
    chunking_config = None
    chunking_mode = (chunking_mode or "auto").lower()
    if chunking_mode not in {"auto", "custom"}:
        raise HTTPException(status_code=400, detail="chunking_mode must be 'auto' or 'custom'")
    
    if chunking_mode == "custom":
        if max_tokens_per_chunk is None or max_tokens_per_chunk <= 0:
            raise HTTPException(
                status_code=400,
                detail="max_tokens_per_chunk must be a positive integer when chunking_mode='custom'",
            )
        if max_overlap_tokens is not None and max_overlap_tokens < 0:
            raise HTTPException(
                status_code=400,
                detail="max_overlap_tokens must be zero or positive when chunking_mode='custom'",
            )
        if max_overlap_tokens is not None and max_overlap_tokens >= max_tokens_per_chunk:
            raise HTTPException(
                status_code=400,
                detail="max_overlap_tokens must be smaller than max_tokens_per_chunk",
            )
        
        chunking_config = {
            "white_space_config": {
                "max_tokens_per_chunk": max_tokens_per_chunk,
                "max_overlap_tokens": max_overlap_tokens or 0,
            }
        }
    
    try:
        # Upload to File Search store
        file_name = await file_upload_interactor.upload_to_file_search_store(
            file_path=tmp_path,
            display_name=display_name or file.filename,
            chunking_config=chunking_config,
        )
        return {
            "message": "File uploaded successfully",
            "file_name": file_name,
            "original_filename": file.filename,
        }
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/files", response_model=List[GeminiFileResponse])
async def list_files(
    file_upload_interactor: FromDishka[GeminiFileUploadInteractor],
):
    """List all files in Gemini Files API."""
    files = await file_upload_interactor.list_files()
    return [
        GeminiFileResponse(
            name=file.name,
            display_name=getattr(file, 'display_name', None),
            mime_type=getattr(file, 'mime_type', None),
            size_bytes=getattr(file, 'size_bytes', None),
            create_time=str(getattr(file, 'create_time', None)) if hasattr(file, 'create_time') else None,
            update_time=str(getattr(file, 'update_time', None)) if hasattr(file, 'update_time') else None,
            expiration_time=str(getattr(file, 'expiration_time', None)) if hasattr(file, 'expiration_time') else None,
            sha256_hash=getattr(file, 'sha256_hash', None),
            uri=getattr(file, 'uri', None),
        )
        for file in files
    ]


@router.get("/files/{file_name}", response_model=GeminiFileResponse)
async def get_file(
    file_name: str,
    file_upload_interactor: FromDishka[GeminiFileUploadInteractor],
):
    """Get file information by name."""
    try:
        file = await file_upload_interactor.get_file(file_name)
        return GeminiFileResponse(
            name=file.name,
            display_name=getattr(file, 'display_name', None),
            mime_type=getattr(file, 'mime_type', None),
            size_bytes=getattr(file, 'size_bytes', None),
            create_time=str(getattr(file, 'create_time', None)) if hasattr(file, 'create_time') else None,
            update_time=str(getattr(file, 'update_time', None)) if hasattr(file, 'update_time') else None,
            expiration_time=str(getattr(file, 'expiration_time', None)) if hasattr(file, 'expiration_time') else None,
            sha256_hash=getattr(file, 'sha256_hash', None),
            uri=getattr(file, 'uri', None),
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")


@router.delete("/files/{file_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_name: str,
    file_upload_interactor: FromDishka[GeminiFileUploadInteractor],
):
    """Delete a file by name."""
    try:
        await file_upload_interactor.delete_file(file_name)
        return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")


# ==================== File Search Store Documents Operations ====================

@router.get("/documents", response_model=List[GeminiDocumentResponse])
async def list_documents(
    file_upload_interactor: FromDishka[GeminiFileUploadInteractor],
):
    """List all documents in the File Search store."""
    documents = await file_upload_interactor.list_documents_in_store()
    return [
        GeminiDocumentResponse(
            name=doc.name,
            display_name=getattr(doc, 'display_name', None),
            mime_type=getattr(doc, 'mime_type', None),
            create_time=str(getattr(doc, 'create_time', None)) if hasattr(doc, 'create_time') else None,
            update_time=str(getattr(doc, 'update_time', None)) if hasattr(doc, 'update_time') else None,
            custom_metadata=getattr(doc, 'custom_metadata', None),
        )
        for doc in documents
    ]


@router.get("/documents/{document_name:path}", response_model=GeminiDocumentResponse)
async def get_document(
    document_name: str,
    file_upload_interactor: FromDishka[GeminiFileUploadInteractor],
):
    """Get document information from File Search store."""
    try:
        # Decode URL-encoded characters
        document_name = urllib.parse.unquote(document_name)
        
        doc = await file_upload_interactor.get_document_in_store(document_name)
        return GeminiDocumentResponse(
            name=doc.name,
            display_name=getattr(doc, 'display_name', None),
            mime_type=getattr(doc, 'mime_type', None),
            create_time=str(getattr(doc, 'create_time', None)) if hasattr(doc, 'create_time') else None,
            update_time=str(getattr(doc, 'update_time', None)) if hasattr(doc, 'update_time') else None,
            custom_metadata=getattr(doc, 'custom_metadata', None),
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Document not found: {str(e)}")


@router.delete("/documents/{document_name:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_name: str,
    file_upload_interactor: FromDishka[GeminiFileUploadInteractor],
    force: bool = Query(True, description="Force delete even if document contains chunks"),
):
    """Delete a document from File Search store."""
    logger = logging.getLogger(__name__)
    try:
        # Decode URL-encoded characters
        document_name = urllib.parse.unquote(document_name)
        logger.info(f"Attempting to delete document: {document_name}, force={force}")
        
        await file_upload_interactor.delete_document_from_store(document_name, force=force)
        logger.info(f"Successfully deleted document: {document_name}")
        return None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error deleting document {document_name}: {error_msg}", exc_info=True)
        
        # Check for common error types
        if "not found" in error_msg.lower() or "404" in error_msg:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_name}")
        elif "FAILED_PRECONDITION" in error_msg or "non-empty" in error_msg.lower():
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot delete document: it contains chunks. Use force=true to delete anyway. Error: {error_msg}"
            )
        elif "400" in error_msg or "bad request" in error_msg.lower():
            raise HTTPException(
                status_code=400, 
                detail=f"Bad request when deleting document: {error_msg}. Document name: {document_name}"
            )
        else:
            raise HTTPException(status_code=500, detail=f"Error deleting document: {error_msg}")


@router.get("/chunks/{chunk_name:path}", response_model=GeminiChunkResponse)
async def get_chunk(
    chunk_name: str,
    file_upload_interactor: FromDishka[GeminiFileUploadInteractor],
):
    """Get chunk content from File Search store."""
    try:
        # Decode URL-encoded characters
        chunk_name = urllib.parse.unquote(chunk_name)
        
        chunk = await file_upload_interactor.get_chunk_content(chunk_name)
        if chunk is None:
            raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_name}")
        
        return GeminiChunkResponse(
            name=chunk['name'],
            data=chunk.get('data'),
            custom_metadata=chunk.get('custom_metadata'),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Chunk not found: {str(e)}")


# ==================== Chat Operations ====================

@router.post("/chat/stream")
async def stream_chat(
    request: GeminiChatMessageRequest,
    chat_interactor: FromDishka[GeminiChatInteractor],
):
    """
    Stream chat response using Gemini with File Search.
    Creates a new chat session or continues existing one.
    """
    # Convert to GenerateChatRequest for compatibility
    # Create a custom request object that includes history
    class ChatRequestWithHistory:
        def __init__(self, prompt, model, session_id, document_ids, history):
            self.prompt = prompt
            self.model = model
            self.session_id = session_id
            self.document_ids = document_ids
            self.history = history
    
    chat_request = ChatRequestWithHistory(
        prompt=request.prompt,
        model=request.model,
        session_id=request.session_id,
        document_ids=request.document_ids,
        history=request.history,
    )
    
    return StreamingResponse(
        chat_interactor.stream(chat_request),
        media_type="text/event-stream",
    )


@router.post("/chat/generate")
async def generate_chat(
    request: GeminiChatMessageRequest,
    chat_interactor: FromDishka[GeminiChatInteractor],
):
    """
    Generate a single chat response (non-streaming) using Gemini with File Search.
    """
    # Convert to GenerateChatRequest for compatibility
    chat_request = GenerateChatRequest(
        prompt=request.prompt,
        model=request.model,
        session_id=request.session_id,
        document_ids=request.document_ids,
    )
    
    response_text = await chat_interactor.generate(chat_request)
    return {"response": response_text}


@router.get("/chat/models")
async def get_chat_models():
    """Get available Gemini models for chat."""
    return {
        "models": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
        "default": "gemini-2.5-flash",
    }


# ==================== File Search Store Operations ====================

@router.get("/stores")
async def list_file_search_stores(
    file_upload_interactor: FromDishka[GeminiFileUploadInteractor],
):
    """List all File Search stores."""
    stores = await file_upload_interactor.list_file_search_stores()
    return [
        {
            "name": store.name,
            "display_name": getattr(store, 'display_name', None),
            "create_time": str(getattr(store, 'create_time', None)) if hasattr(store, 'create_time') else None,
            "update_time": str(getattr(store, 'update_time', None)) if hasattr(store, 'update_time') else None,
            "active_documents_count": getattr(store, 'active_documents_count', None),
            "pending_documents_count": getattr(store, 'pending_documents_count', None),
            "failed_documents_count": getattr(store, 'failed_documents_count', None),
            "size_bytes": getattr(store, 'size_bytes', None),
        }
        for store in stores
    ]


@router.delete("/stores", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file_search_store(
    file_upload_interactor: FromDishka[GeminiFileUploadInteractor],
    store_name: str = Query(..., description="Full store name (e.g., fileSearchStores/documents-xxx)"),
    force: bool = Query(True),
):
    """Delete a File Search store by name."""
    logger = logging.getLogger(__name__)
    try:
        # Decode URL-encoded characters
        store_name = urllib.parse.unquote(store_name)
        logger.info(f"Attempting to delete store: {store_name}")
        
        # Ensure store_name is in correct format if needed
        if not store_name.startswith('fileSearchStores/'):
            store_name = f"fileSearchStores/{store_name}"
        
        logger.info(f"Final store name for deletion: {store_name}")
        
        await file_upload_interactor.client.file_search_stores.delete(
            name=store_name,
            config={'force': force}
        )
        logger.info(f"Successfully deleted store: {store_name}")
        return None
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error deleting store {store_name}: {error_msg}", exc_info=True)
        # Check for common error types
        if "not found" in error_msg.lower() or "404" in error_msg:
            raise HTTPException(status_code=404, detail=f"Store not found: {store_name}")
        elif "failed_precondition" in error_msg.lower() or "contains documents" in error_msg.lower():
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot delete store: it contains documents. Use force=true to delete anyway."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Error deleting store: {error_msg}")


@router.get("/stores/current")
async def get_current_file_search_store(
    file_upload_interactor: FromDishka[GeminiFileUploadInteractor],
):
    """Get current File Search store information."""
    store_name = await file_upload_interactor.get_file_search_store_name()
    return {"store_name": store_name}


@router.get("", include_in_schema=False)
async def gemini_page():
    """Gemini File Search & Chat page."""
    from fastapi.responses import FileResponse
    import os
    gemini_html_path = os.path.join("templates", "gemini.html")
    return FileResponse(gemini_html_path)

