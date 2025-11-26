from uuid import UUID
import os

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

from app.interactors.chat.gemini_agent import GeminiAgentInteractor
from app.dto.chat import GenerateChatRequest
from app.repositories.chat import ChatRepository

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    route_class=DishkaRoute,
)


@router.post("/generate")
async def generate_chat(
    generate_chat_interactor: FromDishka[GeminiAgentInteractor],
    request: GenerateChatRequest,
):
    return StreamingResponse(
        generate_chat_interactor.stream(request),
        media_type="text/event-stream",
    )


@router.get("/models")
async def get_models():
    return ["gemini-1.5-flash"]


@router.get("/sessions")
async def list_sessions(
    chat_repository: FromDishka[ChatRepository],
):
    """List all chat sessions"""
    sessions = await chat_repository.list_sessions()
    return [
        {
            "id": str(s.id),
            "title": s.title or "New Chat",
            "created_at": s.created_at,
            "updated_at": s.updated_at,
        }
        for s in sessions
    ]


@router.post("/sessions")
async def create_session(
    chat_repository: FromDishka[ChatRepository],
):
    """Create a new chat session"""
    session = await chat_repository.create_session()
    return {"id": str(session.id)}


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    chat_repository: FromDishka[ChatRepository],
):
    """Delete a chat session"""
    try:
        uuid_obj = UUID(session_id)
        await chat_repository.delete_session(uuid_obj)
        return {"message": "Session deleted successfully", "session_id": session_id}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID")


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    chat_repository: FromDishka[ChatRepository],
):
    """Get messages for a chat session"""
    try:
        uuid_obj = UUID(session_id)
        session = await chat_repository.get_session(uuid_obj)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages_list = []
        for msg in session.messages:
            # JSONB fields should already be dicts from SQLAlchemy, but ensure they're serializable
            citations = msg.citations if msg.citations is not None else None
            search_results = msg.search_results if msg.search_results is not None else None
            thinking_process = msg.thinking_process if msg.thinking_process is not None else None
            
            messages_list.append({
                "role": msg.role,
                "content": msg.content,
                "citations": citations,
                "search_results": search_results,
                "thinking_process": thinking_process,
                "created_at": msg.created_at,
            })
        
        return messages_list
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID")


@router.get("", include_in_schema=False)
async def chat_page():
    """Chat Page"""
    chat_html_path = os.path.join("templates", "chat.html")
    return FileResponse(chat_html_path)