from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter, HTTPException
from dishka.integrations.fastapi import FromDishka
from app.interactors.chat.generate import GenerateChatResponseInteractor
from app.dto.chat import GenerateChatRequest
from fastapi.responses import StreamingResponse, FileResponse
import os



router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    route_class=DishkaRoute,
)


@router.post("/generate")
async def generate_chat(
    generate_chat_interactor: FromDishka[GenerateChatResponseInteractor],
    request: GenerateChatRequest,
):
    return StreamingResponse(
        generate_chat_interactor.stream(request),
        media_type="application/json",
    )


@router.get("/models")
async def get_models():
    return ["gpt-4o", "gpt-4o-mini"]


@router.get("/sessions/{session_id}/log")
async def get_session_log(
    session_id: str,
    generate_chat_interactor: FromDishka[GenerateChatResponseInteractor],
):
    log = generate_chat_interactor.get_session_log(session_id)
    if not log:
        raise HTTPException(status_code=404, detail="Session log not found")
    return log


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    generate_chat_interactor: FromDishka[GenerateChatResponseInteractor],
):
    """Удаляет сессию чата"""
    deleted = generate_chat_interactor.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted successfully", "session_id": session_id}


@router.get("", include_in_schema=False)
async def chat_page():
    """Страница чата"""
    chat_html_path = os.path.join("templates", "chat.html")
    return FileResponse(chat_html_path)