from fastapi import FastAPI
from app.routers.documents import router as documents_router
from app.routers.chat import router as chat_router 
from app.routers.extraction_fields import router as extraction_fields_router
from app.routers.gemini import router as gemini_router
from app.routers.document_ai import router as document_ai_router

all_routers = [
    documents_router,
    chat_router,
    extraction_fields_router,
    gemini_router,
    document_ai_router,
]


def register_routers(app: FastAPI, prefix: str = ""):
    """
    Initialize all routers in the app.
    """
    for router in all_routers:
        app.include_router(router, prefix=prefix)
