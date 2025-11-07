from fastapi import FastAPI
from app.routers.documents import router as documents_router
from app.routers.chat import router as chat_router 

all_routers = [
    documents_router,
    chat_router,
]


def register_routers(app: FastAPI, prefix: str = ""):
    """
    Initialize all routers in the app.
    """
    for router in all_routers:
        app.include_router(router, prefix=prefix)
