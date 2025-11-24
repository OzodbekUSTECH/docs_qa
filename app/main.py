from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.logger_conf import configure_logging
from app.di.containers import app_container
from app.exceptions.registery import register_exceptions
from app.utils.dependencies import get_current_user_for_docs
from app.middlewares.registery import register_middlewares

from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI, Depends
import os

from app.core.config import settings
from app.routers import register_routers


from agents import set_default_openai_key
configure_logging(level=settings.LOG_LEVEL)

set_default_openai_key(settings.OPENAI_API_KEY)


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # from docling.document_converter import DocumentConverter
    # await app.state.dishka_container.get(DocumentConverter)
    
    # Инициализация при запуске
    # await warmup_dependencies(app.state.dishka_container)
    # await init_qdrant_collection()
    yield
    await app.state.dishka_container.close()
    # Очистка при завершении (если нужно)


def create_app():
    app = FastAPI(
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    setup_dishka(app_container, app)
    
    register_middlewares(app)
    register_exceptions(app)
    register_routers(app)
    
    # Подключаем статические файлы (если они есть)
    app.mount("/storage", StaticFiles(directory="storage"), name="storage")
    
    # Роутер для главной страницы
    @app.get("/", include_in_schema=False)
    async def index():
        template_path = os.path.join("templates", "index.html")
        return FileResponse(template_path)
    
    # Роутер для страницы просмотра документа
    @app.get("/view", include_in_schema=False)
    async def view_document_page():
        template_path = os.path.join("templates", "view_document.html")
        return FileResponse(template_path)
   
    return app


app = create_app()


@app.get(
    "/api/docs",
    include_in_schema=False,
    dependencies=[Depends(get_current_user_for_docs)],
)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title="LLM STARTAP API",
        swagger_ui_parameters={"docExpansion": "none"},
    )


@app.get(
    "/api/openapi.json",
    include_in_schema=False,
    dependencies=[Depends(get_current_user_for_docs)],
)
async def get_open_api_endpoint():
    openapi_schema = get_openapi(
        title="LLM STARTAP API", version="1.0.0", routes=app.routes
    )
    openapi_schema["servers"] = [
        {"url": "/", "description": "Base Path for API"},
    ]
    return openapi_schema
