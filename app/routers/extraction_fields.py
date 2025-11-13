from typing import Annotated
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter, status, Query
from dishka.integrations.fastapi import FromDishka
from fastapi.responses import FileResponse
from app.interactors.extraction_fields.delete import DeleteExtractionFieldInteractor
from app.dto.extraction_fields import CreateExtractionFieldRequest, GetExtractionFieldsParams, UpdateExtractionFieldRequest
from app.interactors.extraction_fields.create import CreateExtractionFieldInteractor
from app.interactors.extraction_fields.get import GetAllExtractionFieldsInteractor, GetExtractionFieldByIdInteractor
from app.interactors.extraction_fields.update import UpdateExtractionFieldInteractor
import os

router = APIRouter(
    prefix="/extraction-fields",
    tags=["Extraction Fields"],
    route_class=DishkaRoute,
)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_extraction_field(
    request: CreateExtractionFieldRequest,
    create_extraction_field_interactor: FromDishka[CreateExtractionFieldInteractor],
):
    extraction_field = await create_extraction_field_interactor.execute(request)
    return extraction_field



@router.get('/')
async def get_extraction_fields(
    request: Annotated[GetExtractionFieldsParams, Query()],
    get_extraction_fields_interactor: FromDishka[GetAllExtractionFieldsInteractor],
):
    return await get_extraction_fields_interactor.execute(request)


@router.get('/{id}')
async def get_extraction_field_by_id(
    id: int,
    get_extraction_field_by_id_interactor: FromDishka[GetExtractionFieldByIdInteractor],
):
    return await get_extraction_field_by_id_interactor.execute(id)


@router.delete('/{id}')
async def delete_extraction_field(
    id: int,
    delete_extraction_field_interactor: FromDishka[DeleteExtractionFieldInteractor],
):
    return await delete_extraction_field_interactor.execute(id)


@router.patch('/{id}')
async def update_extraction_field(
    id: int,
    request: UpdateExtractionFieldRequest,
    update_extraction_field_interactor: FromDishka[UpdateExtractionFieldInteractor],
):
    return await update_extraction_field_interactor.execute(id, request)


@router.get('', include_in_schema=False)
async def extraction_fields_page():
    """Страница управления полями извлечения"""
    html_path = os.path.join("templates", "extraction_fields.html")
    return FileResponse(html_path)