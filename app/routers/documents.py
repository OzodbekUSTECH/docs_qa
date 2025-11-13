from typing import Annotated
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter, status, UploadFile, File, Query, Form
from fastapi import BackgroundTasks
from dishka.integrations.fastapi import FromDishka
from app.interactors.documents.create import CreateDocumentInteractor
from app.interactors.documents.get import GetAllDocumentsInteractor, GetDocumentByIdInteractor
from app.interactors.documents.delete import DeleteDocumentInteractor
from app.utils.enums import DocumentType
from app.dto.documents import CreateDocumentRequest, GetDocumentsParams
from app.utils.bg_tasks import process_document



router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    route_class=DishkaRoute,
)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_document(
    create_document_interactor: FromDishka[CreateDocumentInteractor],
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    type: DocumentType = Form(...),
    extraction_field_ids: list[int] = Form(...),
):
    request = CreateDocumentRequest(file=file, type=type, extraction_field_ids=extraction_field_ids)
    document = await create_document_interactor.execute(request)
    background_tasks.add_task(process_document, document.id, extraction_field_ids)
    return document



@router.get('/')
async def get_documents(
    request: Annotated[GetDocumentsParams, Query()],
    get_documents_interactor: FromDishka[GetAllDocumentsInteractor],
):
    return await get_documents_interactor.execute(request)


@router.get('/{id}')
async def get_document_by_id(
    id: int,
    get_document_by_id_interactor: FromDishka[GetDocumentByIdInteractor],
):
    return await get_document_by_id_interactor.execute(id)


@router.delete('/{id}')
async def delete_document(
    id: int,
    delete_document_interactor: FromDishka[DeleteDocumentInteractor],
):
    return await delete_document_interactor.execute(id)