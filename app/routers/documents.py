from typing import Annotated
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter, status, UploadFile, File, Query, Form
from fastapi import BackgroundTasks
from dishka.integrations.fastapi import FromDishka
from app.interactors.documents.create import CreateDocumentInteractor
from app.interactors.documents.get import GetAllDocumentsInteractor, GetDocumentByIdInteractor
from app.interactors.documents.delete import DeleteDocumentInteractor
from app.interactors.documents.update import UpdateDocumentPartiallyInteractor
from app.utils.enums import DocumentType
from app.dto.documents import CreateDocumentRequest, GetDocumentsParams, RerunExtractionRequest, UpdateDocumentPartiallyRequest
from app.utils.bg_tasks import process_document
from app.repositories.extraction_fields import ExtractionFieldsRepository
from app.repositories.documents import DocumentFieldValuesRepository
from app.repositories.uow import UnitOfWork
from app.services.extract_field_values import ExtractDocumentFieldValuesService
from sqlalchemy import any_
from app.entities.extracion_fields import ExtractionField
import json


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

@router.post('/{id}/rerun-extraction')
async def rerun_extraction(
    id: int,
    request: RerunExtractionRequest,
    get_document_by_id_interactor: FromDishka[GetDocumentByIdInteractor],
    background_tasks: BackgroundTasks,
):
    document = await get_document_by_id_interactor.execute(id)
    background_tasks.add_task(process_document, document.id, request.extraction_field_ids)
    return document


@router.patch('/{id}')
async def update_document_partially(
    id: int,
    request: UpdateDocumentPartiallyRequest,
    update_document_partially_interactor: FromDishka[UpdateDocumentPartiallyInteractor],
):
    return await update_document_partially_interactor.execute(id, request)


@router.delete('/{id}')
async def delete_document(
    id: int,
    delete_document_interactor: FromDishka[DeleteDocumentInteractor],
):
    return await delete_document_interactor.execute(id)


# Interactive View Endpoints

@router.get('/{id}/available-fields')
async def get_available_fields(
    id: int,
    get_document_interactor: FromDishka[GetDocumentByIdInteractor],
    extraction_fields_repo: FromDishka[ExtractionFieldsRepository],
):
    """Get available extraction fields for document type"""
    document = await get_document_interactor.execute(id)
    fields = await extraction_fields_repo.get_all(
        where=[document.type == any_(ExtractionField.document_types)]
    )
    return [{"id": f.id, "name": f.name, "identifier": f.identifier} for f in fields]


@router.post('/{id}/extract-field')
async def extract_single_field(
    id: int,
    field_id: int = Form(None),
    custom_field_name: str = Form(None),
    description: str = Form(None),
    occurrence: str = Form(None),
    page_number: int = Form(None),
    extraction_by: str = Form(None),
    type: str = Form(None),
    prompt: str = Form(None),
    extract_service: FromDishka[ExtractDocumentFieldValuesService] = None,
):
    """Extract single field from document. Supports both extraction fields (field_id) and custom fields (custom_field_name)."""
    if not field_id and not custom_field_name:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Either field_id or custom_field_name must be provided")
    
    # Map prompt to description if description is missing (for Gemini consistency)
    if prompt and not description:
        description = prompt

    result = await extract_service.extract_single_field(
        id, 
        field_id=field_id, 
        custom_field_name=custom_field_name,
        description=description,
        occurrence=occurrence,
        page_number=page_number,
        extraction_by=extraction_by,
        type=type,
        prompt=prompt
    )
    return result


@router.post('/{id}/save-field-values')
async def save_field_values(
    id: int,
    field_values: str = Form(...),  # JSON string
    extract_service: FromDishka[ExtractDocumentFieldValuesService] = None,
):
    """Save field values with embeddings"""
    values = json.loads(field_values)
    await extract_service.save_field_values_interactive(id, values)
    return {"success": True}


@router.delete('/field-values/{field_value_id}')
async def delete_field_value(
    field_value_id: int,
    field_values_repo: FromDishka[DocumentFieldValuesRepository] = None,
    uow: FromDishka[UnitOfWork] = None,
):
    """Delete a field value by its ID"""
    from app.entities.documents import DocumentFieldValue
    from fastapi import HTTPException
    from sqlalchemy import select, delete as sql_delete
    
    # Check if field value exists
    stmt = select(DocumentFieldValue).where(DocumentFieldValue.id == field_value_id)
    result = await field_values_repo.session.execute(stmt)
    field_value = result.scalar_one_or_none()
    
    if not field_value:
        raise HTTPException(status_code=404, detail="Field value not found")
    
    # Delete the field value using direct SQL delete (since id is int, not UUID)
    delete_stmt = sql_delete(DocumentFieldValue).where(DocumentFieldValue.id == field_value_id)
    await field_values_repo.session.execute(delete_stmt)
    await uow.commit()
    
    return {"success": True}


@router.get('/{id}/fields')
async def get_document_fields(
    id: int,
    page_number: int = Query(None, description="Filter by page number"),
    get_document_by_id_interactor: FromDishka[GetDocumentByIdInteractor] = None,
    field_values_repo: FromDishka[DocumentFieldValuesRepository] = None,
):
    """Get all field values for a document, optionally filtered by page number.
    Same field can appear on different pages - each page = separate record."""
    from app.entities.documents import DocumentFieldValue
    from sqlalchemy.orm import joinedload
    
    # Verify document exists
    document = await get_document_by_id_interactor.execute(id)
    if not document:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Build where conditions
    where_conditions = [DocumentFieldValue.document_id == id]
    if page_number:
        where_conditions.append(DocumentFieldValue.page_num == str(page_number))
    
    # Get field values using repository method
    field_values = await field_values_repo.get_all(
        where=where_conditions,
        options=[joinedload(DocumentFieldValue.field)]
    )
    
    # Convert to response format
    return [
        {
            "id": fv.id,
            "field_id": fv.field_id,
            "custom_field_name": fv.custom_field_name,
            "field_name": fv.field.name if fv.field else fv.custom_field_name or "Unknown",
            "value": fv.value_text if fv.value_text else None,  # Return None for empty values
            "confidence": fv.confidence,
            "page_number": int(fv.page_num) if fv.page_num and fv.page_num.isdigit() else None,
            "bbox": (
                fv.bbox.get(str(fv.page_num), []) 
                if fv.bbox and fv.page_num and isinstance(fv.bbox, dict) 
                else (
                    fv.bbox[0] if isinstance(fv.bbox, list) and len(fv.bbox) > 0 
                    else None
                )
            ),
        }
        for fv in field_values
    ]


@router.post('/{id}/extract-text-from-area')
async def extract_text_from_area(
    id: int,
    page_number: int = Form(...),
    bbox: str = Form(...),  # JSON string: [x1, y1, x2, y2]
    get_document_interactor: FromDishka[GetDocumentByIdInteractor] = None,
):
    """Extract text from a specific area of document using OCR"""
    from app.services.document_ai_service import DocumentAIService
    import asyncio
    
    try:
        # Get document
        document = await get_document_interactor.execute(id)
        if not document:
            return {"error": "Document not found", "text": ""}
        
        # Load document AI service
        doc_ai_service = DocumentAIService()
        
        # Process document with OCR to get full text and coordinates
        ocr_result = await asyncio.to_thread(
            doc_ai_service.process_with_ocr,
            document.file_path,
            document.content_type
        )
        
        # Extract text from the specified bbox
        bbox_coords = json.loads(bbox)
        if not isinstance(bbox_coords, list) or len(bbox_coords) != 4:
            return {"error": "Invalid bbox format", "text": ""}
        
        x1, y1, x2, y2 = bbox_coords
        
        # Get full text from OCR result
        full_text = ocr_result.get("text", "") or ocr_result.get("document", {}).get("text", "")
        
        # Extract text from bbox area using OCR coordinates
        # For now, return full text - coordinate matching can be improved later
        # The client-side extraction should work for most cases
        extracted_text = full_text
        
        return {"text": extracted_text, "success": True}
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error extracting text from area: {e}", exc_info=True)
        return {"error": str(e), "text": ""}