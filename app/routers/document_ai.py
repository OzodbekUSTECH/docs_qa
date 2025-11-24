"""
Router for Document AI processing endpoints.
"""

import json
import logging
import os
from typing import Any, List, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from dishka.integrations.fastapi import DishkaRoute, FromDishka

from app.services.document_ai_service import DocumentAIService
from app.services.extract_field_values import ExtractDocumentFieldValuesService
from app.services.gemini_ocr_service import GeminiOCRService
from app.interactors.documents.get import GetDocumentByIdInteractor
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/document-ai",
    tags=["Document AI"],
    route_class=DishkaRoute,
)

def _parse_field_definitions(field_names: Optional[str], field_schema: Optional[str]) -> List[Any]:
    schema_definitions: List[Any] = []
    if field_schema:
        try:
            schema_data = json.loads(field_schema)
            if isinstance(schema_data, list):
                schema_definitions = [
                    item for item in schema_data
                    if isinstance(item, (dict, str))
                ]
            else:
                raise ValueError("field_schema must be a JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid field_schema payload: {exc}"
            )
    if schema_definitions:
        return schema_definitions
    
    if field_names:
        fields = [f.strip() for f in field_names.split(",") if f.strip()]
        if fields:
            return fields
    
    return []

@router.get("", include_in_schema=False)
async def document_ai_page():
    """Document AI processing page with visualization"""
    html_path = os.path.join("templates", "document_ai.html")
    return FileResponse(html_path)


@router.post("/process/form-parser")
async def process_with_form_parser(
    file: UploadFile = File(...),
    document_ai_service: FromDishka[DocumentAIService] = None,
):
    """
    Process document with Form Parser processor.
    Extracts key-value pairs and tables.
    """
    try:
        # Save uploaded file temporarily
        storage_dir = Path("storage") / "temp"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = storage_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process with Form Parser
        result = document_ai_service.process_with_form_parser(
            str(file_path),
            file.content_type or "application/pdf"
        )
        
        # Extract key-value pairs and tables
        # Handle nested document structure
        document = result.get("document", result)
        key_value_pairs = DocumentAIService.extract_key_value_pairs(document)
        tables = DocumentAIService.extract_tables(document)
        
        # Clean up temp file
        file_path.unlink()
        
        return JSONResponse({
            "success": True,
            "document": {
                "text": document.get("text", ""),
                "pages": document.get("pages", []),
            },
            "key_value_pairs": key_value_pairs,
            "tables": tables,
        })
    
    except Exception as e:
        logger.error(f"Error processing document with Form Parser: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.post("/process/ocr")
async def process_with_ocr(
    file: UploadFile = File(...),
    document_ai_service: FromDishka[DocumentAIService] = None,
):
    """
    Process document with OCR processor.
    """
    try:
        # Save uploaded file temporarily
        storage_dir = Path("storage") / "temp"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = storage_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process with OCR
        result = document_ai_service.process_with_ocr(
            str(file_path),
            file.content_type or "application/pdf"
        )
        
        # Handle nested document structure
        document = result.get("document", result)
        
        # Clean up temp file
        file_path.unlink()
        
        return JSONResponse({
            "success": True,
            "document": {
                "text": document.get("text", ""),
                "pages": document.get("pages", []),
            },
        })
    
    except Exception as e:
        logger.error(f"Error processing document with OCR: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.post("/process/custom-extractor")
async def process_with_custom_extractor(
    file: UploadFile = File(...),
    field_names: str = Form(...),  # Comma-separated field names
    field_schema: Optional[str] = Form(None),
    processor_id: Optional[str] = Form(None),
    processor_version: str = Form("rc"),
    document_ai_service: FromDishka[DocumentAIService] = None,
):
    """
    Process document with Custom Extractor processor.
    Extracts custom fields defined in field_names.
    
    Args:
        file: Document file to process
        field_names: Comma-separated list of field names to extract
        processor_id: Optional processor ID (uses default from config if not provided)
        processor_version: Processor version (default: "rc")
    """
    try:
        # Parse field definitions
        fields = _parse_field_definitions(field_names, field_schema)
        if not fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="field definitions cannot be empty"
            )
        logger.info("Custom extractor schema override fields: %s", fields)
        
        # Save uploaded file temporarily
        storage_dir = Path("storage") / "temp"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = storage_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process with Custom Extractor
        result = document_ai_service.process_with_custom_extractor(
            str(file_path),
            fields,
            file.content_type or "application/pdf",
            processor_id,
            processor_version
        )
        
        # Extract custom entities
        entities = DocumentAIService.extract_custom_entities(result)
        
        # Handle nested document structure
        document = result.get("document", result)
        
        # Clean up temp file
        file_path.unlink()
        
        return JSONResponse({
            "success": True,
            "document": {
                "text": document.get("text", ""),
                "pages": document.get("pages", []),
            },
            "entities": entities,
            "raw_result": result,
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document with Custom Extractor: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.post("/process/all")
async def process_with_all(
    file: UploadFile = File(...),
    field_names: Optional[str] = Form(None),  # For Custom Extractor
    field_schema: Optional[str] = Form(None),
    document_ai_service: FromDishka[DocumentAIService] = None,
):
    """
    Process document with all processors (Form Parser, OCR, Custom Extractor).
    Returns combined results.
    """
    try:
        # Save uploaded file temporarily
        storage_dir = Path("storage") / "temp"
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = storage_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        mime_type = file.content_type or "application/pdf"
        
        results = {}
        
        # Process with Form Parser
        try:
            form_parser_result = document_ai_service.process_with_form_parser(
                str(file_path), mime_type
            )
            document = form_parser_result.get("document", form_parser_result)
            results["form_parser"] = {
                "key_value_pairs": DocumentAIService.extract_key_value_pairs(document),
                "tables": DocumentAIService.extract_tables(document),
                "text": document.get("text", ""),
            }
        except Exception as e:
            logger.warning(f"Form Parser failed: {e}")
            results["form_parser"] = {"error": str(e)}
        
        # Process with OCR
        try:
            ocr_result = document_ai_service.process_with_ocr(
                str(file_path), mime_type
            )
            document = ocr_result.get("document", ocr_result)
            results["ocr"] = {
                "text": document.get("text", ""),
                "pages": document.get("pages", []),
            }
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            results["ocr"] = {"error": str(e)}
        
        # Process with Custom Extractor if definitions provided
        if field_names or field_schema:
            try:
                fields = _parse_field_definitions(field_names, field_schema)
                if fields:
                    logger.info("Process-all custom schema fields: %s", fields)
                    custom_result = document_ai_service.process_with_custom_extractor(
                        str(file_path), fields, mime_type
                    )
                    results["custom_extractor"] = {
                        "entities": DocumentAIService.extract_custom_entities(custom_result),
                        "text": custom_result.get("document", custom_result).get("text", ""),
                        "raw_result": custom_result,
                    }
            except Exception as e:
                logger.warning(f"Custom Extractor failed: {e}")
                results["custom_extractor"] = {"error": str(e)}
        
        # Clean up temp file
        file_path.unlink()
        
        return JSONResponse({
            "success": True,
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error processing document with all processors: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.post("/process/console")
async def process_console(
    document_id: int = Form(...),
    extraction_field_ids: str = Form(...), # Comma-separated IDs
    extract_service: FromDishka[ExtractDocumentFieldValuesService] = None,
):
    """
    Process document for Document AI Console.
    Runs multiple processors and aggregates results by page.
    """
    try:
        field_ids = [int(id.strip()) for id in extraction_field_ids.split(",") if id.strip()]
        result = await extract_service.process_document_console(document_id, field_ids)
        return JSONResponse({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error in console processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@router.post("/save/console")
async def save_console(
    document_id: int = Form(...),
    curated_data: str = Form(...), # JSON string
    extract_service: FromDishka[ExtractDocumentFieldValuesService] = None,
):
    """
    Save curated data from Document AI Console.
    """
    try:
        data = json.loads(curated_data)
        await extract_service.save_curated_data(document_id, data)
        return JSONResponse({"success": True})
    except Exception as e:
        logger.error(f"Error saving console data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving data: {str(e)}"
        )


@router.post("/extract-region")
async def extract_region(
    image: UploadFile = File(...),
    document_id: int = Form(...),
    page_number: int = Form(...),
    gemini_ocr_service: FromDishka[GeminiOCRService] = None,
    get_document_interactor: FromDishka[GetDocumentByIdInteractor] = None,
):
    """
    Extract text from a cropped region image using Gemini AI OCR.
    Gemini performs OCR and returns markdown formatted text.
    
    Args:
        image: Cropped region image (PNG)
        document_id: Document ID (for reference)
        page_number: Page number (for reference)
    """
    import tempfile
    from pathlib import Path
    
    try:
        logger.info(f"Processing region image for document_id={document_id}, page={page_number}")
        
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            content = await image.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process image directly with Gemini AI for OCR and markdown conversion
            logger.info(f"Processing cropped image with Gemini AI OCR: {tmp_file_path}")
            markdown = await gemini_ocr_service.extract_markdown_from_image(
                tmp_file_path,
                "image/png"
            )
            
            if not markdown:
                logger.warning("No text extracted from image by Gemini")
                return JSONResponse({
                    "success": True,
                    "text": "",
                    "tables": [],
                    "markdown": ""
                })
            
            logger.info(f"Extracted markdown length: {len(markdown)}")
            logger.info(f"Markdown preview: {markdown[:200]}...")
            
            # Extract plain text from markdown (remove markdown formatting for text field)
            # Simple approach: remove markdown syntax
            import re
            text = re.sub(r'#{1,6}\s+', '', markdown)  # Remove headers
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
            text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
            text = re.sub(r'\|.*?\|', '', text, flags=re.MULTILINE)  # Remove table rows
            text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)  # Remove list markers
            text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize line breaks
            text = text.strip()
            
            return JSONResponse({
                "success": True,
                "text": text,
                "tables": [],
                "markdown": markdown
            })
        
        finally:
            # Clean up temporary file
            try:
                Path(tmp_file_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_file_path}: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting region: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting region: {str(e)}"
        )