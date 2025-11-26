import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re

from openai import AsyncOpenAI
from sqlalchemy import any_, delete as sql_delete, text

from app.dto.field_extraction import BoundingBoxEntry, ExtractedFieldValue
from app.entities.documents import Document, DocumentFieldValue
from app.entities.extracion_fields import ExtractionField
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from app.repositories.documents import (
    DocumentFieldValuesRepository,
    DocumentsRepository,
)
from app.repositories.extraction_fields import ExtractionFieldsRepository
from app.repositories.uow import UnitOfWork
from app.services.document_ai_service import (
    CustomExtractorSchemaField,
    DocumentAIService,
)
from app.services.ocr_field_extraction import OCRFieldExtractionService
from app.services.gemini_ocr_service import GeminiOCRService
from app.utils.enums import (
    DocumentStatus,
    FieldOccurrence,
    ExtractionBy,
    ExtractionFieldType,
)
from app.utils.bbox_finder import (
    extract_pdf_text_with_bbox,
    find_bbox_by_anchors,
    find_bbox_by_plain_anchor,
    normalize_anchor_text,
    sanitize_anchor_input,
)
from google.genai.client import AsyncClient
from google.genai import types
from pydantic import BaseModel, Field, create_model


logger = logging.getLogger("app.services.extract_field_values")


class ExtractDocumentFieldValuesService:
    """Извлекает значения полей: через Document AI кастом-экстрактор или по ключевым словам."""

    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1024

    def __init__(
        self,
        uow: UnitOfWork,
        document_field_values_repository: DocumentFieldValuesRepository,
        openai_client: AsyncOpenAI,
        extraction_fields_repository: ExtractionFieldsRepository,
        document_repository: DocumentsRepository,
        document_ai_service: DocumentAIService,
        ocr_extraction_service: OCRFieldExtractionService,
        gemini_ocr_service: GeminiOCRService,
        gemini_client: AsyncClient,
    ):
        self.uow = uow
        self.document_field_values_repository = document_field_values_repository
        self.openai_client = openai_client
        self.extraction_fields_repository = extraction_fields_repository
        self.document_repository = document_repository
        self.document_ai_service = document_ai_service
        self.ocr_extraction_service = ocr_extraction_service
        self.gemini_ocr_service = gemini_ocr_service
        self.gemini_client = gemini_client

    async def execute(self, document_id: int, extraction_field_ids: List[int]) -> None:
        """Main entry point. Routes extraction based on field configuration."""
        logger.info(
            "ExtractDocumentFieldValuesService: start extraction document_id=%d",
            document_id,
        )

        document = await self.document_repository.get_one(
            where=[Document.id == document_id]
        )
        if not document:
            raise AppError(status_code=404, message=ErrorMessages.DOCUMENT_NOT_FOUND)

        document.status = DocumentStatus.PROCESSING
        await self.uow.commit()

        try:
            extraction_fields = await self.extraction_fields_repository.get_all(
                where=[
                    document.type == any_(ExtractionField.document_types),
                    ExtractionField.id.in_(extraction_field_ids),
                ]
            )
            if not extraction_fields:
                logger.warning(
                    "ExtractDocumentFieldValuesService: no extraction fields matched document_id=%d type=%s",
                    document_id,
                    document.type,
                )
                document.status = DocumentStatus.COMPLETED
                await self.uow.commit()
                return

            await self._wipe_previous_values(document_id)

            # Split fields by extraction strategy
            doc_ai_fields = []
            gemini_fields = []

            for field in extraction_fields:
                if field.extraction_by == ExtractionBy.GEMINI_AI:
                    gemini_fields.append(field)
                else:
                    doc_ai_fields.append(field)

            all_values: List[ExtractedFieldValue] = []

            # 1. Document AI Extraction
            if doc_ai_fields:
                logger.info(
                    "ExtractDocumentFieldValuesService: extracting %d fields via Document AI",
                    len(doc_ai_fields),
                )
                doc_ai_values = await self._extract_with_document_ai(
                    document, doc_ai_fields
                )
                all_values.extend(doc_ai_values)

            # 2. Gemini AI Extraction
            if gemini_fields:
                logger.info(
                    "ExtractDocumentFieldValuesService: extracting %d fields via Gemini AI",
                    len(gemini_fields),
                )
                gemini_values = await self._extract_with_gemini(document, gemini_fields)
                all_values.extend(gemini_values)

            # Save all values
            await self._persist_field_values(document_id, extraction_fields, all_values)

            document.status = DocumentStatus.COMPLETED
            await self.uow.commit()
            logger.info(
                "ExtractDocumentFieldValuesService: completed extraction for document_id=%d (values=%d)",
                document_id,
                len(all_values),
            )
        except Exception as exc:
            logger.error(
                "ExtractDocumentFieldValuesService: unexpected error for document_id=%d: %s",
                document_id,
                exc,
                exc_info=True,
            )
            document.status = DocumentStatus.FAILED
            await self.uow.commit()
            raise

    async def _extract_with_document_ai(
        self, document: Document, fields: List[ExtractionField]
    ) -> List[ExtractedFieldValue]:
        """Extract fields using Google Cloud Document AI Custom Extractor."""
        # Prepare schema for Custom Extractor
        custom_schema = []
        field_map = {}

        for field in fields:
            # Use identifier, prompt as description, and occurrence
            identifier = field.identifier or field.name
            description = self._build_custom_field_description(field)
            occurrence = (
                field.occurrence.value if field.occurrence else "OPTIONAL_MULTIPLE"
            )

            custom_schema.append(
                CustomExtractorSchemaField(
                    name=identifier,
                    description=description,
                    occurrence=occurrence,
                    value_type="string",
                )
            )
            field_map[identifier] = field

        if not custom_schema:
            return []

        try:
            custom_response = await asyncio.to_thread(
                self.document_ai_service.process_with_custom_extractor,
                document.file_path,
                custom_schema,
                document.content_type,
            )

            # Extract entities
            entities = self.document_ai_service.extract_custom_entities(custom_response)

            values = []
            # Convert to ExtractedFieldValue
            for entity in entities:
                field_name = entity.get("type")
                field_def = field_map.get(field_name)

                if not field_def:
                    continue

                # Get page and bbox
                page_anchor = entity.get("page_anchor")
                page_num_str, bbox_entries = self._page_anchor_to_bbox_entries(
                    page_anchor
                )

                # Fallback logic
                if not page_num_str and entity.get("page_number"):
                    page_num_str = str(entity.get("page_number"))

                if not bbox_entries and entity.get("bounding_box"):
                    bbox = entity.get("bounding_box")
                    if isinstance(bbox, dict) and all(
                        k in bbox for k in ["x1", "y1", "x2", "y2"]
                    ):
                        bbox_list = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                        bbox_entries = [
                            BoundingBoxEntry(page=page_num_str or "1", coords=bbox_list)
                        ]
                    elif isinstance(bbox, list) and len(bbox) >= 4:
                        bbox_entries = [
                            BoundingBoxEntry(page=page_num_str or "1", coords=bbox[:4])
                        ]

                values.append(
                    ExtractedFieldValue(
                        field_id=field_def.id,
                        value=entity.get("value", ""),
                        confidence=entity.get("confidence", 1.0),
                        page_num=page_num_str,
                        bbox=bbox_entries,
                    )
                )

            return values

        except Exception as e:
            logger.error(
                "ExtractDocumentFieldValuesService: Document AI extraction failed: %s",
                e,
                exc_info=True,
            )
            return self._empty_values(fields)

    async def _extract_with_gemini(
        self, document: Document, fields: List[ExtractionField]
    ) -> List[ExtractedFieldValue]:
        """Extract fields using Gemini 2.5 Flash with structured output and bbox finding."""
        if not fields:
            return []

        try:
            # 1. Build Pydantic models for structured extraction
            class BboxAnchor(BaseModel):
                start_words: str = Field(
                    ...,
                    description="First 5-10 UNIQUE consecutive words from the BEGINNING of extracted content. Must be verbatim quote that appears ONLY ONCE on this page.",
                )
                end_words: str = Field(
                    ...,
                    description="Last 5-10 UNIQUE consecutive words from the END of extracted content. Must be verbatim quote that appears ONLY ONCE on this page.",
                )

            class ExtractedField(BaseModel):
                """Single extracted field occurrence."""
                name: str = Field(..., description="Field name exactly as requested")
                type: str = Field(..., description="Type: text, table, clause, paragraph, address, etc.")
                value: str = Field(
                    ..., description="The ACTUAL extracted content in markdown format. Never use placeholders like '___' or '...' - extract real text!"
                )
                bbox_anchor: BboxAnchor = Field(
                    ..., description="Unique word anchors to locate this content"
                )
                page: int = Field(..., description="Page number where this content is located", ge=1)
                confidence: str = Field(..., description="high/medium/low")

            class DocumentExtraction(BaseModel):
                """Complete extraction result."""
                fields: List[ExtractedField] = Field(..., description="All extracted field occurrences")
                total_pages: Optional[int] = Field(None, description="Total pages in document")

            ExtractionSchema = DocumentExtraction

            # 2. Build improved prompt with clear rules
            prompt = f"""You are a precise document extraction assistant. Extract the requested fields from this document.

## CRITICAL EXTRACTION RULES:

### 1. VALUE EXTRACTION:
- Extract the ACTUAL content, never placeholders like "___", "...", or blank lines
- If a field label exists but content is on the next page, extract content from that page
- For paragraphs/clauses: extract the BODY text, not just the heading
- Use markdown format for tables and structured content

### 2. MULTI-PAGE CONTENT:
- If field content spans multiple pages, create SEPARATE entries for each page
- Each entry must have its own value, bbox_anchor, and page number
- Example: "6. QUALITY" heading on page 1, but quality specs on page 10 → extract from page 10

### 3. BBOX_ANCHOR - UNIQUENESS IS CRITICAL:
- start_words: First 5-10 consecutive words from YOUR extracted value
- end_words: Last 5-10 consecutive words from YOUR extracted value  
- BOTH must be EXACT verbatim quotes from the document
- BOTH must appear ONLY ONCE on that page (to avoid bbox confusion)
- For addresses/companies: include the FULL company name + first line of address as start_words
- For paragraphs: include section number + first words as start_words

### 4. AVOIDING DUPLICATE ANCHORS:
- DON'T use generic phrases that repeat (like just "UNITED KINGDOM" or just company name)
- DO include context: "GLENCORE ENERGY UK LTD 50, BERKELEY STREET" instead of just "GLENCORE ENERGY UK LTD"
- For similar sections (buyer/seller), use DIFFERENT unique identifiers

## FIELDS TO EXTRACT:
"""
            for i, field in enumerate(fields, 1):
                name = field.identifier or field.name
                field_type = field.type.value if hasattr(field.type, 'value') else str(field.type)
                field_prompt = field.prompt or f'Extract the {name}'
                prompt += f"\n{i}. **{name}** ({field_type}): {field_prompt}"

            prompt += """

## RESPONSE FORMAT:
Return a JSON with "fields" array. Each field occurrence is a separate entry.
If content continues on multiple pages, create multiple entries with same name but different pages."""

            # 3. Call Gemini 2.5 Flash
            uploaded_file = await self.gemini_client.files.upload(
                file=document.file_path
            )

            response = await self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=ExtractionSchema,
                ),
            )

            # 4. Parse Response - matches prototype structure
            import json

            # Log raw response for debugging
            logger.info(f"[Gemini AI] Raw response text length: {len(response.text) if response.text else 0}")
            if response.text:
                logger.debug(f"[Gemini AI] Raw response (first 1000 chars): {response.text[:1000]}")
            
            extraction_data = json.loads(response.text) if response.text else {}
            logger.info(f"[Gemini AI] Parsed extraction_data keys: {list(extraction_data.keys())}")
            logger.info(f"[Gemini AI] Full extraction_data structure: {json.dumps(extraction_data, indent=2, ensure_ascii=False)[:2000]}")

            # Parse as DocumentExtraction (matches prototype)
            extraction = DocumentExtraction(**extraction_data) if extraction_data else None
            if not extraction or not extraction.fields:
                logger.warning("[Gemini AI] No fields extracted")
                return self._empty_values(fields)

            logger.info(f"[Gemini AI] Extracted {len(extraction.fields)} field entries")

            # 5. Find BBoxes
            # Extract OCR data once
            ocr_data = await asyncio.to_thread(
                extract_pdf_text_with_bbox, document.file_path
            )
            logger.info(f"[Gemini AI] Extracted {len(ocr_data)} OCR text blocks")

            # Map field names to ExtractionField objects
            field_map = {field.identifier or field.name: field for field in fields}

            results = []
            # Process each extracted field
            for extracted_field in extraction.fields:
                field_name = extracted_field.name
                field_def = field_map.get(field_name)
                
                if not field_def:
                    logger.warning(f"[Gemini AI] Extracted field '{field_name}' not in requested fields, skipping")
                    continue

                # Filter out placeholder/empty values
                value = extracted_field.value.strip() if extracted_field.value else ""
                if not value or self._is_placeholder_value(value):
                    logger.warning(f"[Gemini AI] Field '{field_name}' has placeholder/empty value, skipping: '{value[:50]}'")
                    continue

                logger.info(f"[Gemini AI] Processing field '{field_name}' on page {extracted_field.page}")
                
                anchor = extracted_field.bbox_anchor
                raw_start = anchor.start_words
                raw_end = anchor.end_words
                clean_start = sanitize_anchor_input(raw_start)
                clean_end = sanitize_anchor_input(raw_end)
                
                logger.info(
                    "[Gemini AI] Field: %s | page %d | anchors raw(start=%r, end=%r) -> clean(start=%r, end=%r)",
                    field_name,
                    extracted_field.page,
                    raw_start,
                    raw_end,
                    clean_start,
                    clean_end,
                )
                
                # Log normalized anchors for debugging
                norm_start = normalize_anchor_text(clean_start)
                norm_end = normalize_anchor_text(clean_end)
                logger.info(
                    f"[Gemini AI] Normalized anchors: start='{norm_start}' end='{norm_end}'"
                )
                
                bbox = None
                if anchor and ocr_data:
                    # Use find_bbox_by_anchors directly
                    bbox_coords = find_bbox_by_anchors(
                        ocr_data=ocr_data,
                        start_words=clean_start,
                        end_words=clean_end,
                        page_num=extracted_field.page,
                    )
                    if bbox_coords:
                        # Handle both tuple and list return types
                        if isinstance(bbox_coords, tuple):
                            bbox_coords, _ = bbox_coords
                        bbox = [
                            BoundingBoxEntry(page=str(extracted_field.page), coords=bbox_coords)
                        ]

                extracted_value = ExtractedFieldValue(
                    field_id=field_def.id,
                    value=value,
                    confidence=1.0 if extracted_field.confidence == "high" else (0.7 if extracted_field.confidence == "medium" else 0.5),
                    page_num=str(extracted_field.page),
                    bbox=bbox,
                )
                results.append(extracted_value)
                logger.info(
                    f"[Gemini AI] Added field '{field_name}': page={extracted_field.page}, value_length={len(value)}, bbox={'found' if bbox else 'not found'}"
                )

            logger.info(f"[Gemini AI] Total results extracted: {len(results)} field entries")
            return results

        except Exception as e:
            logger.error(
                "ExtractDocumentFieldValuesService: Gemini extraction failed: %s",
                e,
                exc_info=True,
            )
            return self._empty_values(fields)

    async def _extract_manual_fields_with_coordinates(
        self,
        markdown_text: str,
        keyword_fields: List[ExtractionField],
        coordinates_map: Optional[Dict[str, Any]],
    ) -> List[ExtractedFieldValue]:
        """
        Extract manual fields with coordinates using pre-extracted coordinates map.
        """
        # Use asyncio to run sync method in async context
        import asyncio

        def _extract_sync():
            return self.ocr_extraction_service.extract_by_schema(
                markdown_text=markdown_text,
                extraction_fields=keyword_fields,
            )

        # Extract values from markdown
        values = await asyncio.to_thread(_extract_sync)

        # Get coordinates from pre-extracted map (no additional API calls)
        if coordinates_map:
            for value in values:
                if value.value and value.value != "NOT FOUND":
                    # Find which field this value belongs to
                    field = next(
                        (f for f in keyword_fields if f.id == value.field_id), None
                    )
                    if not field:
                        continue

                    # Get keywords for this field
                    options = self.ocr_extraction_service._load_manual_options(field)
                    keywords = options.get(
                        "keywords"
                    ) or self.ocr_extraction_service._derive_default_keywords(field)

                    if keywords:
                        # Find coordinates in pre-extracted map
                        coord_page_num, coord_bbox_entries = (
                            self.gemini_ocr_service.find_text_coordinates_in_map(
                                text_to_find=value.value,
                                keywords=keywords,
                                coordinates_map=coordinates_map,
                                markdown_text=markdown_text,
                            )
                        )

                        if coord_page_num:
                            value.page_num = coord_page_num
                        if coord_bbox_entries:
                            value.bbox = [
                                BoundingBoxEntry(
                                    page=entry["page"], coords=entry["coords"]
                                )
                                for entry in coord_bbox_entries
                            ]
                            logger.debug(
                                "ExtractDocumentFieldValuesService: found coordinates from map for field_id=%d (page=%s, bbox_entries=%d)",
                                value.field_id,
                                coord_page_num,
                                len(coord_bbox_entries),
                            )

        return values

    async def _extract_markdown_hybrid(
        self,
        file_path: str,
        mime_type: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Extract markdown and coordinates using Document AI Layout Parser.
        Replaces previous hybrid approach.
        """
        try:
            logger.info(
                "ExtractDocumentFieldValuesService: processing with Document AI Layout Parser..."
            )

            # Run sync Document AI call in thread
            response = await asyncio.to_thread(
                self.document_ai_service.process_with_layout, file_path, mime_type
            )

            if not response:
                logger.error(
                    "ExtractDocumentFieldValuesService: empty response from Layout Parser"
                )
                return None, None

            # Parse response
            markdown, coordinates_map = self._parse_layout_response(response)

            logger.info(
                "ExtractDocumentFieldValuesService: Layout Parser extraction completed (markdown_length=%d, text_ranges=%d)",
                len(markdown),
                len(coordinates_map.get("text_ranges", [])) if coordinates_map else 0,
            )

            return markdown, coordinates_map

        except Exception as e:
            logger.error(
                "ExtractDocumentFieldValuesService: error in Layout Parser extraction: %s",
                e,
                exc_info=True,
            )
            return None, None

    def _parse_layout_response(
        self, response: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Parse Document AI Layout Parser response to Markdown and extract coordinates.
        """
        document = response.get("document", response)
        layout = document.get("document_layout", {})
        blocks = layout.get("blocks", [])
        pages = document.get("pages", [])

        markdown_parts = []
        text_ranges = []

        for block in blocks:
            block_id = block.get("block_id")
            page_span = block.get("page_span", {})

            # Handle Text Block
            if "text_block" in block:
                text_block = block.get("text_block", {})
                text = text_block.get("text", "").strip()
                type_ = text_block.get("type_", "")

                if not text:
                    continue

                # Format based on type
                if type_ == "header" or type_ == "heading-1":
                    markdown = f"# {text}"
                elif type_ == "heading-2":
                    markdown = f"## {text}"
                elif type_ == "heading-3":
                    markdown = f"### {text}"
                elif type_ == "list-item":
                    markdown = f"* {text}"
                else:
                    markdown = text

                markdown_parts.append(markdown)

                # Calculate indices
                start_idx = (
                    len("\n\n".join(markdown_parts[:-1])) + 2
                    if len(markdown_parts) > 1
                    else 0
                )
                end_idx = start_idx + len(markdown)

                # Map coordinates
                page_numbers, bboxes = self._map_block_to_bbox(text, page_span, pages)

                text_ranges.append(
                    {
                        "text": markdown,
                        "start_index": start_idx,
                        "end_index": end_idx,
                        "page_numbers": page_numbers,
                        "bboxes": bboxes,
                        "block_id": block_id,
                    }
                )

            # Handle Table Block
            elif "table_block" in block:
                table_block = block.get("table_block", {})
                table_md = self._convert_table_to_markdown(table_block)
                if table_md:
                    markdown_parts.append(table_md)

                    start_idx = (
                        len("\n\n".join(markdown_parts[:-1])) + 2
                        if len(markdown_parts) > 1
                        else 0
                    )
                    end_idx = start_idx + len(table_md)

                    # For tables, we try to get bbox from the block if available, or combined from cells?
                    # Layout Parser blocks usually don't have bbox directly in this view,
                    # but we can try to map it. For now, use page_span.
                    page_numbers, bboxes = self._map_block_to_bbox(
                        "TABLE", page_span, pages
                    )

                    text_ranges.append(
                        {
                            "text": "TABLE",  # Placeholder or full text?
                            "start_index": start_idx,
                            "end_index": end_idx,
                            "page_numbers": page_numbers,
                            "bboxes": bboxes,
                            "block_id": block_id,
                            "is_table": True,
                        }
                    )

        full_markdown = "\n\n".join(markdown_parts)
        return full_markdown, {"text_ranges": text_ranges}

    def _convert_table_to_markdown(self, table_block: Dict[str, Any]) -> str:
        """Convert table block to Markdown."""
        rows = []

        def get_cell_text(cell):
            blocks = cell.get("blocks", [])
            texts = []
            for b in blocks:
                if "text_block" in b:
                    texts.append(b["text_block"].get("text", "").strip())
            return " ".join(texts).strip()

        # Header Rows
        header_rows = table_block.get("header_rows", [])
        if header_rows:
            headers = [get_cell_text(cell) for cell in header_rows[0].get("cells", [])]
            rows.append("| " + " | ".join(headers) + " |")
            rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

            for hr in header_rows[1:]:
                cells = [get_cell_text(cell) for cell in hr.get("cells", [])]
                rows.append("| " + " | ".join(cells) + " |")
        else:
            body_rows = table_block.get("body_rows", [])
            if body_rows:
                col_count = len(body_rows[0].get("cells", []))
                rows.append("| " + " | ".join([""] * col_count) + " |")
                rows.append("| " + " | ".join(["---"] * col_count) + " |")

        # Body Rows
        body_rows = table_block.get("body_rows", [])
        for row in body_rows:
            cells = [get_cell_text(cell) for cell in row.get("cells", [])]
            rows.append("| " + " | ".join(cells) + " |")

        return "\n".join(rows)

    def _map_block_to_bbox(
        self, block_text: str, page_span: Dict[str, Any], pages: List[Dict[str, Any]]
    ) -> Tuple[List[int], Dict[str, List[float]]]:
        """
        Map layout block to coordinates using pages data.
        """
        if not pages:
            return [], {}

        page_start = page_span.get("page_start", 1)
        page_end = page_span.get("page_end", page_start)

        page_numbers = list(range(page_start, page_end + 1))
        bboxes = {}

        is_table = block_text == "TABLE"

        for page_num in page_numbers:
            # Adjust for 0-indexed list if pages are 1-indexed in span
            if page_num - 1 < len(pages):
                page = pages[page_num - 1]
                found_bbox = None

                if is_table:
                    # Handle Table: look for tables in page.tables
                    # We try to match based on location or content if possible
                    # Since we don't have easy content matching for the whole table here,
                    # we might just take the first table on the page that hasn't been used?
                    # Or better, try to match the first cell text?
                    # For now, let's try to find *any* table on this page.
                    # Ideally we should pass the table content to this method to match.
                    # But since we don't have it easily passed, let's look at page.tables.
                    page_tables = page.get("tables", [])
                    if page_tables:
                        # If multiple tables, this is ambiguous.
                        # But often there's 1 table per block.
                        # We could try to match the bounding box of the block if we had it.
                        # Layout Parser block doesn't give bbox in the root, but maybe in 'layout'?
                        # The 'block' object passed to _parse_layout_response has 'page_span'.
                        # It might not have bbox.

                        # Fallback: Union of all tables on the page? Or just the first?
                        # Let's take the union of all tables for now to be safe,
                        # or if we can, match text.

                        # Actually, let's try to match the table by checking if it's not already covered?
                        # Too complex for now. Let's just take the bounding box of all tables on the page.
                        table_bboxes = []
                        for tbl in page_tables:
                            layout = tbl.get("layout", {})
                            bbox = self._extract_bbox_from_poly(
                                layout.get("boundingPoly")
                            )
                            if bbox:
                                table_bboxes.append(bbox)

                        if table_bboxes:
                            found_bbox = self._combine_bboxes(table_bboxes)

                else:
                    # Handle Text
                    clean_text = block_text.replace("#", "").replace("*", "").strip()
                    if not clean_text:
                        continue

                    # Normalize text for matching (remove whitespace)
                    norm_text = "".join(clean_text.split()).lower()

                    # Search in paragraphs
                    paragraphs = page.get("paragraphs", [])
                    matched_bboxes = []

                    for para in paragraphs:
                        layout = para.get("layout", {})
                        text_anchor = layout.get("textAnchor", {})
                        content = text_anchor.get("content", "")

                        if not content:
                            continue

                        # Normalize paragraph content
                        norm_content = "".join(content.split()).lower()

                        # Check for overlap
                        if norm_text in norm_content or norm_content in norm_text:
                            bbox = self._extract_bbox_from_poly(
                                layout.get("boundingPoly")
                            )
                            if bbox:
                                matched_bboxes.append(bbox)

                    # If no paragraph match, try blocks (broader)
                    if not matched_bboxes:
                        blocks = page.get("blocks", [])
                        for blk in blocks:
                            layout = blk.get("layout", {})
                            text_anchor = layout.get("textAnchor", {})
                            content = text_anchor.get("content", "")
                            if not content:
                                continue
                            norm_content = "".join(content.split()).lower()
                            if norm_text in norm_content or norm_content in norm_text:
                                bbox = self._extract_bbox_from_poly(
                                    layout.get("boundingPoly")
                                )
                                if bbox:
                                    matched_bboxes.append(bbox)

                    if matched_bboxes:
                        found_bbox = self._combine_bboxes(matched_bboxes)

                if found_bbox:
                    bboxes[str(page_num)] = found_bbox

        return page_numbers, bboxes

    def _extract_bbox_from_poly(
        self, bounding_poly: Dict[str, Any]
    ) -> Optional[List[float]]:
        """Extract normalized bounding box from Document AI boundingPoly"""
        if not bounding_poly:
            return None

        # OCR processor uses normalizedVertices (0.0-1.0)
        vertices = bounding_poly.get("normalizedVertices", []) or bounding_poly.get(
            "vertices", []
        )
        if not vertices or len(vertices) < 4:
            return None

        xs = []
        ys = []
        for v in vertices:
            x = v.get("x")
            y = v.get("y")
            if x is not None:
                xs.append(float(x))
            if y is not None:
                ys.append(float(y))

        if not xs or not ys:
            return None

        # Return normalized coordinates [x1, y1, x2, y2]
        return [
            min(xs),  # x1
            min(ys),  # y1
            max(xs),  # x2
            max(ys),  # y2
        ]

    async def _wipe_previous_values(self, document_id: int) -> None:
        logger.info(
            "ExtractDocumentFieldValuesService: removing previous DocumentFieldValue rows (document_id=%d)",
            document_id,
        )
        delete_stmt = sql_delete(DocumentFieldValue).where(
            DocumentFieldValue.document_id == document_id
        )
        await self.document_field_values_repository.session.execute(delete_stmt)
        await self.uow.commit()

    async def _extract_ai_fields_with_gemini(
        self,
        fields: List[ExtractionField],
        file_path: str,
        markdown_text: Optional[str],
    ) -> List[ExtractedFieldValue]:
        """
        Extract AI fields using Gemini with response_schema.
        Uses markdown_text if available, otherwise uploads file.
        """
        if not fields:
            return []

        # Build Pydantic model for response schema
        field_models: Dict[str, Any] = {}
        property_to_field: Dict[str, ExtractionField] = {}

        for field in fields:
            property_name = (field.identifier or field.name or "").strip()
            if not property_name:
                logger.warning(
                    "ExtractDocumentFieldValuesService: field %s has no identifier/name, skipping AI extraction",
                    field.id,
                )
                continue

            # Create field model based on type
            description = self._build_custom_field_description(field)
            if field.type.value == "TEXT":
                field_models[property_name] = (
                    str,
                    Field(description=description, default=""),
                )
            elif field.type.value == "NUMBER":
                field_models[property_name] = (
                    float,
                    Field(description=description, default=0.0),
                )
            elif field.type.value == "DATE":
                field_models[property_name] = (
                    str,
                    Field(description=description, default=""),
                )
            elif field.type.value == "BOOL":
                field_models[property_name] = (
                    bool,
                    Field(description=description, default=False),
                )
            else:
                field_models[property_name] = (
                    str,
                    Field(description=description, default=""),
                )

            property_to_field[property_name] = field

        if not field_models:
            logger.warning(
                "ExtractDocumentFieldValuesService: no valid schema fields for AI extraction"
            )
            return self._empty_values(fields)

        # Create dynamic Pydantic model using create_model
        schema_fields = {}
        for prop_name, (field_type, field_info) in field_models.items():
            schema_fields[prop_name] = (field_type, field_info)

        ExtractionSchema = create_model("ExtractionSchema", **schema_fields)

        # Build prompt
        field_descriptions = []
        for field in fields:
            desc = self._build_custom_field_description(field)
            field_descriptions.append(f"- {field.identifier or field.name}: {desc}")

        prompt = f"""Extract the following fields from this document:

{chr(10).join(field_descriptions)}

Return the extracted values in the requested format."""

        try:
            # Use markdown if available, otherwise upload file
            if markdown_text:
                logger.info(
                    "ExtractDocumentFieldValuesService: using markdown text for AI extraction"
                )
                response = await self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[markdown_text, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=ExtractionSchema,
                    ),
                )
            else:
                logger.info(
                    "ExtractDocumentFieldValuesService: uploading file for AI extraction"
                )
                uploaded_file = await self.gemini_client.files.upload(file=file_path)

                response = await self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=ExtractionSchema,
                    ),
                )

            # Parse response
            import json

            result_data = json.loads(response.text) if response.text else {}

            results: List[ExtractedFieldValue] = []
            for field in fields:
                property_name = (field.identifier or field.name or "").strip()
                if not property_name or property_name not in result_data:
                    results.append(self._empty_value(field))
                    continue

                value = result_data[property_name]
                if value is None:
                    results.append(self._empty_value(field))
                    continue

                # Convert to string
                value_str = str(value) if not isinstance(value, str) else value

                results.append(
                    ExtractedFieldValue(
                        field_id=field.id,
                        value=value_str,
                        confidence=1.0,
                        page_num=None,
                        bbox=None,
                    )
                )

            return results

        except Exception as exc:
            logger.error(
                "ExtractDocumentFieldValuesService: Gemini AI extraction failed: %s",
                exc,
                exc_info=True,
            )
            return self._empty_values(fields)

    def _build_custom_field_description(self, field: ExtractionField) -> Optional[str]:
        parts: List[str] = []
        if field.short_description:
            parts.append(field.short_description)
        if field.prompt:
            parts.append(field.prompt)
        if field.examples:
            parts.append("Examples: " + "; ".join(field.examples[:3]))
        return " ".join(parts) if parts else None

    @staticmethod
    def _extract_property_name(entity_type: str) -> str:
        if not entity_type:
            return ""
        if "/" in entity_type:
            entity_type = entity_type.split("/")[-1]
        if ":" in entity_type:
            entity_type = entity_type.split(":")[-1]
        return entity_type.strip()

    def _allows_multiple(self, occurrence: Optional[FieldOccurrence]) -> bool:
        if occurrence is None:
            return False
        return occurrence in {
            FieldOccurrence.OPTIONAL_MULTIPLE,
            FieldOccurrence.REQUIRED_MULTIPLE,
        }

    def _combine_field_values(
        self,
        field: ExtractionField,
        values: List[ExtractedFieldValue],
    ) -> ExtractedFieldValue:
        combined_texts = [value.value.strip() for value in values if value.value]
        combined_value = "\n\n".join(combined_texts)
        combined_confidence = sum((value.confidence or 0.0) for value in values) / len(
            values
        )
        combined_pages = self._format_page_numbers(
            self._collect_pages_from_values(values)
        )
        combined_bbox = self._merge_bbox_entries(values)

        return ExtractedFieldValue(
            field_id=field.id,
            value=combined_value,
            confidence=combined_confidence,
            page_num=combined_pages,
            bbox=combined_bbox,
        )

    def _collect_pages_from_values(
        self, values: List[ExtractedFieldValue]
    ) -> List[int]:
        pages: set[int] = set()
        for value in values:
            pages.update(self._parse_page_numbers(value.page_num))
        return sorted(pages)

    def _parse_page_numbers(self, page_num: Optional[str]) -> List[int]:
        if not page_num:
            return []
        page_num = page_num.strip()
        results: List[int] = []
        for part in page_num.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                try:
                    start = int(start_str)
                    end = int(end_str)
                except ValueError:
                    continue
                if start <= end:
                    results.extend(range(start, end + 1))
            else:
                try:
                    results.append(int(part))
                except ValueError:
                    continue
        return results

    def _format_page_numbers(self, pages: List[int]) -> Optional[str]:
        if not pages:
            return None
        pages = sorted(set(pages))
        if len(pages) == 1:
            return str(pages[0])
        if len(pages) == 2:
            return f"{pages[0]},{pages[1]}"
        return f"{pages[0]}-{pages[-1]}"

    def _page_anchor_to_bbox_entries(
        self,
        page_anchor: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[List[BoundingBoxEntry]]]:
        if not page_anchor:
            logger.debug(
                "ExtractDocumentFieldValuesService._page_anchor_to_bbox_entries: page_anchor is None or empty"
            )
            return None, None

        page_refs = page_anchor.get("pageRefs") or page_anchor.get("page_refs") or []
        if not page_refs:
            logger.debug(
                "ExtractDocumentFieldValuesService._page_anchor_to_bbox_entries: no pageRefs found, "
                "page_anchor keys: %s",
                (
                    list(page_anchor.keys())
                    if isinstance(page_anchor, dict)
                    else "not a dict"
                ),
            )
            return None, None

        page_numbers: List[int] = []
        bbox_map: Dict[str, List[float]] = {}

        for ref_idx, ref in enumerate(page_refs):
            page_index = ref.get("page")
            if page_index is None:
                logger.debug(
                    "ExtractDocumentFieldValuesService._page_anchor_to_bbox_entries: page_ref[%d] has no 'page' field, "
                    "ref keys: %s",
                    ref_idx,
                    list(ref.keys()) if isinstance(ref, dict) else "not a dict",
                )
                continue
            try:
                page_int = int(page_index) + 1
            except (TypeError, ValueError) as e:
                logger.debug(
                    "ExtractDocumentFieldValuesService._page_anchor_to_bbox_entries: invalid page_index '%s': %s",
                    page_index,
                    e,
                )
                continue
            page_numbers.append(page_int)
            page_key = str(page_int)
            bounding_poly = ref.get("boundingPoly") or ref.get("bounding_poly") or {}
            if not bounding_poly:
                logger.debug(
                    "ExtractDocumentFieldValuesService._page_anchor_to_bbox_entries: page_ref[%d] (page=%d) has no boundingPoly, "
                    "ref keys: %s",
                    ref_idx,
                    page_int,
                    list(ref.keys()) if isinstance(ref, dict) else "not a dict",
                )
                continue
            bbox = self._extract_bbox_from_poly(bounding_poly)
            if not bbox:
                logger.debug(
                    "ExtractDocumentFieldValuesService._page_anchor_to_bbox_entries: failed to extract bbox from bounding_poly "
                    "for page_ref[%d] (page=%d), bounding_poly keys: %s",
                    ref_idx,
                    page_int,
                    (
                        list(bounding_poly.keys())
                        if isinstance(bounding_poly, dict)
                        else "not a dict"
                    ),
                )
                continue
            if page_key in bbox_map:
                combined = self._combine_bboxes([bbox_map[page_key], bbox])
                bbox_map[page_key] = combined or bbox_map[page_key]
            else:
                bbox_map[page_key] = bbox

        if (
            len([page for page in bbox_map.keys() if page != "combined"]) > 1
            and "combined" not in bbox_map
        ):
            combined_bbox = self._combine_bboxes(
                [coords for key, coords in bbox_map.items() if key != "combined"]
            )
            if combined_bbox:
                bbox_map["combined"] = combined_bbox

        bbox_entries = [
            BoundingBoxEntry(page=page, coords=coords)
            for page, coords in bbox_map.items()
        ]
        page_num_str = self._format_page_numbers(sorted(set(page_numbers)))

        logger.debug(
            "ExtractDocumentFieldValuesService._page_anchor_to_bbox_entries: extracted page_num_str=%s, "
            "bbox_entries count=%d",
            page_num_str,
            len(bbox_entries),
        )

        return page_num_str, bbox_entries or None

    def _merge_bbox_entries(
        self,
        values: List[ExtractedFieldValue],
    ) -> Optional[List[BoundingBoxEntry]]:
        merged: Dict[str, List[float]] = {}
        for value in values:
            if not value.bbox:
                continue
            for entry in value.bbox:
                if entry.page in merged:
                    combined = self._combine_bboxes([merged[entry.page], entry.coords])
                    merged[entry.page] = combined or merged[entry.page]
                else:
                    merged[entry.page] = entry.coords

        if not merged:
            return None

        base_pages = [page for page in merged.keys() if page != "combined"]
        if len(base_pages) > 1:
            combined_coords = self._combine_bboxes(
                [merged[page] for page in base_pages]
            )
            if combined_coords:
                merged["combined"] = combined_coords

        return [
            BoundingBoxEntry(page=page, coords=coords)
            for page, coords in merged.items()
        ]

    def _extract_form_fields(
        self, ocr_document: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        document = ocr_document.get("document", ocr_document)
        pages = document.get("pages", [])
        text_content = document.get("text", "")
        form_fields: List[Tuple[str, str]] = []

        for page in pages:
            entries = page.get("formFields", []) or page.get("form_fields", [])
            for entry in entries:
                key_text = self._layout_to_text(
                    entry.get("fieldName") or entry.get("field_name"), text_content
                )
                value_text = self._layout_to_text(
                    entry.get("fieldValue") or entry.get("field_value"), text_content
                )
                if key_text and value_text:
                    form_fields.append((key_text.strip(), value_text.strip()))

        return form_fields

    @staticmethod
    def _layout_to_text(layout: Optional[Dict[str, Any]], full_text: str) -> str:
        if not layout:
            return ""

        text_anchor = layout.get("textAnchor") or layout.get("text_anchor")
        if not text_anchor:
            return ""

        segments = text_anchor.get("textSegments") or text_anchor.get("text_segments")
        if not segments:
            return text_anchor.get("content", "")

        parts: List[str] = []
        for segment in segments:
            start_index = int(
                segment.get("startIndex", segment.get("start_index", 0)) or 0
            )
            end_index = int(
                segment.get("endIndex", segment.get("end_index", len(full_text)))
                or len(full_text)
            )
            parts.append(full_text[start_index:end_index])

        return "".join(parts)

    def _get_document_text(self, ocr_document: Dict[str, Any]) -> str:
        if "text" in ocr_document and ocr_document["text"]:
            return ocr_document["text"]
        if "document" in ocr_document:
            doc = ocr_document["document"]
            if isinstance(doc, dict):
                return doc.get("text", "") or ""
        return ""

    def _find_text_coordinates_in_ocr(
        self,
        text_to_find: str,
        ocr_document: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[List[BoundingBoxEntry]]]:
        if not text_to_find or not text_to_find.strip():
            return None, None

        search_text = text_to_find.strip()[:200].lower()
        if len(search_text) < 4:
            return None, None

        pages = ocr_document.get("pages") or ocr_document.get("document", {}).get(
            "pages", []
        )
        if not pages:
            return None, None

        full_text = self._get_document_text(ocr_document)
        text_lower = full_text.lower()
        search_pos = text_lower.find(search_text)
        if search_pos == -1 and len(search_text) > 50:
            search_pos = text_lower.find(search_text[:50])
            if search_pos == -1:
                return None, None

        found_bboxes: Dict[str, List[float]] = {}
        found_pages = set()

        for page in pages:
            page_num = page.get("pageNumber") or page.get("page_number") or 1
            paragraphs = page.get("paragraphs", [])
            for para in paragraphs:
                layout = para.get("layout", {})
                anchor = layout.get("textAnchor") or layout.get("text_anchor") or {}
                segments = (
                    anchor.get("textSegments") or anchor.get("text_segments") or []
                )

                para_start = None
                para_end = None
                if segments:
                    para_start = int(
                        segments[0].get("startIndex", segments[0].get("start_index", 0))
                        or 0
                    )
                    para_end = int(
                        segments[-1].get(
                            "endIndex", segments[-1].get("end_index", len(full_text))
                        )
                        or len(full_text)
                    )

                include = False
                if para_start is not None and para_end is not None:
                    include = para_start <= search_pos < para_end
                else:
                    para_text = anchor.get("content", "")
                    if (
                        not para_text
                        and para_start is not None
                        and para_end is not None
                    ):
                        para_text = full_text[para_start:para_end]
                    include = search_text in (para_text or "").lower()

                if include:
                    bbox = self._extract_bbox_from_poly(
                        layout.get("boundingPoly") or layout.get("bounding_poly") or {}
                    )
                    if bbox:
                        found_bboxes[str(page_num)] = bbox
                        found_pages.add(page_num)
                        break

        if not found_bboxes:
            return None, None

        if len(found_bboxes) > 1:
            combined = self._combine_bboxes(list(found_bboxes.values()))
            if combined:
                found_bboxes["combined"] = combined

        if len(found_pages) == 1:
            page_num_str = str(next(iter(found_pages)))
        else:
            sorted_pages = sorted(found_pages)
            if not sorted_pages:
                page_num_str = None
            elif len(sorted_pages) == 2:
                page_num_str = f"{sorted_pages[0]},{sorted_pages[1]}"
            else:
                page_num_str = f"{sorted_pages[0]}-{sorted_pages[-1]}"

        bbox_entries = [
            BoundingBoxEntry(page=page, coords=coords)
            for page, coords in found_bboxes.items()
        ]
        return page_num_str, bbox_entries or None

    @staticmethod
    def _extract_bbox_from_poly(bounding_poly: Dict[str, Any]) -> Optional[List[float]]:
        if not bounding_poly:
            return None

        vertices = (
            bounding_poly.get("normalizedVertices")
            or bounding_poly.get("normalized_vertices")
            or []
        )
        if not vertices:
            return None

        xs = []
        ys = []
        for vertex in vertices:
            if not isinstance(vertex, dict):
                continue
            x_val = vertex.get("x")
            y_val = vertex.get("y")
            if x_val is not None:
                xs.append(float(x_val))
            if y_val is not None:
                ys.append(float(y_val))

        if not xs or not ys:
            return None

        return [
            max(0.0, min(1.0, min(xs))),
            max(0.0, min(1.0, min(ys))),
            max(0.0, min(1.0, max(xs))),
            max(0.0, min(1.0, max(ys))),
        ]

    @staticmethod
    def _combine_bboxes(bboxes: List[List[float]]) -> Optional[List[float]]:
        valid = [bbox for bbox in bboxes if bbox and len(bbox) == 4]
        if not valid:
            return None

        min_x = min(b[0] for b in valid)
        min_y = min(b[1] for b in valid)
        max_x = max(b[2] for b in valid)
        max_y = max(b[3] for b in valid)

        return [
            max(0.0, min(1.0, min_x)),
            max(0.0, min(1.0, min_y)),
            max(0.0, min(1.0, max_x)),
            max(0.0, min(1.0, max_y)),
        ]

    async def _persist_field_values(
        self,
        document_id: int,
        fields: List[ExtractionField],
        extracted_values: List[ExtractedFieldValue],
    ) -> None:
        if not extracted_values:
            logger.warning(
                "ExtractDocumentFieldValuesService: nothing to persist for document_id=%d",
                document_id,
            )
            return

        value_texts = [value.value for value in extracted_values if value.value]
        embeddings: List[List[float]] = []
        if value_texts:
            try:
                embeddings = await self._create_embeddings_batch(value_texts)
            except AppError as exc:
                logger.error(
                    "ExtractDocumentFieldValuesService: embeddings failed: %s",
                    exc,
                    exc_info=True,
                )

        embedding_idx = 0
        fields_by_id = {field.id: field for field in fields}
        entities: List[DocumentFieldValue] = []
        for value in extracted_values:
            if value.field_id not in fields_by_id:
                continue

            embedding_vector = None
            if value.value and embedding_idx < len(embeddings):
                embedding_vector = embeddings[embedding_idx]
                embedding_idx += 1

            entities.append(
                DocumentFieldValue(
                    document_id=document_id,
                    field_id=value.field_id,
                    value_text=value.value or "",
                    confidence=value.confidence,
                    page_num=value.page_num,
                    bbox=self._bbox_entries_to_dict(value.bbox),
                    embedding=embedding_vector,
                )
            )

        if not entities:
            logger.warning(
                "ExtractDocumentFieldValuesService: no DocumentFieldValue entities created"
            )
            return

        await self.document_field_values_repository.bulk_create(entities)
        await self._update_value_tsv(document_id)

    def _bbox_entries_to_dict(
        self, entries: Optional[List[BoundingBoxEntry]]
    ) -> Optional[Dict[str, List[float]]]:
        if not entries:
            return None
        result: Dict[str, List[float]] = {}
        for entry in entries:
            result[entry.page] = entry.coords
        return result

    async def _update_value_tsv(self, document_id: int) -> None:
        try:
            update_query = text(
                """
                UPDATE document_field_values
                SET value_tsv = to_tsvector('simple', unaccent(value_text))
                WHERE document_id = :document_id
                """
            )
            await self.document_field_values_repository.session.execute(
                update_query, {"document_id": document_id}
            )
        except Exception as exc:
            logger.warning(
                "ExtractDocumentFieldValuesService: unaccent update failed (%s), fallback",
                exc,
            )
            fallback_query = text(
                """
                UPDATE document_field_values
                SET value_tsv = to_tsvector('simple', value_text)
                WHERE document_id = :document_id
                """
            )
            await self.document_field_values_repository.session.execute(
                fallback_query, {"document_id": document_id}
            )

    async def _create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.openai_client.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=texts,
                dimensions=self.EMBEDDING_DIMENSIONS,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise AppError(
                status_code=500, message=f"Failed to create embeddings: {exc}"
            ) from exc

    def _split_fields_by_extraction_method(
        self,
        extraction_fields: List[ExtractionField],
    ) -> Tuple[List[ExtractionField], List[ExtractionField]]:
        ai_fields = []
        keyword_fields = []

        for field in extraction_fields:
            if field.use_ai:
                ai_fields.append(field)
            else:
                keyword_fields.append(field)

        logger.info(
            "ExtractDocumentFieldValuesService: field split result ai=%d keyword=%d",
            len(ai_fields),
            len(keyword_fields),
        )
        return ai_fields, keyword_fields

    def _is_placeholder_value(self, value: str) -> bool:
        """Check if value is a placeholder (underscores, dots, empty, etc.)"""
        if not value:
            return True
        # Remove whitespace and check
        cleaned = value.strip()
        if not cleaned:
            return True
        # Check for placeholder patterns
        placeholder_patterns = [
            r'^[_\-\.]+$',  # Only underscores, dashes, or dots
            r'^_{2,}$',     # Multiple underscores
            r'^\.*$',       # Only dots
            r'^\-+$',       # Only dashes
            r'^N/A$',       # N/A
            r'^n/a$',       # n/a
            r'^None$',      # None
            r'^null$',      # null
            r'^\[.*\]$',    # Just brackets with placeholder
        ]
        import re
        for pattern in placeholder_patterns:
            if re.match(pattern, cleaned, re.IGNORECASE):
                return True
        # Too short values are likely placeholders
        if len(cleaned) < 3 and not cleaned.isalnum():
            return True
        return False

    def _empty_value(self, field: ExtractionField) -> ExtractedFieldValue:
        return ExtractedFieldValue(
            field_id=field.id,
            value="",
            confidence=0.0,
            page_num=None,
            bbox=None,
        )

    def _empty_values(self, fields: List[ExtractionField]) -> List[ExtractedFieldValue]:
        return [self._empty_value(field) for field in fields]

    async def process_document_console(
        self, document_id: int, extraction_field_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Process document for Document AI Console.
        Uses only Custom Extractor and returns results grouped by page.
        Supports multiple occurrences of same field across pages.
        """
        document = await self.document_repository.get_one(
            where=[Document.id == document_id]
        )
        if not document:
            raise AppError(status_code=404, message=ErrorMessages.DOCUMENT_NOT_FOUND)

        # Get extraction fields for Custom Extractor
        extraction_fields = await self.extraction_fields_repository.get_all(
            where=[
                document.type == any_(ExtractionField.document_types),
                ExtractionField.id.in_(extraction_field_ids),
            ]
        )

        # Prepare schema for Custom Extractor
        custom_schema = []
        field_map = {}
        for field in extraction_fields:
            if field.use_ai:
                # Use OPTIONAL_MULTIPLE to allow multiple occurrences
                occurrence = (
                    field.occurrence
                    if hasattr(field, "occurrence")
                    else "OPTIONAL_MULTIPLE"
                )
                custom_schema.append(
                    CustomExtractorSchemaField(
                        name=field.identifier or field.name,
                        description=self._build_custom_field_description(field),
                        occurrence=occurrence,
                        value_type="string",
                    )
                )
                field_map[field.identifier or field.name] = field

        # Run Custom Extractor
        custom_data = {}
        if custom_schema:
            try:
                custom_response = await asyncio.to_thread(
                    self.document_ai_service.process_with_custom_extractor,
                    document.file_path,
                    custom_schema,
                    document.content_type,
                )
                custom_data = custom_response
            except Exception as e:
                logger.error(f"Custom Extractor failed: {e}")

        # Aggregate results by page and field
        pages_data = {}

        # Helper to init page data
        def get_page_data(page_num):
            if page_num not in pages_data:
                pages_data[page_num] = {
                    "page_number": page_num,
                    "fields": {},  # field_id -> list of occurrences
                }
            return pages_data[page_num]

        # Process Custom Extractor Results
        if custom_data:
            entities = self.document_ai_service.extract_custom_entities(custom_data)
            for entity in entities:
                page_num = entity.get("page_number", 1)
                page_data = get_page_data(page_num)

                # Enrich with field metadata
                field_name = entity.get("type")
                field_def = field_map.get(field_name)

                if field_def:
                    field_id = field_def.id
                    if field_id not in page_data["fields"]:
                        page_data["fields"][field_id] = {
                            "field_id": field_id,
                            "field_name": field_def.name,
                            "field_identifier": field_def.identifier,
                            "occurrences": [],
                        }

                    page_data["fields"][field_id]["occurrences"].append(
                        {
                            "value": entity.get("value"),
                            "confidence": entity.get("confidence"),
                            "bbox": entity.get("bounding_box"),
                        }
                    )

        # Add all requested fields to all pages (even if not found)
        # This allows user to manually add values for missing fields
        for field in extraction_fields:
            if field.use_ai:
                # Ensure field appears on at least page 1 if not found anywhere
                if not any(
                    field.id in page_data["fields"] for page_data in pages_data.values()
                ):
                    page_data = get_page_data(1)
                    if field.id not in page_data["fields"]:
                        page_data["fields"][field.id] = {
                            "field_id": field.id,
                            "field_name": field.name,
                            "field_identifier": field.identifier,
                            "occurrences": [],  # Empty - not found
                        }

        # Convert to list sorted by page number
        sorted_pages = []
        for page_num in sorted(pages_data.keys()):
            page_data = pages_data[page_num]
            # Convert fields dict to list
            page_data["fields"] = list(page_data["fields"].values())
            sorted_pages.append(page_data)

        return {
            "document_id": document_id,
            "pages": sorted_pages,
            "total_pages": len(sorted_pages) if sorted_pages else 1,
        }

    async def save_curated_data(
        self, document_id: int, curated_data: Dict[str, Any]
    ) -> None:
        """
        Save curated data from Document AI Console.
        Handles multiple occurrences of same field across pages.
        Updates document_field_values and generates embeddings for RAG.
        """
        # 1. Wipe previous values
        await self._wipe_previous_values(document_id)

        # 2. Create new DocumentFieldValue entities
        entities = []

        # Process all pages and field occurrences
        pages = curated_data.get("pages", [])
        for page in pages:
            page_num = page.get("page_number", 1)

            # Process each field on this page
            for field_data in page.get("fields", []):
                field_id = field_data.get("field_id")
                if not field_id:
                    continue

                # Process each occurrence of this field on this page
                for occurrence in field_data.get("occurrences", []):
                    value = occurrence.get("value")
                    if not value:  # Skip empty values
                        continue

                    bbox = occurrence.get("bbox")
                    bbox_dict = {str(page_num): bbox} if bbox else None

                    entities.append(
                        DocumentFieldValue(
                            document_id=document_id,
                            field_id=field_id,
                            value_text=value,
                            confidence=occurrence.get("confidence", 1.0),
                            page_num=str(page_num),
                            bbox=bbox_dict,
                        )
                    )

        # 3. Generate embeddings and save
        if entities:
            value_texts = [e.value_text for e in entities]
            embeddings = await self._create_embeddings_batch(value_texts)
            for entity, embedding in zip(entities, embeddings):
                entity.embedding = embedding

            await self.document_field_values_repository.bulk_create(entities)
            await self._update_value_tsv(document_id)

        # 4. Update document status
        document = await self.document_repository.get_one(
            where=[Document.id == document_id]
        )
        if document:
            document.status = DocumentStatus.COMPLETED
            await self.uow.commit()

    async def extract_single_field(
        self,
        document_id: int,
        field_id: int = None,
        custom_field_name: str = None,
        description: str = None,
        occurrence: str = None,
        page_number: int = None,
        extraction_by: str = None,
        type: str = None,
        prompt: str = None,
    ) -> Dict[str, Any]:
        """Extract single field from document using Custom Extractor or Gemini. Supports both extraction fields and custom fields."""
        document = await self.document_repository.get_one(
            where=[Document.id == document_id]
        )
        if not document:
            raise AppError(status_code=404, message=ErrorMessages.DOCUMENT_NOT_FOUND)

        # Determine field configuration
        field = None
        if field_id:
            field = await self.extraction_fields_repository.get_one(
                where=[ExtractionField.id == field_id]
            )
            if not field:
                raise AppError(status_code=404, message="Field not found")

            # Allow overriding params for testing/interactive mode
            if extraction_by:
                field.extraction_by = ExtractionBy(extraction_by)
            if prompt:
                field.prompt = prompt

            identifier = field.identifier or field.name
            field_name = field.name
        elif custom_field_name:
            # Create temporary field object
            identifier = custom_field_name
            field_name = custom_field_name

            # Determine extraction strategy
            strategy = ExtractionBy.DOCUMENT_AI
            if extraction_by:
                try:
                    strategy = ExtractionBy(extraction_by)
                except ValueError:
                    pass

            field = ExtractionField(
                id=0,  # Dummy ID
                name=custom_field_name,
                identifier=custom_field_name,
                prompt=prompt or description,
                short_description=description,
                extraction_by=strategy,
                occurrence=(
                    FieldOccurrence(occurrence)
                    if occurrence
                    else FieldOccurrence.OPTIONAL_ONCE
                ),
                type=ExtractionFieldType(type) if type else ExtractionFieldType.TEXT,
                document_types=[document.type] if document.type else [],
            )
        else:
            raise AppError(
                status_code=400,
                message="Either field_id or custom_field_name must be provided",
            )

        # Execute extraction based on strategy
        try:
            if field.extraction_by == ExtractionBy.GEMINI_AI:
                results = await self._extract_with_gemini(document, [field])
                # For Gemini AI, don't filter by page - it searches all pages
                logger.info(f"[extract_single_field] Gemini AI returned {len(results)} results for field '{field_name}'")
                for idx, r in enumerate(results):
                    logger.info(
                        f"[extract_single_field] Result {idx+1}: page={r.page_num}, "
                        f"value_length={len(r.value) if r.value else 0}, "
                        f"value_preview={r.value[:100] if r.value else 'None'}, "
                        f"bbox={'found' if r.bbox else 'not found'}"
                    )
            else:
                results = await self._extract_with_document_ai(document, [field])
                # Filter by page if specified (for Document AI)
                if page_number is not None:
                    results = [
                        r
                        for r in results
                        if r.page_num and str(page_number) in r.page_num.split(",")
                    ]

            if not results:
                logger.warning(f"[extract_single_field] No results found for field '{field_name}'")
                result = {
                    "field_id": field_id,
                    "custom_field_name": custom_field_name,
                    "field_name": field_name,
                    "value": None,
                    "found": False,
                }
                if page_number:
                    result["page_number"] = page_number
                return result

            # Save all occurrences to DB immediately (like prototype does)
            # This allows frontend to load them separately
            if results:
                # Convert to field_values format for saving
                field_values_to_save = []
                for r in results:
                    # Get page number for this occurrence
                    page_num = int(r.page_num) if r.page_num and r.page_num.isdigit() else None
                    
                    # Convert bbox to dict format: {page: [x1, y1, x2, y2]}
                    bbox_dict = None
                    if r.bbox and page_num:
                        # r.bbox is List[BoundingBoxEntry], each entry has page and coords
                        # We need to create dict with page as key
                        bbox_dict = {}
                        for entry in r.bbox:
                            # entry.page is string like "10", entry.coords is [x1, y1, x2, y2]
                            bbox_dict[entry.page] = entry.coords
                        # If we have page_num but no matching entry, use first entry's coords
                        if str(page_num) not in bbox_dict and r.bbox:
                            bbox_dict[str(page_num)] = r.bbox[0].coords
                    
                    field_values_to_save.append({
                        "field_id": field_id,
                        "custom_field_name": custom_field_name if not field_id else None,
                        "value": r.value,
                        "confidence": r.confidence or 1.0,
                        "bbox": bbox_dict,  # This is already {page: [x1, y1, x2, y2]}
                        "page_number": page_num,
                    })
                
                
                # Save all occurrences
                try:
                    await self.save_field_values_interactive(document_id, field_values_to_save)
                    logger.info(f"[extract_single_field] Saved {len(field_values_to_save)} occurrences to DB for field '{field_name}'")
                except Exception as e:
                    logger.error(f"[extract_single_field] Failed to save occurrences: {e}", exc_info=True)

            # Return first result for backward compatibility with frontend
            # Frontend will reload all fields from /documents/{id}/fields endpoint
            extracted_value = results[0]

            # Convert bbox entries to simple list [x1, y1, x2, y2] for frontend
            bbox = None
            page_num = None

            if extracted_value.bbox:
                # Get first bbox entry
                entry = extracted_value.bbox[0]
                bbox = entry.coords  # [x1, y1, x2, y2]
                page_num = int(entry.page) if entry.page.isdigit() else None

            # Fallback for page number if not in bbox
            if page_num is None and extracted_value.page_num:
                try:
                    page_num = int(extracted_value.page_num.split(",")[0])
                except:
                    pass

            result = {
                "field_id": field_id,
                "custom_field_name": custom_field_name,
                "field_name": field_name,
                "value": extracted_value.value,
                "confidence": extracted_value.confidence,
                "bbox": bbox,
                "page_number": page_num,
                "found": True,
            }
            
            # Log all occurrences for debugging
            if len(results) > 1:
                result["occurrences_count"] = len(results)
                logger.info(
                    f"[extract_single_field] Field '{field_name}' has {len(results)} occurrences. "
                    f"All saved to DB. Returning first (page {page_num})."
                )
                for idx, r in enumerate(results):
                    r_page = int(r.page_num) if r.page_num and r.page_num.isdigit() else None
                    logger.info(
                        f"[extract_single_field] Occurrence {idx+1}: page={r_page}, "
                        f"value_length={len(r.value) if r.value else 0}, "
                        f"value_preview={r.value[:150] if r.value else 'None'}"
                    )
            
            return result

        except Exception as e:
            logger.error(
                f"Error extracting field {field_id or custom_field_name}: {e}",
                exc_info=True,
            )
            result = {
                "field_id": field_id,
                "custom_field_name": custom_field_name,
                "field_name": field_name,
                "value": None,
                "found": False,
            }
            if page_number:
                result["page_number"] = page_number
            return result

    async def save_field_values_interactive(
        self, document_id: int, field_values: List[Dict[str, Any]]
    ) -> None:
        """Save field values from interactive view. Supports both extraction fields and custom fields.
        If field_value has 'id' - updates existing record, otherwise creates new one.
        Same field can appear multiple times on same page - each is a separate record.
        NO uniqueness constraint on field_id + page_num - allows duplicates."""

        entities_to_create = []
        updates_to_apply = []

        for fv in field_values:
            value = fv.get("value")
            # Skip if value is empty, "----", "processing", or None
            if not value or value == "----" or value == "processing":
                continue

            # bbox format handling:
            # - From extract_single_field: already dict {page: [x1, y1, x2, y2]} - use as is
            # - From frontend: list [x1, y1, x2, y2] - convert to dict
            bbox_dict = fv.get("bbox")
            if bbox_dict is None:
                bbox_dict = None
            elif isinstance(bbox_dict, dict):
                # Already a dict from extract_single_field, use as is
                # Should be {page: [x1, y1, x2, y2]} format, don't wrap again!
                pass
            elif isinstance(bbox_dict, list) and fv.get("page_number"):
                # Frontend sends list [x1, y1, x2, y2], convert to dict
                bbox_dict = {str(fv["page_number"]): bbox_dict}
            else:
                # Unknown format, try to convert if we have page_number
                if fv.get("page_number"):
                    bbox_dict = {str(fv["page_number"]): bbox_dict}

            field_id = fv.get("field_id")
            custom_field_name = fv.get("custom_field_name") if not field_id else None

            # Handle None value_text - set to empty string for nullable field
            value_text = fv["value"]
            if value_text is None:
                value_text = ""  # Empty string for nullable field

            # If record has id, update it; otherwise create new
            record_id = fv.get("id")
            if record_id:
                # Update existing record
                update_data = {
                    "value_text": value_text,
                    "confidence": fv.get("confidence", 1.0),
                    "page_num": (
                        str(fv.get("page_number", ""))
                        if fv.get("page_number")
                        else None
                    ),
                    "bbox": bbox_dict,
                }
                updates_to_apply.append((record_id, update_data))
            else:
                # Create new record
                entity = DocumentFieldValue(
                    document_id=document_id,
                    field_id=field_id,
                    custom_field_name=custom_field_name,
                    value_text=value_text,
                    confidence=fv.get("confidence", 1.0),
                    page_num=(
                        str(fv.get("page_number", ""))
                        if fv.get("page_number")
                        else None
                    ),
                    bbox=bbox_dict,
                )
                entities_to_create.append(entity)

        # Generate embeddings for new entities
        if entities_to_create:
            value_texts = [e.value_text for e in entities_to_create]
            embeddings = await self._create_embeddings_batch(value_texts)
            for entity, embedding in zip(entities_to_create, embeddings):
                entity.embedding = embedding

            await self.document_field_values_repository.bulk_create(entities_to_create)

        # Update existing entities
        if updates_to_apply:
            # Generate embeddings for updated values
            value_texts = [
                update_data["value_text"] for _, update_data in updates_to_apply
            ]
            embeddings = await self._create_embeddings_batch(value_texts)

            # Update each entity - use direct SQL update since DocumentFieldValue.id is int, not UUID
            from sqlalchemy import update as sql_update

            for (record_id, update_data), embedding in zip(
                updates_to_apply, embeddings
            ):
                update_data["embedding"] = embedding
                # Convert record_id to int if it's not already
                record_id_int = int(record_id) if record_id else None
                if record_id_int:
                    stmt = (
                        sql_update(DocumentFieldValue)
                        .where(DocumentFieldValue.id == record_id_int)
                        .values(**update_data)
                    )
                    await self.uow.session.execute(stmt)

        # Update TSV for search
        await self._update_value_tsv(document_id)

        # Update document status
        document = await self.document_repository.get_one(
            where=[Document.id == document_id]
        )
        if document:
            document.status = DocumentStatus.COMPLETED
            await self.uow.commit()
