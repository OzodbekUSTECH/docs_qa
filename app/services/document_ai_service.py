"""
Service for processing documents with Google Cloud Document AI.
Supports multiple processor types: Custom Extractor, Form Parser, OCR.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1beta3 as documentai
from google.protobuf.json_format import MessageToDict

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CustomExtractorSchemaField:
    name: str
    description: Optional[str] = None
    occurrence: str = "OPTIONAL_ONCE"
    value_type: str = "string"


class DocumentAIService:
    """Service for processing documents with Document AI processors"""
    
    def __init__(self):
        self.project_id = settings.DOC_AI_PROJECT_ID or "docsqa-478609"
        self.location = settings.DOC_AI_LOCATION or "us"
        self._client: Optional[documentai.DocumentProcessorServiceClient] = None
    
    def _get_client(self) -> documentai.DocumentProcessorServiceClient:
        """Get or create Document AI client"""
        if self._client is None:
            api_endpoint = f"{self.location}-documentai.googleapis.com"
            client_options = ClientOptions(api_endpoint=api_endpoint)
            self._client = documentai.DocumentProcessorServiceClient(
                client_options=client_options
            )
        return self._client
    
    def _get_processor_name(
        self, 
        processor_id: str, 
        processor_version: Optional[str] = None
    ) -> str:
        """Get processor name (path)"""
        client = self._get_client()
        if processor_version:
            return client.processor_version_path(
                self.project_id, self.location, processor_id, processor_version
            )
        return client.processor_path(self.project_id, self.location, processor_id)
    
    def process_with_form_parser(
        self, 
        file_path: str, 
        mime_type: str = "application/pdf"
    ) -> Dict[str, Any]:
        """
        Process document with Form Parser processor.
        Extracts key-value pairs and tables.
        
        Args:
            file_path: Path to the document file
            mime_type: MIME type of the document
            
        Returns:
            Full response as dictionary
        """
        # Form Parser processor ID (you may need to adjust this)
        processor_id = "eb5406357093dae6"  # From test.py
        
        return self._process_document_raw(processor_id, file_path, mime_type)
    
    def _build_ocr_options(self) -> documentai.ProcessOptions:
        """
        Build ProcessOptions with enhanced OCR settings.
        
        Настройки для OCR-процессора:
        - enable_native_pdf_parsing: использовать встроенный текст в digital PDF
        - enable_image_quality_scores: вернуть оценки качества страниц
        - hints.language_hints: подсказать языки (например, английский + русский)
        - advanced_ocr_options: включить legacy_layout (альтернативный layout)
        - enable_symbol: символ-левел (pages.symbols) для более детального анализа
        """
        try:
            ocr_config = documentai.OcrConfig(
                enable_native_pdf_parsing=True,
                enable_image_quality_scores=True,
                enable_symbol=True,  # символ-левел (pages.symbols)
                hints=documentai.OcrConfig.Hints(
                    language_hints=["en", "ru"]  # подставь свои реальные языки
                ),
            )
            
            # Try to add advanced_ocr_options if supported
            try:
                # Check if AdvancedOcrOptions enum exists and use it
                if hasattr(documentai.OcrConfig, 'AdvancedOcrOptions'):
                    legacy_layout_enum = documentai.OcrConfig.AdvancedOcrOptions.LEGACY_LAYOUT
                    ocr_config.advanced_ocr_options = [legacy_layout_enum]
                elif hasattr(ocr_config, 'advanced_ocr_options'):
                    # Try string if enum doesn't exist
                    ocr_config.advanced_ocr_options = ["legacy_layout"]
            except (AttributeError, TypeError) as e:
                logger.debug(
                    "DocumentAIService: advanced_ocr_options not available, skipping: %s",
                    str(e)
                )
                # Continue without advanced_ocr_options
            
            return documentai.ProcessOptions(ocr_config=ocr_config)
        except Exception as e:
            logger.warning(
                "DocumentAIService: failed to build enhanced OCR options, using basic config: %s",
                str(e)
            )
            # Fallback to basic OCR config
            return documentai.ProcessOptions(
                ocr_config=documentai.OcrConfig(
                    enable_native_pdf_parsing=True,
                    enable_image_quality_scores=True,
                    hints=documentai.OcrConfig.Hints(
                        language_hints=["en", "ru", "en-t-i0-handwrit"]
                    ),
                )
            )
    
    def process_with_ocr(
        self, 
        file_path: str, 
        mime_type: str = "application/pdf"
    ) -> Dict[str, Any]:
        """
        Process document with OCR processor with enhanced settings.
        
        Args:
            file_path: Path to the document file
            mime_type: MIME type of the document
            
        Returns:
            Full response as dictionary
        """
        # OCR processor ID (you may need to adjust this)
        processor_id = "436de875fd0c802a"  # From test.py
        
        # Build enhanced OCR options
        process_options = self._build_ocr_options()
        
        # Log OCR settings being used
        if process_options and process_options.ocr_config:
            ocr_cfg = process_options.ocr_config
            logger.info(
                "DocumentAIService: using enhanced OCR settings - "
                "native_pdf_parsing=%s, image_quality_scores=%s, symbol=%s, "
                "language_hints=%s, advanced_options=%s",
                ocr_cfg.enable_native_pdf_parsing,
                ocr_cfg.enable_image_quality_scores,
                ocr_cfg.enable_symbol,
                ocr_cfg.hints.language_hints if ocr_cfg.hints else None,
                list(ocr_cfg.advanced_ocr_options) if hasattr(ocr_cfg, 'advanced_ocr_options') and ocr_cfg.advanced_ocr_options else None
            )
        
        return self._process_document_with_options(
            processor_id, 
            file_path, 
            mime_type, 
            process_options,
            None
        )
    
    def _build_layout_config(self) -> documentai.ProcessOptions:
        """
        Build LayoutConfig with ChunkingConfig for better text grouping.
        This helps avoid splitting text into individual words.
        """
        chunking_config = documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
            chunk_size=1000,  # Chunk size in characters
            include_ancestor_headings=True,  # Include ancestor headings
        )
        
        layout_config = documentai.ProcessOptions.LayoutConfig(
            chunking_config=chunking_config
        )
        
        return documentai.ProcessOptions(layout_config=layout_config)
    
    def process_with_layout(
        self,
        file_path: str,
        mime_type: str = "application/pdf",
        processor_id: Optional[str] = None,
        processor_version: str = "rc",
    ) -> Dict[str, Any]:
        """
        Process document with Layout Parser processor.
        Extracts document content elements like text, tables, and lists,
        and creates context-aware chunks.
        
        Args:
            file_path: Path to the document file
            mime_type: MIME type of the document
            processor_id: Layout Parser processor ID (uses default if None)
            processor_version: Processor version (default: "rc" for LayoutConfig support)
            
        Returns:
            Full response as dictionary with document_layout and chunked_document
        """
        if processor_id is None:
            processor_id = "be93aeec14f92fab"  # Layout Processor ID
        
        # Build LayoutConfig for better text grouping
        process_options = self._build_layout_config()
        
        logger.info(
            "DocumentAIService: using Layout Parser with LayoutConfig "
            "(chunk_size=1000, include_ancestor_headings=True)"
        )
        
        return self._process_document_with_options(
            processor_id,
            file_path,
            mime_type,
            process_options,
            processor_version
        )
    
    def process_with_custom_extractor(
        self,
        file_path: str,
        schema_fields: List[Union[str, CustomExtractorSchemaField]],
        mime_type: str = "application/pdf",
        processor_id: Optional[str] = None,
        processor_version: str = "rc",
    ) -> Dict[str, Any]:
        """
        Process document with Custom Extractor processor.
        Extracts custom fields defined in schema_override.
        
        Args:
            file_path: Path to the document file
            field_names: List of field names to extract
            mime_type: MIME type of the document
            processor_id: Custom Extractor processor ID (uses default from config if None)
            processor_version: Processor version (default: "rc")
            
        Returns:
            Full response as dictionary
        """
        if processor_id is None:
            processor_id = "d7107a9f973a8448"
        
        normalized_fields = self._normalize_schema_fields(schema_fields)
        process_options = self._build_schema_override(normalized_fields)
        
        return self._process_document_with_options(
            processor_id, 
            file_path, 
            mime_type, 
            process_options,
            processor_version
        )
    
    def _normalize_schema_fields(
        self,
        schema_fields: List[Union[str, CustomExtractorSchemaField, Dict[str, Any]]],
    ) -> List[CustomExtractorSchemaField]:
        normalized: List[CustomExtractorSchemaField] = []
        for field in schema_fields:
            if isinstance(field, CustomExtractorSchemaField):
                normalized.append(field)
            elif isinstance(field, str) and field.strip():
                normalized.append(CustomExtractorSchemaField(name=field.strip()))
            elif isinstance(field, dict):
                name = (field.get("name") or field.get("field_name") or "").strip()
                if not name:
                    continue
                normalized.append(
                    CustomExtractorSchemaField(
                        name=name,
                        description=field.get("description"),
                        occurrence=field.get("occurrence") or field.get("occurrence_type") or "OPTIONAL_ONCE",
                        value_type=(field.get("value_type") or field.get("type") or "string").lower(),
                    )
                )
        return normalized

    def _build_schema_override(
        self,
        schema_fields: List[CustomExtractorSchemaField],
    ) -> documentai.ProcessOptions:
        """
        Build schema_override for Custom Extractor.
        All fields are set as string + OPTIONAL_ONCE.
        """
        properties = []
        for field in schema_fields:
            if not field.name:
                continue

            properties.append(
                documentai.DocumentSchema.EntityType.Property(
                    name=field.name,
                    display_name=field.name,
                    description=field.description or field.name,
                    value_type=(field.value_type or "string"),
                    occurrence_type=self._map_occurrence(field.occurrence),
                )
            )
        
        schema = documentai.DocumentSchema(
            display_name="Dynamic CDE schema",
            description="Per-request schema override",
            entity_types=[
                documentai.DocumentSchema.EntityType(
                    name="custom_extraction_document_type",
                    base_types=["document"],
                    properties=properties,
                )
            ],
        )
        
        return documentai.ProcessOptions(schema_override=schema)

    @staticmethod
    def _map_occurrence(
        occurrence: Optional[str],
    ) -> documentai.DocumentSchema.EntityType.Property.OccurrenceType:
        occ = (occurrence or "OPTIONAL_ONCE").upper()
        occurrence_enum = documentai.DocumentSchema.EntityType.Property.OccurrenceType
        mapping = {
            "OPTIONAL_ONCE": occurrence_enum.OPTIONAL_ONCE,
            "OPTIONAL_MULTIPLE": occurrence_enum.OPTIONAL_MULTIPLE,
            "REQUIRED_ONCE": occurrence_enum.REQUIRED_ONCE,
            "REQUIRED_MULTIPLE": occurrence_enum.REQUIRED_MULTIPLE,
        }
        return mapping.get(occ, occurrence_enum.OPTIONAL_ONCE)
    
    def _process_document_raw(
        self,
        processor_id: str,
        file_path: str,
        mime_type: str
    ) -> Dict[str, Any]:
        """Process document without custom options"""
        return self._process_document_with_options(
            processor_id, file_path, mime_type, None, None
        )
    
    def _process_document_with_options(
        self,
        processor_id: str,
        file_path: str,
        mime_type: str,
        process_options: Optional[documentai.ProcessOptions] = None,
        processor_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Internal method to process document"""
        client = self._get_client()
        
        name = self._get_processor_name(processor_id, processor_version)
        
        with open(file_path, "rb") as f:
            content = f.read()
        
        raw_document = documentai.RawDocument(
            content=content,
            mime_type=mime_type
        )
        
        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document,
            process_options=process_options
        )
        
        result = client.process_document(request=request)
        
        # Convert to dict
        result_dict = MessageToDict(
            result._pb, 
            preserving_proto_field_name=True
        )
        
        return result_dict
    
    @staticmethod
    def extract_key_value_pairs(document_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract key-value pairs from Form Parser response.
        Checks both form_fields in pages and entities.
        
        Returns:
            List of dicts with 'key', 'value', 'confidence', 'bounding_box', 'key_bbox', 'value_bbox'
        """
        key_value_pairs = []
        full_text = document_dict.get("text", "")
        
        # First, try to extract from form_fields in pages (Form Parser)
        # Handle both direct pages and nested in document
        pages = document_dict.get("pages", [])
        if not pages and "document" in document_dict:
            pages = document_dict["document"].get("pages", [])
        
        for page in pages:
            page_number = page.get("pageNumber", page.get("page_number", 1))
            form_fields = page.get("formFields", []) or page.get("form_fields", [])
            
            for form_field in form_fields:
                # Extract key (field_name) - handle both camelCase and snake_case
                field_name = form_field.get("fieldName", {}) or form_field.get("field_name", {})
                # Try content first, then text_anchor
                text_anchor = field_name.get("textAnchor", {}) or field_name.get("text_anchor", {})
                key_text = text_anchor.get("content", "")
                if not key_text:
                    key_text = DocumentAIService._extract_text_from_layout(
                        field_name, full_text
                    )
                
                # Extract value (field_value) - handle both camelCase and snake_case
                field_value = form_field.get("fieldValue", {}) or form_field.get("field_value", {})
                text_anchor = field_value.get("textAnchor", {}) or field_value.get("text_anchor", {})
                value_text = text_anchor.get("content", "")
                if not value_text:
                    value_text = DocumentAIService._extract_text_from_layout(
                        field_value, full_text
                    )
                
                # Clean up text
                key_text = key_text.strip().rstrip(":")
                value_text = value_text.strip()
                
                if not key_text and not value_text:
                    continue
                
                # Extract bounding boxes - handle both camelCase and snake_case
                bounding_poly = field_name.get("boundingPoly", {}) or field_name.get("bounding_poly", {})
                key_bbox = DocumentAIService._extract_bounding_box(bounding_poly)
                bounding_poly = field_value.get("boundingPoly", {}) or field_value.get("bounding_poly", {})
                value_bbox = DocumentAIService._extract_bounding_box(bounding_poly)
                
                # Use value bbox as main, or key bbox if value is missing
                bounding_box = value_bbox or key_bbox
                
                key_value_pairs.append({
                    "key": key_text,
                    "value": value_text,
                    "confidence": form_field.get("confidence", 0.0) or field_name.get("confidence", 0.0) or field_value.get("confidence", 0.0),
                    "bounding_box": bounding_box,
                    "key_bbox": key_bbox,
                    "value_bbox": value_bbox,
                    "page_number": page_number
                })
        
        # Also check entities (for other processors or additional data)
        entities = document_dict.get("entities", [])
        for entity in entities:
            entity_type = entity.get("type_", "")
            if not entity_type or entity_type.startswith("generic_"):
                continue
            if entity_type and ":" in entity_type:
                # Some form parsers return key-value as "key:value" in type
                parts = entity_type.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                else:
                    key = entity_type
                    value = entity.get("mentionText", "") or DocumentAIService._extract_text_from_layout(
                        entity.get("textAnchor", {}), full_text
                    )
            else:
                # Try to get from properties
                properties = entity.get("properties", [])
                key = entity_type or ""
                value = ""
                for prop in properties:
                    if prop.get("type_") == "value":
                        value = prop.get("mentionText", "") or DocumentAIService._extract_text_from_layout(
                            prop.get("textAnchor", {}), full_text
                        )
            
            if not key and not value:
                continue
            
            # Extract bounding box
            bounding_box = None
            page_anchor = entity.get("pageAnchor", {})
            if page_anchor:
                page_refs = page_anchor.get("pageRefs", [])
                if page_refs:
                    bounding_poly = page_refs[0].get("boundingPoly", {})
                    if bounding_poly:
                        vertices = bounding_poly.get("normalizedVertices", [])
                        if vertices:
                            bounding_box = {
                                "x1": vertices[0].get("x", 0),
                                "y1": vertices[0].get("y", 0),
                                "x2": vertices[-1].get("x", 1),
                                "y2": vertices[-1].get("y", 1),
                            }
            
            key_value_pairs.append({
                "key": key,
                "value": value,
                "confidence": entity.get("confidence", 0.0),
                "bounding_box": bounding_box,
                "page_number": page_anchor.get("pageRefs", [{}])[0].get("page", 1) if page_anchor.get("pageRefs") else 1
            })
        
        return key_value_pairs
    
    @staticmethod
    def _extract_text_from_layout(layout_dict: Dict[str, Any], full_text: str) -> str:
        """Extract text from layout using textAnchor"""
        if not layout_dict:
            return ""
        
        # Try content first (direct text)
        if isinstance(layout_dict, dict):
            text_anchor = layout_dict.get("textAnchor", {})
            if text_anchor and "content" in text_anchor:
                return text_anchor["content"].strip()
        
        layout = layout_dict.get("layout", layout_dict)
        text_anchor = layout.get("textAnchor", {}) or layout.get("text_anchor", {})
        
        # Try content in text_anchor
        if text_anchor and "content" in text_anchor:
            return text_anchor["content"].strip()
        
        # Fallback to text_segments
        text_segments = text_anchor.get("textSegments", []) or text_anchor.get("text_segments", [])
        
        if not text_segments:
            return ""
        
        parts = []
        for segment in text_segments:
            start_index = int(segment.get("startIndex", segment.get("start_index", 0)))
            end_index = int(segment.get("endIndex", segment.get("end_index", len(full_text))))
            parts.append(full_text[start_index:end_index])
        
        return "".join(parts).strip()
    
    @staticmethod
    def _extract_bounding_box(bounding_poly: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract normalized bounding box from bounding_poly"""
        if not bounding_poly:
            return None
        
        vertices = bounding_poly.get("normalizedVertices", []) or bounding_poly.get("normalized_vertices", [])
        if not vertices or len(vertices) < 2:
            return None
        
        # Get min/max coordinates
        xs = [v.get("x", 0) for v in vertices if "x" in v]
        ys = [v.get("y", 0) for v in vertices if "y" in v]
        
        if not xs or not ys:
            return None
        
        return {
            "x1": min(xs),
            "y1": min(ys),
            "x2": max(xs),
            "y2": max(ys),
        }
    
    @staticmethod
    def extract_tables(document_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tables from document.
        
        Returns:
            List of tables with rows, cells, and bounding boxes
        """
        tables = []
        full_text = document_dict.get("text", "")
        # Handle both direct pages and nested in document
        pages = document_dict.get("pages", [])
        if not pages and "document" in document_dict:
            pages = document_dict["document"].get("pages", [])
        
        for page in pages:
            page_number = page.get("pageNumber", page.get("page_number", 1))
            page_tables = page.get("tables", [])
            
            for table_idx, table in enumerate(page_tables):
                table_data = {
                    "table_index": table_idx + 1,
                    "page_number": page_number,
                    "rows": [],
                    "bounding_box": None
                }
                
                # Extract bounding box from layout
                layout = table.get("layout", {})
                bounding_poly = layout.get("boundingPoly", {}) or layout.get("bounding_poly", {})
                table_data["bounding_box"] = DocumentAIService._extract_bounding_box(bounding_poly)
                
                # Extract header rows
                header_rows = table.get("headerRows", []) or table.get("header_rows", [])
                for header_row in header_rows:
                    row = DocumentAIService._extract_row_cells(
                        header_row.get("cells", []), full_text
                    )
                    if row:
                        table_data["rows"].append(row)
                
                # Extract body rows
                body_rows = table.get("bodyRows", []) or table.get("body_rows", [])
                for body_row in body_rows:
                    row = DocumentAIService._extract_row_cells(
                        body_row.get("cells", []), full_text
                    )
                    if row:
                        table_data["rows"].append(row)
                
                if table_data["rows"]:
                    tables.append(table_data)
        
        return tables
    
    @staticmethod
    def _extract_row_cells(cells: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        """Extract cell data from row"""
        row_cells = []
        for cell in cells:
            layout = cell.get("layout", {})
            cell_text = DocumentAIService._extract_text_from_layout(layout, full_text)
            
            # Extract bounding box
            bounding_poly = layout.get("boundingPoly", {}) or layout.get("bounding_poly", {})
            bounding_box = DocumentAIService._extract_bounding_box(bounding_poly)
            
            row_cells.append({
                "text": cell_text,
                "bounding_box": bounding_box,
                "row_span": cell.get("rowSpan", cell.get("row_span", 1)),
                "col_span": cell.get("colSpan", cell.get("col_span", 1))
            })
        
        return row_cells
    
    @staticmethod
    def extract_custom_entities(document_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract custom entities from Custom Extractor response.
        
        Returns:
            List of entities with 'type', 'value', 'confidence', 'bounding_box', 'page_number', 'page_anchor'
        """
        entities = []
        full_text = document_dict.get("text", "")
        
        # Get document from response (might be nested)
        document = document_dict.get("document", document_dict)
        doc_entities = document.get("entities", [])
        
        for entity in doc_entities:
            entity_type = entity.get("type_", "")
            mention_text = entity.get("mentionText", "") or entity.get("mention_text", "")
            
            # Try to get text from textAnchor if mention_text is empty
            if not mention_text:
                text_anchor = entity.get("textAnchor", {}) or entity.get("text_anchor", {})
                mention_text = DocumentAIService._extract_text_from_layout(text_anchor, full_text)
            
            # Extract bounding box from page_anchor
            bounding_box = None
            page_number = 1
            page_anchor = entity.get("pageAnchor", {}) or entity.get("page_anchor", {})
            if page_anchor:
                page_refs = page_anchor.get("pageRefs", []) or page_anchor.get("page_refs", [])
                if page_refs:
                    page_ref = page_refs[0]
                    # Page is 0-indexed in response, convert to 1-indexed
                    page_number = int(page_ref.get("page", 0)) + 1
                    bounding_poly = page_ref.get("boundingPoly", {}) or page_ref.get("bounding_poly", {})
                    bounding_box = DocumentAIService._extract_bounding_box(bounding_poly)
            
            entities.append({
                "type": entity_type,
                "value": mention_text.strip(),
                "confidence": entity.get("confidence", 0.0),
                "bounding_box": bounding_box,
                "page_number": page_number,
                "page_anchor": page_anchor  # Include for frontend
            })
        
        return entities
    
    @staticmethod
    def extract_region_from_layout(
        document_dict: Dict[str, Any],
        page_number: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> Dict[str, Any]:
        """
        Extract text and tables from a specific region of a page using Layout Parser results.
        
        Args:
            document_dict: Full Document AI Layout Parser response
            page_number: Page number (1-indexed)
            x1, y1, x2, y2: Normalized coordinates (0-1) of the region
            
        Returns:
            Dictionary with 'text', 'tables', and 'markdown' fields
        """
        document = document_dict.get("document", document_dict)
        layout = document.get("document_layout", {})
        blocks = layout.get("blocks", [])
        pages = document.get("pages", [])
        
        # Find target page
        target_page = None
        for page in pages:
            page_num = page.get("pageNumber", page.get("page_number", 1))
            if page_num == page_number:
                target_page = page
                break
        
        if not target_page:
            return {"text": "", "tables": [], "markdown": ""}
        
        # Extract text blocks and tables that intersect with the region
        text_parts = []
        tables = []
        markdown_parts = []
        region_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        
        # First, try to extract text from paragraphs/tokens on the page (more accurate coordinates)
        paragraphs = target_page.get("paragraphs", [])
        for para in paragraphs:
            layout_obj = para.get("layout", {})
            bounding_poly = layout_obj.get("boundingPoly", {}) or layout_obj.get("bounding_poly", {})
            bbox = DocumentAIService._extract_bounding_box(bounding_poly)
            
            # Check if paragraph intersects with region
            if bbox and DocumentAIService._bboxes_intersect(bbox, region_bbox):
                para_text = DocumentAIService._extract_text_from_layout(layout_obj, document.get("text", ""))
                if para_text.strip():
                    text_parts.append(para_text.strip())
                    markdown_parts.append(para_text.strip())
        
        # Also process layout blocks for structured content (headers, lists, etc.)
        for block in blocks:
            block_page_span = block.get("page_span", {})
            block_page_start = block_page_span.get("page_start", 1)
            block_page_end = block_page_span.get("page_end", block_page_start)
            
            # Skip blocks not on target page
            if not (block_page_start <= page_number <= block_page_end):
                continue
            
            # Check text blocks - try to match with paragraphs we already found
            if "text_block" in block:
                text_block = block.get("text_block", {})
                text = text_block.get("text", "").strip()
                
                if text:
                    # Check if this text is already included from paragraphs
                    # If not, include it (blocks provide structure like headers/lists)
                    type_ = text_block.get("type_", "")
                    formatted_text = text
                    
                    if type_ == "header" or type_ == "heading-1":
                        formatted_text = f"# {text}"
                    elif type_ == "heading-2":
                        formatted_text = f"## {text}"
                    elif type_ == "heading-3":
                        formatted_text = f"### {text}"
                    elif type_ == "list-item":
                        formatted_text = f"* {text}"
                    
                    # Only add if not already included from paragraphs
                    if formatted_text not in markdown_parts:
                        text_parts.append(text)
                        markdown_parts.append(formatted_text)
            
            # Check table blocks
            if "table_block" in block:
                table_block = block.get("table_block", {})
                # Extract table data
                table_data = DocumentAIService._extract_table_from_block(
                    table_block, document.get("text", ""), page_number
                )
                if table_data:
                    # Check if table intersects with region (simplified check)
                    # For now, include all tables on the page
                    tables.append(table_data)
                    # Convert table to markdown
                    markdown_parts.append(DocumentAIService._table_to_markdown(table_data))
        
        # Also check tables from pages array (Form Parser tables)
        page_tables = target_page.get("tables", [])
        for table_idx, table in enumerate(page_tables):
            layout_obj = table.get("layout", {})
            bounding_poly = layout_obj.get("boundingPoly", {}) or layout_obj.get("bounding_poly", {})
            bbox = DocumentAIService._extract_bounding_box(bounding_poly)
            
            # Check if table bounding box intersects with region
            if bbox and DocumentAIService._bboxes_intersect(bbox, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}):
                table_data = {
                    "table_index": table_idx + 1,
                    "page_number": page_number,
                    "rows": [],
                    "bounding_box": bbox
                }
                
                full_text = document.get("text", "")
                header_rows = table.get("headerRows", []) or table.get("header_rows", [])
                for header_row in header_rows:
                    row = DocumentAIService._extract_row_cells(
                        header_row.get("cells", []), full_text
                    )
                    if row:
                        table_data["rows"].append(row)
                
                body_rows = table.get("bodyRows", []) or table.get("body_rows", [])
                for body_row in body_rows:
                    row = DocumentAIService._extract_row_cells(
                        body_row.get("cells", []), full_text
                    )
                    if row:
                        table_data["rows"].append(row)
                
                if table_data["rows"]:
                    tables.append(table_data)
                    markdown_parts.append(DocumentAIService._table_to_markdown(table_data))
        
        return {
            "text": "\n".join(text_parts),
            "tables": tables,
            "markdown": "\n\n".join(markdown_parts)
        }
    
    @staticmethod
    def _bboxes_intersect(bbox1: Dict[str, float], bbox2: Dict[str, float]) -> bool:
        """Check if two bounding boxes intersect"""
        return not (bbox1["x2"] < bbox2["x1"] or bbox1["x1"] > bbox2["x2"] or
                    bbox1["y2"] < bbox2["y1"] or bbox1["y1"] > bbox2["y2"])
    
    @staticmethod
    def _extract_table_from_block(
        table_block: Dict[str, Any],
        full_text: str,
        page_number: int
    ) -> Optional[Dict[str, Any]]:
        """Extract table data from table_block"""
        # This is a simplified extraction - can be enhanced
        return None  # For now, we use tables from pages array
    
    @staticmethod
    def _table_to_markdown(table_data: Dict[str, Any]) -> str:
        """Convert table data to markdown format"""
        if not table_data.get("rows"):
            return ""
        
        markdown_lines = []
        for row_idx, row in enumerate(table_data["rows"]):
            cells = [cell.get("text", "").replace("|", "\\|") for cell in row]
            markdown_lines.append("| " + " | ".join(cells) + " |")
            
            # Add header separator after first row
            if row_idx == 0:
                markdown_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        
        return "\n".join(markdown_lines)
    
    @staticmethod
    def _table_block_to_markdown(table_block: Dict[str, Any]) -> str:
        """Convert table_block from Layout Parser to markdown format"""
        def get_cell_text(cell: Dict[str, Any]) -> str:
            """Extract text from a table cell"""
            blocks = cell.get("blocks", [])
            texts = []
            for b in blocks:
                if "text_block" in b:
                    text = b["text_block"].get("text", "").strip()
                    if text:
                        texts.append(text)
            return " ".join(texts).strip()
        
        rows = []
        
        # Extract header rows
        header_rows = table_block.get("header_rows", [])
        for header_row in header_rows:
            cells = header_row.get("cells", [])
            row_cells = [get_cell_text(cell) for cell in cells]
            if any(row_cells):  # Only add if at least one cell has text
                rows.append(row_cells)
        
        # Extract body rows
        body_rows = table_block.get("body_rows", [])
        for body_row in body_rows:
            cells = body_row.get("cells", [])
            row_cells = [get_cell_text(cell) for cell in cells]
            if any(row_cells):  # Only add if at least one cell has text
                rows.append(row_cells)
        
        if not rows:
            return ""
        
        # Convert to markdown
        markdown_lines = []
        for row_idx, row_cells in enumerate(rows):
            # Escape pipe characters
            escaped_cells = [cell.replace("|", "\\|") for cell in row_cells]
            markdown_lines.append("| " + " | ".join(escaped_cells) + " |")
            
            # Add header separator after first row
            if row_idx == 0:
                markdown_lines.append("| " + " | ".join(["---"] * len(escaped_cells)) + " |")
        
        return "\n".join(markdown_lines)
    

