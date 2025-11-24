"""
Service for extracting field values from markdown text.
Works with Gemini OCR markdown output for manual extraction.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from app.entities.extracion_fields import ExtractionField
from app.dto.field_extraction import BoundingBoxEntry, ExtractedFieldValue
from app.utils.enums import ExtractionFieldType

logger = logging.getLogger(__name__)


@dataclass
class OCRTextElement:
    """Represents a text element with coordinates from OCR"""
    text: str
    start_index: int
    end_index: int
    page_number: int
    bbox: Optional[List[float]]  # [x1, y1, x2, y2] normalized
    confidence: float = 1.0


class OCRFieldExtractionService:
    """
    Service for extracting field values from markdown text.
    Works with Gemini OCR markdown output for manual extraction.
    Supports extraction of clauses, text (key-value pairs), and tables.
    """

    def __init__(self):
        pass

    @staticmethod
    def _extract_bounding_box(bounding_poly: Dict[str, Any]) -> Optional[List[float]]:
        """Extract normalized bounding box from bounding_poly"""
        if not bounding_poly:
            logger.debug("OCRFieldExtractionService: bounding_poly is None or empty")
            return None

        vertices = bounding_poly.get("normalizedVertices", []) or bounding_poly.get("normalized_vertices", [])
        if not vertices:
            logger.debug("OCRFieldExtractionService: no normalizedVertices found in bounding_poly, keys: %s", list(bounding_poly.keys()))
            return None
        
        if len(vertices) < 2:
            logger.debug("OCRFieldExtractionService: insufficient vertices (%d), need at least 2", len(vertices))
            return None

        # Get all x and y coordinates (handle missing values and empty dicts)
        xs = []
        ys = []
        for v in vertices:
            if isinstance(v, dict):
                # Skip empty dicts like {}
                if not v:
                    continue
                if "x" in v:
                    x_val = v.get("x")
                    if x_val is not None:
                        try:
                            xs.append(float(x_val))
                        except (ValueError, TypeError):
                            logger.debug("OCRFieldExtractionService: invalid x value: %s", x_val)
                if "y" in v:
                    y_val = v.get("y")
                    if y_val is not None:
                        try:
                            ys.append(float(y_val))
                        except (ValueError, TypeError):
                            logger.debug("OCRFieldExtractionService: invalid y value: %s", y_val)

        if not xs or not ys:
            logger.warning(
                "OCRFieldExtractionService: insufficient coordinates in bounding_poly, xs=%s (%d), ys=%s (%d), vertices=%s",
                xs, len(xs), ys, len(ys), vertices[:4]  # Log first 4 vertices
            )
            return None

        # Calculate bounding box: [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)

        # Ensure normalized coordinates are in [0.0, 1.0] range
        result = [
            max(0.0, min(1.0, min_x)),
            max(0.0, min(1.0, min_y)),
            max(0.0, min(1.0, max_x)),
            max(0.0, min(1.0, max_y))
        ]

        # Validate result
        if result[2] <= result[0] or result[3] <= result[1]:
            logger.warning(
                "OCRFieldExtractionService: invalid bbox dimensions: %s (width=%f, height=%f)",
                result, result[2] - result[0], result[3] - result[1]
            )
            return None

        logger.debug("OCRFieldExtractionService: extracted bbox %s from %d vertices (xs=%d, ys=%d)", result, len(vertices), len(xs), len(ys))
        return result

    @staticmethod
    def _parse_layout_response(layout_response: Dict[str, Any]) -> Tuple[str, List[OCRTextElement]]:
        """
        Parse Layout Parser response and extract text elements with coordinates.
        Works with both chunks (chunked_document.chunks) and blocks (document_layout.blocks).
        
        Returns:
            Tuple of (full_text, list of OCRTextElement)
        """
        # Handle nested document structure
        document = layout_response.get("document", layout_response)
        
        # Try to get text from chunks first (better quality)
        chunked_doc = document.get("chunked_document", {})
        chunks = chunked_doc.get("chunks", [])
        
        elements_map: Dict[Tuple[int, int, int], OCRTextElement] = {}
        full_text_parts = []
        
        if chunks:
            logger.debug("OCRFieldExtractionService: parsing Layout response with %d chunks", len(chunks))
            
            for chunk in chunks:
                content = chunk.get("content", "").strip()
                if not content:
                    continue
                
                page_span = chunk.get("page_span", {})
                page_start = page_span.get("page_start", 1) if page_span else 1
                page_end = page_span.get("page_end", page_start) if page_span else page_start
                
                # Use first page for element
                page_number = page_start
                
                # Try to find bounding box in page_headers or use approximate
                bbox = None
                page_headers = chunk.get("page_headers", [])
                if page_headers:
                    # Try to extract bbox from first header if available
                    first_header = page_headers[0]
                    # Headers might have layout info, but usually not in chunks
                    pass
                
                # For chunks, we don't have direct bbox, but we can approximate
                # or try to find it in blocks later
                
                full_text_parts.append(content)
                
                # Create element for this chunk
                start_idx = len("\n\n".join(full_text_parts[:-1])) if len(full_text_parts) > 1 else 0
                end_idx = len("\n\n".join(full_text_parts))
                
                key = (int(page_number), int(start_idx), int(end_idx))
                elements_map[key] = OCRTextElement(
                    text=content,
                    start_index=start_idx,
                    end_index=end_idx,
                    page_number=page_number,
                    bbox=bbox,  # Will be filled from blocks if available
                    confidence=1.0,
                )
        
        # Also parse blocks to get better coordinates
        document_layout = document.get("document_layout", {})
        blocks = document_layout.get("blocks", [])
        
        if blocks:
            logger.debug("OCRFieldExtractionService: found %d blocks in document_layout", len(blocks))
            
            def extract_text_from_block(block: Dict[str, Any], page_num: int) -> List[OCRTextElement]:
                """Recursively extract text from block and nested blocks"""
                elements = []
                
                # Check for text_block
                text_block = block.get("text_block", {})
                if text_block:
                    text = text_block.get("text", "").strip()
                    if text:
                        # Try to find bounding box - blocks might have it in layout
                        # But in layout response, blocks usually don't have direct bounding_poly
                        # We'll need to approximate or use page_span
                        bbox = None
                        
                        page_span = block.get("page_span", {})
                        block_page = page_span.get("page_start", page_num) if page_span else page_num
                        
                        elements.append(OCRTextElement(
                            text=text,
                            start_index=0,  # Will be updated when we have full text
                            end_index=len(text),
                            page_number=block_page,
                            bbox=bbox,
                            confidence=1.0,
                        ))
                    
                    # Process nested blocks
                    nested_blocks = text_block.get("blocks", [])
                    for nested in nested_blocks:
                        elements.extend(extract_text_from_block(nested, page_num))
                
                # Check for table_block
                table_block = block.get("table_block", {})
                if table_block:
                    # Extract table as text (markdown format)
                    table_text = OCRFieldExtractionService._extract_table_text(table_block)
                    if table_text:
                        page_span = block.get("page_span", {})
                        block_page = page_span.get("page_start", page_num) if page_span else page_num
                        
                        elements.append(OCRTextElement(
                            text=table_text,
                            start_index=0,
                            end_index=len(table_text),
                            page_number=block_page,
                            bbox=None,  # Tables might have complex layout
                            confidence=1.0,
                        ))
                
                return elements
            
            for block in blocks:
                page_span = block.get("page_span", {})
                page_num = page_span.get("page_start", 1) if page_span else 1
                block_elements = extract_text_from_block(block, page_num)
                
                for elem in block_elements:
                    # Try to merge with existing elements from chunks if text matches
                    key = (elem.page_number, 0, len(elem.text))
                    if key not in elements_map or len(elements_map[key].text) < len(elem.text):
                        # Update bbox if available
                        if elem.bbox:
                            if key in elements_map:
                                elements_map[key].bbox = elem.bbox
                            else:
                                elements_map[key] = elem
        
        full_text = "\n\n".join(full_text_parts) if full_text_parts else document.get("text", "")
        
        # Update start/end indices based on full text
        current_pos = 0
        for key in sorted(elements_map.keys()):
            elem = elements_map[key]
            elem.start_index = current_pos
            current_pos += len(elem.text) + 2  # +2 for "\n\n"
            elem.end_index = current_pos - 2
        
        elements = [elements_map[key] for key in sorted(elements_map.keys())]
        logger.info("OCRFieldExtractionService: parsed %d text elements from Layout response", len(elements))
        return full_text, elements
    
    @staticmethod
    def _extract_table_text(table_block: Dict[str, Any]) -> str:
        """Extract table text in markdown format from table_block"""
        rows = []
        
        # Extract header rows
        header_rows = table_block.get("header_rows", [])
        for header_row in header_rows:
            cells = header_row.get("cells", [])
            row_data = []
            for cell in cells:
                cell_text = OCRFieldExtractionService._extract_cell_text(cell)
                row_data.append(cell_text)
            if row_data:
                rows.append("| " + " | ".join(row_data) + " |")
                rows.append("| " + " | ".join(["---"] * len(row_data)) + " |")
        
        # Extract body rows
        body_rows = table_block.get("body_rows", [])
        for body_row in body_rows:
            cells = body_row.get("cells", [])
            row_data = []
            for cell in cells:
                cell_text = OCRFieldExtractionService._extract_cell_text(cell)
                row_data.append(cell_text)
            if row_data:
                rows.append("| " + " | ".join(row_data) + " |")
        
        return "\n".join(rows)
    
    @staticmethod
    def _extract_cell_text(cell: Dict[str, Any]) -> str:
        """Extract text from table cell"""
        blocks = cell.get("blocks", [])
        texts = []
        for block in blocks:
            text_block = block.get("text_block", {})
            text = text_block.get("text", "").strip()
            if text:
                texts.append(text)
        return " ".join(texts).strip()

    def extract_clause_from_ocr(
        self,
        ocr_elements: List[OCRTextElement],
        full_text: str,
        keywords: List[str],
        layout_response: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, Optional[str], Optional[List[BoundingBoxEntry]]]]:
        """
        Extract clause (paragraph block) from Layout elements by keywords.
        Works with both chunks (markdown) and blocks.
        Handles multi-page paragraphs correctly.
        
        Returns:
            Tuple of (extracted_text, page_num_str, bbox_dict) or None
            page_num_str: "2" or "2,3" or "2-3" for multi-page
            bbox_dict: {"2": [x1, y1, x2, y2], "3": [x1, y1, x2, y2]} or {"combined": [x1, y1, x2, y2]}
        """
        kw_upper = [kw.upper() for kw in keywords]
        logger.info("OCRFieldExtractionService: searching for clause with keywords: %s", keywords)
        
        # First, try to find in full_text using markdown headers (from chunks) - faster and more accurate
        # Chunks already have markdown format with ## headers
        if "##" in full_text or "#" in full_text:
            clause_text = self._extract_clause_from_markdown(full_text, keywords)
            if clause_text:
                # Try to find matching blocks from document_layout to get bbox
                # Match clause text with blocks by finding blocks that contain the text
                bbox_dict = None
                page_num_str = None
                
                if layout_response:
                    bbox_dict, page_num_str = self._find_bbox_from_blocks_by_text(
                        layout_response, clause_text, keywords
                    )
                
                # Fallback: use page info from matching elements if bbox not found
                if not bbox_dict:
                    matching_elements = []
                    clause_words = set(clause_text[:200].upper().split())
                    for elem in ocr_elements:
                        elem_words = set(elem.text[:200].upper().split())
                        if len(clause_words & elem_words) >= 3:
                            matching_elements.append(elem)
                    
                    if matching_elements:
                        pages_set = {e.page_number for e in matching_elements}
                        sorted_pages = sorted(pages_set)
                        if len(sorted_pages) == 1:
                            page_num_str = str(sorted_pages[0])
                        elif len(sorted_pages) == 2:
                            page_num_str = f"{sorted_pages[0]},{sorted_pages[1]}"
                        else:
                            page_num_str = f"{sorted_pages[0]}-{sorted_pages[-1]}"
                
                bbox_entries = self._bbox_dict_to_entries(bbox_dict) if bbox_dict else None
                
                logger.info(
                    "OCRFieldExtractionService: extracted clause from markdown, pages=%s, text_length=%d, bbox=%s",
                    page_num_str, len(clause_text), "yes" if bbox_entries else "no"
                )
                
                return clause_text, page_num_str, bbox_entries
        
        # Fallback: Find starting element that contains keyword (for non-markdown or if markdown failed)
        # First, try to find a numbered header (e.g., "9. PAYMENT")
        start_idx = None
        best_match_score = 0
        
        for i, elem in enumerate(ocr_elements):
            elem_text = elem.text.strip()
            elem_upper = elem_text.upper()
            
            # Check if this element contains any keyword
            keyword_found = False
            match_score = 0
            
            for kw in kw_upper:
                if kw in elem_upper:
                    keyword_found = True
                    # Prefer numbered headers (e.g., "9. PAYMENT")
                    if re.match(r"^\s*\d+\.\s*" + re.escape(kw), elem_upper):
                        match_score = 100  # Best match - numbered header
                    elif re.match(r"^\s*\d+\.\s+.*" + re.escape(kw), elem_upper):
                        match_score = 90  # Numbered header with text before keyword
                    elif self._is_likely_header(elem_text) and kw in elem_upper:
                        match_score = 80  # Header-like element
                    elif kw == elem_upper.strip():
                        match_score = 70  # Exact match
                    else:
                        match_score = 50  # Keyword found but not in header
                    break
            
            if keyword_found and match_score > best_match_score:
                start_idx = i
                best_match_score = match_score
                logger.debug(
                    "OCRFieldExtractionService: found potential match at index %d, page %d, score %d, text: '%s'",
                    i, elem.page_number, match_score, elem_text[:100]
                )
        
        if start_idx is None:
            logger.warning("OCRFieldExtractionService: no element found with keywords %s", keywords)
            # Log first few elements for debugging
            for i, elem in enumerate(ocr_elements[:10]):
                logger.debug("OCRFieldExtractionService: element %d (page %d): '%s'", i, elem.page_number, elem.text[:100])
            return None
        
        logger.info(
            "OCRFieldExtractionService: found clause start at index %d, page %d, text: '%s'",
            start_idx, ocr_elements[start_idx].page_number, ocr_elements[start_idx].text[:100]
        )
        
        # Collect elements until we find next header or end
        # Include the header element itself, then collect all content after it
        collected_elements: List[OCRTextElement] = [ocr_elements[start_idx]]
        start_page = ocr_elements[start_idx].page_number
        max_page = start_page
        pages_set = {start_page}
        
        # Look for next potential header (uppercase line, numbered, etc.)
        # Allow continuation across pages (up to 5 pages max to avoid collecting too much)
        # This handles paragraphs that span multiple pages (e.g., starts on page 2, continues on page 3)
        for i in range(start_idx + 1, len(ocr_elements)):
            elem = ocr_elements[i]
            elem_text = elem.text.strip()
            
            # Skip empty elements but continue
            if not elem_text:
                continue
            
            # Stop ONLY if we find another numbered header (e.g., "10. REACH", "11. FORCE MAJEURE")
            # This is the next section, so we stop here
            # DO NOT stop on uppercase text like "IN US DOLLARS" - that's part of the clause content!
            if self._is_clause_header(elem_text):
                header_num = elem_text.strip().split(".", 1)[0]
                logger.info(
                    "OCRFieldExtractionService: found next numbered header '%s' (section %s) at index %d, stopping collection",
                    elem_text[:50], header_num, i
                )
                break
            
            # Continue collecting everything else - uppercase text like "IN US DOLLARS" is part of the clause!
            
            # Continue collecting if on same page, next page, or pages after (max 5 pages total)
            # This handles paragraphs that span multiple pages
            if elem.page_number <= start_page + 4:
                collected_elements.append(elem)
                max_page = max(max_page, elem.page_number)
                pages_set.add(elem.page_number)
                logger.debug(
                    "OCRFieldExtractionService: collected element %d (page %d): '%s'",
                    i, elem.page_number, elem_text[:80]
                )
            else:
                # If we've moved too far, stop collecting
                logger.debug("OCRFieldExtractionService: moved too far (page %d > %d), stopping collection", elem.page_number, start_page + 4)
                break
        
        logger.info(
            "OCRFieldExtractionService: collected %d elements across pages %s (from index %d to %d)",
            len(collected_elements), sorted(pages_set), start_idx, start_idx + len(collected_elements) - 1
        )
        
        # Combine text
        extracted_text = "\n".join([e.text for e in collected_elements]).strip()
        
        if not extracted_text:
            logger.warning("OCRFieldExtractionService: extracted text is empty for keywords %s", keywords)
            return None
        
        # Build bbox dictionary with coordinates per page
        bbox_dict: dict = {}
        
        # Group elements by page
        elements_by_page: dict[int, List[OCRTextElement]] = {}
        for elem in collected_elements:
            if elem.page_number not in elements_by_page:
                elements_by_page[elem.page_number] = []
            elements_by_page[elem.page_number].append(elem)
        
        # Extract bbox for each page
        for page_num, page_elements in elements_by_page.items():
            page_bboxes = [e.bbox for e in page_elements if e.bbox]
            if page_bboxes:
                combined_page_bbox = self._combine_bboxes(page_bboxes)
                if combined_page_bbox:
                    bbox_dict[str(page_num)] = combined_page_bbox
                    logger.debug(
                        "OCRFieldExtractionService: extracted bbox for page %d: %s (from %d elements)",
                        page_num, combined_page_bbox, len(page_bboxes)
                    )
            else:
                logger.warning(
                    "OCRFieldExtractionService: no bboxes found for page %d (collected %d elements)",
                    page_num, len(page_elements)
                )
        
        # Also add combined bbox for all pages
        all_bboxes = [e.bbox for e in collected_elements if e.bbox]
        if all_bboxes:
            combined_bbox = self._combine_bboxes(all_bboxes)
            if combined_bbox:
                bbox_dict["combined"] = combined_bbox
        
        # Format page_num string
        sorted_pages = sorted(pages_set)
        if len(sorted_pages) == 1:
            page_num_str = str(sorted_pages[0])
        elif len(sorted_pages) == 2:
            page_num_str = f"{sorted_pages[0]},{sorted_pages[1]}"
        else:
            page_num_str = f"{sorted_pages[0]}-{sorted_pages[-1]}"
        
        if not bbox_dict:
            logger.error(
                "OCRFieldExtractionService: no bbox found for clause with keywords %s, collected %d elements across pages %s",
                keywords, len(collected_elements), sorted_pages
            )
            # Log details about collected elements
            for elem in collected_elements:
                logger.debug(
                    "OCRFieldExtractionService: element on page %d, bbox=%s, text_length=%d",
                    elem.page_number, elem.bbox, len(elem.text)
                )
        else:
            logger.info(
                "OCRFieldExtractionService: successfully extracted clause from pages %s, bbox_dict keys: %s, text_length=%d",
                page_num_str, list(bbox_dict.keys()), len(extracted_text)
            )
        
        bbox_entries = self._bbox_dict_to_entries(bbox_dict)
        return extracted_text, page_num_str, bbox_entries
    
    @staticmethod
    def _extract_clause_from_markdown(markdown_text: str, keywords: List[str]) -> Optional[str]:
        """
        Extract clause from markdown text.
        Supports multiple formats:
        - Markdown headers: ## 9. PAYMENT or # 23. ISPS CODE
        - Numbered lines: 7. PRICE or 9. PAYMENT
        - Bold headers: **7. PRICE**
        Returns full paragraph from start to end.
        """
        kw_upper = [kw.upper() for kw in keywords]
        lines = markdown_text.split("\n")
        
        start_idx = None
        end_idx = None
        
        # Find header with keyword - try multiple patterns
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_upper = line_stripped.upper()
            
            # Pattern 1: Markdown headers: ## 9. PAYMENT or # 23. ISPS CODE
            if line.startswith("##") or line.startswith("#"):
                for kw in kw_upper:
                    if kw in line_upper:
                        # Check if it's a numbered header (e.g., "## 9. PAYMENT")
                        if re.match(r"^#+\s*\d+\.\s*.*" + re.escape(kw), line_upper):
                            start_idx = i
                            logger.debug(
                                "OCRFieldExtractionService: found clause header (markdown) at line %d: '%s'",
                                i, line[:100]
                            )
                            break
                if start_idx is not None:
                    break
            
            # Pattern 2: Numbered line without markdown: 7. PRICE or 9. PAYMENT
            # Match: "7. PRICE" or "9. PAYMENT" (number followed by keyword)
            numbered_match = re.match(r"^(\d+)\.\s+(.+)$", line_stripped)
            if numbered_match:
                number = numbered_match.group(1)
                rest_text = numbered_match.group(2).upper()
                for kw in kw_upper:
                    if kw in rest_text:
                        start_idx = i
                        logger.debug(
                            "OCRFieldExtractionService: found clause header (numbered) at line %d: '%s'",
                            i, line[:100]
                        )
                        break
                if start_idx is not None:
                    break
            
            # Pattern 3: Bold header: **7. PRICE** or **9. PAYMENT**
            if line_stripped.startswith("**") and line_stripped.endswith("**"):
                bold_text = line_stripped[2:-2].strip().upper()
                numbered_match = re.match(r"^(\d+)\.\s+(.+)$", bold_text)
                if numbered_match:
                    rest_text = numbered_match.group(2)
                    for kw in kw_upper:
                        if kw in rest_text:
                            start_idx = i
                            logger.debug(
                                "OCRFieldExtractionService: found clause header (bold) at line %d: '%s'",
                                i, line[:100]
                            )
                            break
                    if start_idx is not None:
                        break
        
        if start_idx is None:
            logger.debug("OCRFieldExtractionService: no clause header found with keywords %s", keywords)
            # Log first 20 lines for debugging
            logger.debug("OCRFieldExtractionService: first 20 lines of markdown:")
            for i, line in enumerate(lines[:20]):
                logger.debug("  Line %d: '%s'", i, line[:100])
            return None
        
        # Find next header (stop condition) - support same patterns
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            
            # Stop at next numbered header (any format)
            # Pattern 1: Markdown header with number
            if line.startswith("##") or line.startswith("#"):
                if re.match(r"^#+\s*\d+\.\s+", line):
                    end_idx = i
                    logger.debug(
                        "OCRFieldExtractionService: found next header (markdown) at line %d: '%s', stopping",
                        i, line[:100]
                    )
                    break
            
            # Pattern 2: Numbered line (next clause)
            numbered_match = re.match(r"^(\d+)\.\s+", line)
            if numbered_match:
                # Check if it's a different number (not continuation of current clause)
                current_line = lines[start_idx].strip()
                current_match = re.match(r"^(\d+)\.\s+", current_line)
                if current_match:
                    current_num = current_match.group(1)
                    next_num = numbered_match.group(1)
                    if next_num != current_num:
                        end_idx = i
                        logger.debug(
                            "OCRFieldExtractionService: found next header (numbered) at line %d: '%s', stopping",
                            i, line[:100]
                        )
                        break
        
        if end_idx is None:
            end_idx = len(lines)
        
        # Extract clause text - include header and all content until next header
        clause_lines = lines[start_idx:end_idx]
        clause_text = "\n".join(clause_lines).strip()
        
        if not clause_text:
            logger.warning("OCRFieldExtractionService: extracted clause text is empty")
            return None
        
        logger.info(
            "OCRFieldExtractionService: extracted clause from markdown (lines %d-%d, length=%d)",
            start_idx, end_idx, len(clause_text)
        )
        
        return clause_text
    
    @staticmethod
    def _extract_text_value_from_markdown(markdown_text: str, keywords: List[str], pattern: Optional[str] = None) -> Optional[str]:
        """
        Extract text value (key-value pair) from markdown.
        Supports multiple patterns:
        - **Key:** Value
        - Key: Value
        - Key: Value (multiline)
        - Key - Value
        """
        kw_upper = [kw.upper() for kw in keywords]
        
        if pattern:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            m = regex.search(markdown_text)
            if m:
                value = m.group("value").strip() if "value" in m.groupdict() else m.group(1).strip()
                return value
        
        # Search for key-value patterns
        lines = markdown_text.split("\n")
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_upper = line_stripped.upper()
            
            # Check if line contains any keyword
            for kw in kw_upper:
                if kw in line_upper:
                    # Pattern 1: **Key:** Value or Key: Value (same line)
                    match = re.search(r"[:\-]\s*(.+)$", line_stripped)
                    if match:
                        value = match.group(1).strip()
                        # Remove markdown formatting
                        value = re.sub(r'\*\*', '', value)  # Remove bold
                        value = re.sub(r'\*', '', value)    # Remove italic
                        value = re.sub(r'`', '', value)     # Remove code
                        if value and len(value) > 0:
                            logger.debug(
                                "OCRFieldExtractionService: extracted text value '%s' for keyword '%s'",
                                value[:100], kw
                            )
                            return value
                    
                    # Pattern 2: Key on one line, value on next line(s)
                    # Check if current line is just the key (ends with : or -)
                    if re.match(r".*[:\-]\s*$", line_stripped) or line_stripped.upper() == kw:
                        # Look at next few lines for value
                        value_parts = []
                        for j in range(i + 1, min(i + 5, len(lines))):
                            next_line = lines[j].strip()
                            if not next_line:
                                continue
                            # Stop if we hit a new section (numbered header, markdown header, or another key-value)
                            if re.match(r"^\d+\.\s+", next_line) or next_line.startswith("#"):
                                break
                            if re.search(r"[:\-]\s+", next_line) and any(k in next_line.upper() for k in kw_upper if k != kw):
                                break
                            # Collect value lines
                            clean_line = re.sub(r'\*\*|\*|`', '', next_line)
                            if clean_line:
                                value_parts.append(clean_line)
                        
                        if value_parts:
                            value = " ".join(value_parts).strip()
                            if value:
                                logger.debug(
                                    "OCRFieldExtractionService: extracted text value (multiline) '%s' for keyword '%s'",
                                    value[:100], kw
                                )
                                return value
                    
                    # Pattern 3: Key: Value format where value might be on same line after colon
                    # More flexible pattern - look for keyword followed by colon/dash and text
                    colon_match = re.search(rf"{re.escape(kw)}\s*[:\-]\s*(.+)$", line_upper)
                    if colon_match:
                        value = colon_match.group(1).strip()
                        # Get original case from line
                        original_match = re.search(rf"{re.escape(kw)}\s*[:\-]\s*(.+)$", line_stripped, re.IGNORECASE)
                        if original_match:
                            value = original_match.group(1).strip()
                            value = re.sub(r'\*\*|\*|`', '', value)
                            if value:
                                logger.debug(
                                    "OCRFieldExtractionService: extracted text value (colon pattern) '%s' for keyword '%s'",
                                    value[:100], kw
                                )
                                return value
        
        logger.debug("OCRFieldExtractionService: no text value found for keywords %s", keywords)
        return None
    
    @staticmethod
    def _extract_table_from_markdown(markdown_text: str, keywords: List[str]) -> Optional[str]:
        """
        Extract table from markdown by keywords.
        Searches for heading with keywords and extracts table after it.
        Supports multiple header formats.
        """
        kw_upper = [kw.upper() for kw in keywords]
        lines = markdown_text.split("\n")
        
        # Find header with keyword - try multiple patterns
        start_idx = None
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_upper = line_stripped.upper()
            
            # Pattern 1: Markdown headers
            if line.startswith("##") or line.startswith("#"):
                for kw in kw_upper:
                    if kw in line_upper:
                        start_idx = i
                        logger.debug(
                            "OCRFieldExtractionService: found table header (markdown) at line %d: '%s'",
                            i, line[:100]
                        )
                        break
                if start_idx is not None:
                    break
            
            # Pattern 2: Numbered line without markdown
            numbered_match = re.match(r"^(\d+)\.\s+(.+)$", line_stripped)
            if numbered_match:
                rest_text = numbered_match.group(2).upper()
                for kw in kw_upper:
                    if kw in rest_text:
                        start_idx = i
                        logger.debug(
                            "OCRFieldExtractionService: found table header (numbered) at line %d: '%s'",
                            i, line[:100]
                        )
                        break
                if start_idx is not None:
                    break
            
            # Pattern 3: Bold header
            if line_stripped.startswith("**") and line_stripped.endswith("**"):
                bold_text = line_stripped[2:-2].strip().upper()
                for kw in kw_upper:
                    if kw in bold_text:
                        start_idx = i
                        logger.debug(
                            "OCRFieldExtractionService: found table header (bold) at line %d: '%s'",
                            i, line[:100]
                        )
                        break
                if start_idx is not None:
                    break
        
        if start_idx is None:
            logger.debug("OCRFieldExtractionService: no table header found with keywords %s", keywords)
            return None
        
        # Find table start (line with |) - search within reasonable distance (next 50 lines)
        table_start_idx = None
        search_end = min(start_idx + 50, len(lines))
        for i in range(start_idx + 1, search_end):
            line = lines[i].strip()
            # Look for markdown table row
            if line.startswith("|") and "|" in line[1:]:
                table_start_idx = i
                logger.debug(
                    "OCRFieldExtractionService: found table start at line %d: '%s'",
                    i, line[:100]
                )
                break
        
        if table_start_idx is None:
            logger.debug("OCRFieldExtractionService: no table found after header (searched %d lines)", search_end - start_idx - 1)
            return None
        
        # Extract table until next non-table line or next header
        table_lines = []
        for i in range(table_start_idx, len(lines)):
            line = lines[i].strip()
            
            # Stop if we hit next numbered header (any format)
            if re.match(r"^\d+\.\s+", line) or (line.startswith("#") and re.match(r"^#+\s*\d+\.\s+", line)):
                # Check if it's a different number
                current_line = lines[start_idx].strip()
                current_match = re.match(r"^(\d+)\.\s+", current_line)
                if current_match:
                    current_num = current_match.group(1)
                    next_match = re.match(r"^(\d+)\.\s+", line)
                    if next_match and next_match.group(1) != current_num:
                        break
            
            # Stop if line doesn't look like table row
            if not line.startswith("|") and table_lines:
                # Check if it's separator line (---|:---|:---)
                if re.match(r"^[\|\-\s:]+$", line):
                    table_lines.append(lines[i])
                    continue
                # Otherwise stop
                break
            
            if line.startswith("|") or re.match(r"^[\|\-\s:]+$", line):
                table_lines.append(lines[i])
        
        if not table_lines:
            logger.debug("OCRFieldExtractionService: no table content found")
            return None
        
        table_text = "\n".join(table_lines).strip()
        
        logger.info(
            "OCRFieldExtractionService: extracted table from markdown (lines %d-%d, length=%d)",
            table_start_idx, table_start_idx + len(table_lines), len(table_text)
        )
        
        return table_text

    def extract_text_value_from_ocr(
        self,
        ocr_elements: List[OCRTextElement],
        full_text: str,
        keywords: List[str],
        pattern: Optional[str] = None
    ) -> Optional[Tuple[str, Optional[str], Optional[List[BoundingBoxEntry]]]]:
        """
        Extract text value from OCR elements by keywords.
        Similar to extract_text_field but with coordinates.
        
        Returns:
            Tuple of (extracted_value, page_num_str, bbox_dict) or None
            page_num_str: "2" for single page
            bbox_dict: {"2": [x1, y1, x2, y2]} or {"combined": [x1, y1, x2, y2]}
        """
        if pattern:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            m = regex.search(full_text)
            if m:
                value = m.group("value").strip() if "value" in m.groupdict() else m.group(1).strip()
                # Try to find bbox for this value
                value_start = m.start()
                value_end = m.end()
                for elem in ocr_elements:
                    if elem.start_index <= value_start <= elem.end_index:
                        bbox_dict = {str(elem.page_number): elem.bbox} if elem.bbox else None
                        return value, str(elem.page_number), self._bbox_dict_to_entries(bbox_dict)
                return value, None, None

        kw_upper = [kw.upper() for kw in keywords]

        # Search in elements
        for i, elem in enumerate(ocr_elements):
            elem_upper = elem.text.upper()
            if not any(kw in elem_upper for kw in kw_upper):
                continue

            # Try to extract value from same element
            line = elem.text.strip()
            # Pattern: "KEYWORD: value" or "KEYWORD - value"
            m = re.search(r"[:\-]\s*(.+)$", line)
            if m:
                value = m.group(1).strip()
                if value and elem.bbox:
                    bbox_dict = {str(elem.page_number): elem.bbox}
                    return value, str(elem.page_number), self._bbox_dict_to_entries(bbox_dict)

            # Try next element
            if i + 1 < len(ocr_elements):
                next_elem = ocr_elements[i + 1]
                next_text = next_elem.text.strip()
                if next_text and not self._is_likely_header(next_text):
                    # Combine bboxes if value spans multiple elements
                    bboxes_to_combine = [e.bbox for e in [elem, next_elem] if e.bbox]
                    combined_bbox = self._combine_bboxes(bboxes_to_combine)
                    if combined_bbox:
                        # If on same page, use that page, otherwise use combined
                        if elem.page_number == next_elem.page_number:
                            bbox_dict = {str(elem.page_number): combined_bbox}
                        else:
                            bbox_dict = {
                                str(elem.page_number): elem.bbox,
                                str(next_elem.page_number): next_elem.bbox,
                                "combined": combined_bbox
                            }
                        page_num_str = str(next_elem.page_number) if elem.page_number == next_elem.page_number else f"{elem.page_number},{next_elem.page_number}"
                        return next_text, page_num_str, self._bbox_dict_to_entries(bbox_dict)

        return None

    @staticmethod
    def _is_likely_header(text: str) -> bool:
        """Check if text looks like a header"""
        text = text.strip()
        if not text:
            return False
        
        # Numbered header: "7. PRICE", "23. ISPS CODE"
        if re.match(r"^\s*\d+\.\s+[A-Z]", text):
            return True
        
        # Uppercase only (likely header)
        if text.isupper() and len(text) > 2 and not re.search(r"[a-z]", text):
            return True
        
        # Markdown header
        if re.match(r"^\s*#{1,6}\s+", text):
            return True
        
        return False

    @staticmethod
    def _is_clause_header(text: str) -> bool:
        """
        Heuristic to detect clause headers like "9. PAYMENT".
        Requires a number followed by an uppercase word (to avoid breaking on lists like "1.")
        """
        if not text:
            return False
        normalized = text.strip()
        match = re.match(r"^\d+\.\s+[A-Z][A-Z0-9\s\-&,()/\.]*$", normalized)
        return bool(match)

    @staticmethod
    def _combine_bboxes(bboxes: List[Optional[List[float]]]) -> Optional[List[float]]:
        """Combine multiple bboxes into one"""
        valid_bboxes = [b for b in bboxes if b and len(b) >= 4]
        if not valid_bboxes:
            logger.debug("OCRFieldExtractionService: no valid bboxes to combine")
            return None
        
        if len(valid_bboxes) == 1:
            return valid_bboxes[0]
        
        # Get min/max across all bboxes
        # For multi-page content, we combine all bboxes into one
        min_x = min(b[0] for b in valid_bboxes)
        min_y = min(b[1] for b in valid_bboxes)
        max_x = max(b[2] for b in valid_bboxes)
        max_y = max(b[3] for b in valid_bboxes)
        
        # Ensure normalized coordinates
        result = [
            max(0.0, min(1.0, min_x)),
            max(0.0, min(1.0, min_y)),
            max(0.0, min(1.0, max_x)),
            max(0.0, min(1.0, max_y))
        ]
        
        logger.debug("OCRFieldExtractionService: combined %d bboxes into %s", len(valid_bboxes), result)
        return result

    def extract_by_schema(
        self,
        markdown_text: str,
        extraction_fields: List[ExtractionField],
        gemini_ocr_service: Optional[Any] = None,
        file_path: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> List[ExtractedFieldValue]:
        """
        Extract field values from markdown text.
        Supports clauses, text (key-value pairs), and tables extraction.
        
        Args:
            markdown_text: Full markdown text from Gemini OCR
            extraction_fields: List of fields to extract
            
        Returns:
            List of ExtractedFieldValue
        """
        if not markdown_text:
            logger.warning("OCRFieldExtractionService: empty markdown text")
            return [ExtractedFieldValue(
                field_id=field.id,
                value="",
                confidence=0.0,
                page_num=None,
                bbox=None,
            ) for field in extraction_fields]
        
        results: List[ExtractedFieldValue] = []

        for field in extraction_fields:
            # Load options
            options = self._load_manual_options(field)
            keywords = options.get("keywords") or self._derive_default_keywords(field)
            
            if not keywords:
                logger.warning(
                    "OCRFieldExtractionService: no keywords defined for field_id=%d, name=%s",
                    field.id,
                    field.name,
                )
                results.append(ExtractedFieldValue(
                    field_id=field.id,
                    value="",
                    confidence=0.0,
                    page_num=None,
                    bbox=None,
                ))
                continue

            value: Optional[str] = None
            page_num: Optional[str] = None
            bbox: Optional[List[BoundingBoxEntry]] = None

            if field.type == ExtractionFieldType.CLAUSE:
                value = self._extract_clause_from_markdown(markdown_text, keywords)
            elif field.type == ExtractionFieldType.TABLE:
                value = self._extract_table_from_markdown(markdown_text, keywords)
                if not value:
                    value = "NOT FOUND"
            elif field.type == ExtractionFieldType.TEXT:
                pattern = options.get("pattern")
                value = self._extract_text_value_from_markdown(markdown_text, keywords, pattern)
            
            # Note: Coordinates will be fetched separately in ExtractDocumentFieldValuesService
            # This method is synchronous and cannot await async operations
            else:
                logger.warning(
                    "OCRFieldExtractionService: unsupported field type %s for field_id=%d",
                    field.type.value,
                    field.id,
                )

            results.append(ExtractedFieldValue(
                field_id=field.id,
                value=value or "",
                confidence=1.0 if value else 0.0,
                page_num=page_num,
                bbox=bbox,
            ))

        extracted_count = len([r for r in results if r.value])
        
        logger.info(
            "OCRFieldExtractionService: extracted %d field values from %d fields",
            extracted_count, len(extraction_fields)
        )
        
        # Log details for debugging
        for i, result in enumerate(results):
            if result.value:
                logger.debug(
                    "OCRFieldExtractionService: field_id=%d, value_length=%d",
                    result.field_id, len(result.value)
                )

        return results

    @staticmethod
    def _derive_default_keywords(field: ExtractionField) -> List[str]:
        """Derive default keywords from field name"""
        if field.keywords:
            return field.keywords

        base = re.sub(r"CLAUSE$", "", field.name or "", flags=re.IGNORECASE)
        base = base.upper().strip()
        return [base] if base else []

    @staticmethod
    def _load_manual_options(field: ExtractionField) -> Dict[str, Any]:
        """Load manual extraction options from field prompt or short_description"""
        import json
        
        for source in (field.prompt, field.short_description):
            if not source:
                continue
            stripped = source.strip()
            if not stripped.startswith("{"):
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            if isinstance(data, dict) and data.get("manual_extraction"):
                return data["manual_extraction"]
            if isinstance(data, dict):
                return data
        return {}

    def extract_table_from_layout(
        self,
        layout_response: Dict[str, Any],
        keywords: List[str],
        field: Optional[ExtractionField] = None,
    ) -> Optional[Tuple[str, Optional[str], Optional[List[BoundingBoxEntry]]]]:
        """
        Extract table from Layout Parser response by keywords.
        Works like extract_clause_from_ocr - searches for heading with keywords and extracts table after it.
        
        Strategy:
        1. Search in chunks (markdown) - find heading with keywords, extract table after it
        2. Fallback to blocks (table_block) - search by caption/header/body
        
        Args:
            layout_response: Full Layout Parser response
            keywords: Keywords to search for
            field: Optional ExtractionField for additional context
            
        Returns:
            Tuple of (table_text_markdown, page_num_str, bbox_entries) or None
        """
        document = layout_response.get("document", layout_response)
        kw_upper = [kw.upper() for kw in keywords]
        
        # Expand keywords with field context
        search_keywords = set(kw_upper)
        if field:
            if field.name:
                search_keywords.add(field.name.upper())
            if field.identifier:
                search_keywords.add(field.identifier.upper())
            if field.short_description:
                desc_words = [w.upper() for w in field.short_description.split() if len(w) > 3]
                search_keywords.update(desc_words[:5])
        
        logger.info(
            "OCRFieldExtractionService: searching for table with keywords: %s (field: %s)",
            keywords, field.name if field else "N/A"
        )
        
        # Strategy 1: Search in chunks (markdown) - same logic as extract_clause_from_ocr
        chunked_document = document.get("chunked_document", {})
        chunks = chunked_document.get("chunks", [])
        
        if chunks:
            logger.debug("OCRFieldExtractionService: checking %d chunks for markdown tables", len(chunks))
            
            for chunk_idx, chunk in enumerate(chunks):
                content = chunk.get("content", "").strip()
                if not content:
                    continue
                
                # Search for markdown headings (## or ###) - same as clauses
                for match in re.finditer(r"^(#+)\s*(.*)", content, re.MULTILINE):
                    heading_text = match.group(2).strip()
                    heading_upper = heading_text.upper()
                    
                    # Check if heading contains any keyword
                    if any(kw in heading_upper for kw in search_keywords):
                        logger.debug(
                            "OCRFieldExtractionService: found matching heading in chunk %d: '%s'",
                            chunk_idx, heading_text[:80]
                        )
                        
                        # Found heading, now search for table after it
                        heading_end = match.end()
                        remaining_content = content[heading_end:]
                        
                        # Search for markdown table (lines starting with |)
                        table_match = re.search(r"^\|.*\|", remaining_content, re.MULTILINE)
                        if table_match:
                            # Found table start, extract until next heading
                            table_start_pos = heading_end + table_match.start()
                            remaining_after_table = content[table_start_pos:]
                            
                            # Find next heading (same as clauses)
                            next_heading_match = re.search(r"^(#+)\s+", remaining_after_table, re.MULTILINE)
                            if next_heading_match:
                                # Table ends before next heading
                                table_end_pos = table_start_pos + next_heading_match.start()
                                table_content = content[table_start_pos:table_end_pos].strip()
                            else:
                                # No next heading, take until end of chunk
                                table_content = remaining_after_table.strip()
                            
                            # Clean table from ``` separators and trailing empty lines
                            table_lines = []
                            for line in table_content.split('\n'):
                                line_stripped = line.strip()
                                # Skip separators like ```html
                                if line_stripped.startswith('```'):
                                    continue
                                table_lines.append(line)
                            
                            # Remove trailing empty lines
                            while table_lines and not table_lines[-1].strip():
                                table_lines.pop()
                            
                            table_text = '\n'.join(table_lines).strip()
                            
                            if not table_text:
                                logger.debug("OCRFieldExtractionService: no table content found after heading")
                                continue
                            
                            # Get page information
                            page_span = chunk.get("page_span", {})
                            page_start = page_span.get("page_start", 1) if page_span else 1
                            page_end = page_span.get("page_end", page_start) if page_span else page_start
                            
                            # Format page number string
                            if page_start == page_end:
                                page_num_str = str(page_start)
                            elif page_end == page_start + 1:
                                page_num_str = f"{page_start},{page_end}"
                            else:
                                page_num_str = f"{page_start}-{page_end}"
                            
                            # Try to find bbox from matching blocks
                            bbox_dict, _ = self._find_bbox_from_blocks_by_text(
                                layout_response, table_text, keywords
                            )
                            bbox_entries = self._bbox_dict_to_entries(bbox_dict) if bbox_dict else None
                            
                            logger.info(
                                "OCRFieldExtractionService: extracted table from chunk %d, pages %s, length %d, bbox=%s",
                                chunk_idx, page_num_str, len(table_text), "yes" if bbox_entries else "no"
                            )
                            
                            return table_text, page_num_str, bbox_entries
        
        # Strategy 2: Fallback to blocks (table_block) - search by caption/header/body
        logger.debug("OCRFieldExtractionService: falling back to blocks search")
        document_layout = document.get("document_layout", {})
        blocks = document_layout.get("blocks", [])
        
        table_candidates: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
        
        for block in blocks:
            table_block = block.get("table_block", {})
            if not table_block:
                continue
            
            match_score = 0.0
            match_reasons = []
            
            # Check table caption
            caption = table_block.get("caption", "").strip()
            if caption:
                caption_upper = caption.upper()
                for kw in search_keywords:
                    if kw in caption_upper:
                        match_score += 50.0
                        match_reasons.append(f"caption: '{caption[:50]}'")
                        break
            
            # Check header rows
            header_rows = table_block.get("header_rows", [])
            for header_row in header_rows:
                cells = header_row.get("cells", [])
                for cell in cells:
                    cell_text = self._extract_cell_text(cell).strip()
                    if cell_text:
                        for kw in search_keywords:
                            if kw in cell_text.upper():
                                match_score += 30.0
                                match_reasons.append(f"header: '{cell_text[:50]}'")
                                break
                    if match_score > 0:
                        break
                if match_score > 0:
                    break
            
            # Check body rows (first 10)
            if match_score == 0:
                body_rows = table_block.get("body_rows", [])[:10]
                for body_row in body_rows:
                    cells = body_row.get("cells", [])
                    for cell in cells:
                        cell_text = self._extract_cell_text(cell).upper()
                        for kw in search_keywords:
                            if kw in cell_text:
                                match_score += 5.0
                                break
                        if match_score > 0:
                            break
                    if match_score > 0:
                        break
            
            if match_score > 0:
                table_candidates.append((block, table_block, match_score))
                logger.debug(
                    "OCRFieldExtractionService: table candidate found (score=%.1f, reasons=%s)",
                    match_score, match_reasons
                )
        
        if not table_candidates:
            logger.warning(
                "OCRFieldExtractionService: no table found with keywords %s",
                keywords
            )
            return None
        
        # Sort by match score (highest first)
        table_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Use the best matching table
        best_block, best_table_block, best_score = table_candidates[0]
        
        logger.info(
            "OCRFieldExtractionService: selected table from blocks with score %.1f (from %d candidates)",
            best_score, len(table_candidates)
        )
        
        # Extract table text
        table_text = self._extract_table_text(best_table_block)
        
        # Get page information
        page_span = best_block.get("page_span", {})
        page_start = page_span.get("page_start", 1) if page_span else 1
        page_end = page_span.get("page_end", page_start) if page_span else page_start
        
        # Format page number string
        if page_start == page_end:
            page_num_str = str(page_start)
        elif page_end == page_start + 1:
            page_num_str = f"{page_start},{page_end}"
        else:
            page_num_str = f"{page_start}-{page_end}"
        
        # Try to extract bounding box from table cells
        bbox_dict: Dict[str, List[float]] = {}
        all_cell_bboxes = []
        
        # Collect bboxes from all cells
        header_rows = best_table_block.get("header_rows", [])
        for header_row in header_rows:
            for cell in header_row.get("cells", []):
                bbox = self._extract_bbox_from_table_cell(cell)
                if bbox:
                    all_cell_bboxes.append(bbox)
        
        body_rows = best_table_block.get("body_rows", [])
        for body_row in body_rows:
            for cell in body_row.get("cells", []):
                bbox = self._extract_bbox_from_table_cell(cell)
                if bbox:
                    all_cell_bboxes.append(bbox)
        
        if all_cell_bboxes:
            combined_bbox = self._combine_bboxes(all_cell_bboxes)
            if combined_bbox:
                bbox_dict[str(page_start)] = combined_bbox
                if page_start != page_end:
                    bbox_dict["combined"] = combined_bbox
        
        bbox_entries = self._bbox_dict_to_entries(bbox_dict) if bbox_dict else None
        
        logger.info(
            "OCRFieldExtractionService: extracted table from blocks, pages %s, text_length=%d, bbox=%s",
            page_num_str, len(table_text), "yes" if bbox_entries else "no"
        )
        
        return table_text, page_num_str, bbox_entries
    
    @staticmethod
    def _extract_text_from_block(block: Dict[str, Any]) -> str:
        """Extract text from a block (text_block only)"""
        text_block = block.get("text_block", {})
        if not text_block:
            return ""
        
        text = text_block.get("text", "").strip()
        
        # Also check nested blocks
        nested_blocks = text_block.get("blocks", [])
        nested_texts = []
        for nested in nested_blocks:
            nested_text = OCRFieldExtractionService._extract_text_from_block(nested)
            if nested_text:
                nested_texts.append(nested_text)
        
        if nested_texts:
            return f"{text} {' '.join(nested_texts)}".strip()
        
        return text
    
    @staticmethod
    def _extract_bbox_from_table_cell(cell: Dict[str, Any]) -> Optional[List[float]]:
        """Extract bounding box from table cell by searching in nested blocks"""
        blocks = cell.get("blocks", [])
        bboxes = []
        
        def search_bbox_in_block(block: Dict[str, Any]) -> Optional[List[float]]:
            """Recursively search for bounding_poly in block"""
            # Check if block has direct bounding_poly (unlikely in layout response)
            bounding_poly = block.get("bounding_poly") or block.get("boundingPoly")
            if bounding_poly:
                return OCRFieldExtractionService._extract_bounding_box(bounding_poly)
            
            # Check text_block
            text_block = block.get("text_block", {})
            if text_block:
                # Check nested blocks
                nested_blocks = text_block.get("blocks", [])
                for nested in nested_blocks:
                    bbox = search_bbox_in_block(nested)
                    if bbox:
                        return bbox
            
            return None
        
        for block in blocks:
            bbox = search_bbox_in_block(block)
            if bbox:
                bboxes.append(bbox)
        
        if bboxes:
            return OCRFieldExtractionService._combine_bboxes(bboxes)
        return None
    
    @staticmethod
    def _bbox_dict_to_entries(
        bbox_dict: Optional[Dict[str, Optional[List[float]]]]
    ) -> Optional[List[BoundingBoxEntry]]:
        if not bbox_dict:
            return None
        entries: List[BoundingBoxEntry] = []
        for page, coords in bbox_dict.items():
            if not coords or len(coords) < 4:
                continue
            entries.append(BoundingBoxEntry(page=str(page), coords=coords[:4]))
        return entries or None
    
    def _find_bbox_from_blocks_by_text(
        self,
        layout_response: Dict[str, Any],
        search_text: str,
        keywords: List[str]
    ) -> Tuple[Optional[Dict[str, List[float]]], Optional[str]]:
        """
        Находит bounding box для текста, сопоставляя его с блоками из document_layout.blocks.
        Ищет блоки, которые содержат части текста, и извлекает координаты.
        
        Returns:
            Tuple of (bbox_dict, page_num_str) or (None, page_num_str)
        """
        document = layout_response.get("document", layout_response)
        document_layout = document.get("document_layout", {})
        blocks = document_layout.get("blocks", [])
        
        if not blocks:
            return None, None
        
        # Нормализуем поисковый текст
        search_text_clean = re.sub(r'\s+', ' ', search_text.strip().upper())
        search_words = set(re.findall(r'\b\w+\b', search_text_clean))
        
        # Ищем блоки, которые содержат keywords или части текста
        matching_blocks: List[Tuple[Dict[str, Any], int, float]] = []  # (block, page, score)
        
        def search_in_block_recursive(block: Dict[str, Any], page_num: int, depth: int = 0) -> None:
            """Рекурсивно ищем блоки с текстом"""
            if depth > 5:  # Ограничиваем глубину
                return
            
            # Проверяем text_block
            text_block = block.get("text_block", {})
            if text_block:
                block_text = text_block.get("text", "").strip()
                if block_text:
                    block_text_clean = re.sub(r'\s+', ' ', block_text.upper())
                    block_words = set(re.findall(r'\b\w+\b', block_text_clean))
                    
                    # Вычисляем score - сколько общих слов
                    common_words = search_words & block_words
                    if len(common_words) >= 3:  # Минимум 3 общих слова
                        score = len(common_words) / max(len(search_words), len(block_words))
                        
                        # Пробуем извлечь bbox из layout
                        bbox = self._extract_bbox_from_block_layout(block)
                        if bbox:
                            matching_blocks.append((block, page_num, score))
                    
                    # Рекурсивно ищем во вложенных блоках
                    nested_blocks = text_block.get("blocks", [])
                    for nested in nested_blocks:
                        nested_page_span = nested.get("page_span", {})
                        nested_page = nested_page_span.get("page_start", page_num) if nested_page_span else page_num
                        search_in_block_recursive(nested, nested_page, depth + 1)
            
            # Проверяем table_block
            table_block = block.get("table_block", {})
            if table_block:
                # Для таблиц извлекаем текст и проверяем
                table_text = self._extract_table_text(table_block)
                if table_text:
                    table_text_clean = re.sub(r'\s+', ' ', table_text.upper())
                    table_words = set(re.findall(r'\b\w+\b', table_text_clean))
                    
                    common_words = search_words & table_words
                    if len(common_words) >= 5:  # Для таблиц нужно больше совпадений
                        score = len(common_words) / max(len(search_words), len(table_words))
                        bbox = self._extract_bbox_from_block_layout(block)
                        if bbox:
                            matching_blocks.append((block, page_num, score))
        
        # Ищем во всех блоках
        for block in blocks:
            page_span = block.get("page_span", {})
            page_num = page_span.get("page_start", 1) if page_span else 1
            search_in_block_recursive(block, page_num)
        
        if not matching_blocks:
            logger.debug("OCRFieldExtractionService: no matching blocks found for text extraction")
            return None, None
        
        # Сортируем по score и берем лучшие совпадения
        matching_blocks.sort(key=lambda x: x[2], reverse=True)
        
        # Группируем по страницам и собираем bbox
        bbox_dict: Dict[str, List[float]] = {}
        pages_set = set()
        
        # Берем топ-10 совпадений
        for block, page_num, score in matching_blocks[:10]:
            pages_set.add(page_num)
            bbox = self._extract_bbox_from_block_layout(block)
            if bbox:
                page_key = str(page_num)
                if page_key in bbox_dict:
                    # Объединяем с существующим bbox
                    combined = self._combine_bboxes([bbox_dict[page_key], bbox])
                    if combined:
                        bbox_dict[page_key] = combined
                else:
                    bbox_dict[page_key] = bbox
        
        # Форматируем page_num_str
        if not pages_set:
            return None, None
        
        sorted_pages = sorted(pages_set)
        if len(sorted_pages) == 1:
            page_num_str = str(sorted_pages[0])
        elif len(sorted_pages) == 2:
            page_num_str = f"{sorted_pages[0]},{sorted_pages[1]}"
        else:
            page_num_str = f"{sorted_pages[0]}-{sorted_pages[-1]}"
        
        # Добавляем combined bbox если несколько страниц
        if len(bbox_dict) > 1:
            all_bboxes = list(bbox_dict.values())
            combined = self._combine_bboxes(all_bboxes)
            if combined:
                bbox_dict["combined"] = combined
        
        logger.debug(
            "OCRFieldExtractionService: found bbox for text (pages=%s, blocks=%d)",
            page_num_str, len(matching_blocks)
        )
        
        return bbox_dict if bbox_dict else None, page_num_str
    
    @staticmethod
    def _extract_bbox_from_block_layout(block: Dict[str, Any]) -> Optional[List[float]]:
        """
        Извлекает bounding box из блока.
        Проверяет layout.bounding_poly или другие возможные места.
        """
        # Проверяем layout в блоке
        layout = block.get("layout", {})
        if layout:
            bounding_poly = layout.get("bounding_poly") or layout.get("boundingPoly")
            if bounding_poly:
                return OCRFieldExtractionService._extract_bounding_box(bounding_poly)
        
        # Проверяем напрямую bounding_poly в блоке
        bounding_poly = block.get("bounding_poly") or block.get("boundingPoly")
        if bounding_poly:
            return OCRFieldExtractionService._extract_bounding_box(bounding_poly)
        
        # Для text_block проверяем вложенные блоки
        text_block = block.get("text_block", {})
        if text_block:
            nested_blocks = text_block.get("blocks", [])
            for nested in nested_blocks:
                nested_layout = nested.get("layout", {})
                if nested_layout:
                    nested_bounding_poly = nested_layout.get("bounding_poly") or nested_layout.get("boundingPoly")
                    if nested_bounding_poly:
                        bbox = OCRFieldExtractionService._extract_bounding_box(nested_bounding_poly)
                        if bbox:
                            return bbox
        
        # Для table_block проверяем ячейки
        table_block = block.get("table_block", {})
        if table_block:
            # Собираем bbox из всех ячеек
            all_cell_bboxes = []
            
            for header_row in table_block.get("header_rows", []):
                for cell in header_row.get("cells", []):
                    bbox = OCRFieldExtractionService._extract_bbox_from_table_cell(cell)
                    if bbox:
                        all_cell_bboxes.append(bbox)
            
            for body_row in table_block.get("body_rows", []):
                for cell in body_row.get("cells", []):
                    bbox = OCRFieldExtractionService._extract_bbox_from_table_cell(cell)
                    if bbox:
                        all_cell_bboxes.append(bbox)
            
            if all_cell_bboxes:
                return OCRFieldExtractionService._combine_bboxes(all_cell_bboxes)
        
        return None

