"""
Gemini OCR Service for extracting markdown from documents with coordinates.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json

from google.genai.client import AsyncClient
from google.genai import types
from pydantic import Field, create_model

logger = logging.getLogger(__name__)


class GeminiOCRService:
    """Service for extracting markdown content from documents using Gemini OCR."""
    
    MODEL = "gemini-2.5-flash"
    
    def __init__(self, client: AsyncClient):
        self.client = client
    
    async def extract_markdown_with_coordinates(
        self,
        file_path: str,
        mime_type: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Extract markdown content with coordinates in ONE request (like Document AI Layout Parser).
        Fast and efficient - returns both markdown and coordinates simultaneously.
        
        Args:
            file_path: Path to the document file
            mime_type: Optional MIME type
            
        Returns:
            Tuple of (markdown_text, coordinates_map) where coordinates_map is:
            {
                "text_ranges": [
                    {
                        "text": "9. PAYMENT",
                        "start_index": 1234,
                        "end_index": 1245,
                        "page_numbers": [2, 3],
                        "bboxes": {
                            "2": [x1, y1, x2, y2],
                            "3": [x1, y1, x2, y2]
                        }
                    },
                    ...
                ]
            }
        """
        if not Path(file_path).exists():
            logger.error("GeminiOCRService: file not found: %s", file_path)
            return None, None
        
        # Detect mime type if not provided
        if not mime_type:
            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "application/pdf"
                logger.warning("GeminiOCRService: could not detect mime type, using default: %s", mime_type)
        
        try:
            logger.info("GeminiOCRService: uploading file: %s (mime_type=%s)", file_path, mime_type)
            uploaded_file = await self.client.files.upload(file=str(file_path))
            logger.info("GeminiOCRService: file uploaded: %s", uploaded_file.name)
            
            # Single request: Extract markdown AND coordinates together
            # Create schema for structured response with both markdown and coordinates
            PageBboxSchema = create_model(
                "PageBboxSchema",
                page=(int, Field(description="Page number (1-indexed)")),
                x1=(float, Field(description="Left coordinate (normalized 0.0-1.0)")),
                y1=(float, Field(description="Top coordinate (normalized 0.0-1.0)")),
                x2=(float, Field(description="Right coordinate (normalized 0.0-1.0)")),
                y2=(float, Field(description="Bottom coordinate (normalized 0.0-1.0)")),
            )
            
            TextElementSchema = create_model(
                "TextElementSchema",
                text=(str, Field(description="The text content (header + complete content block for numbered sections)")),
                start_index=(int, Field(description="Start character index in markdown text")),
                end_index=(int, Field(description="End character index in markdown text")),
                page_bboxes=(List[PageBboxSchema], Field(description="Bounding boxes for EACH page where text appears (if spans multiple pages, include ALL pages)")),
            )
            
            DocumentOutputSchema = create_model(
                "DocumentOutputSchema",
                markdown_text=(str, Field(description="Complete markdown text of the document, well-structured and readable")),
                text_ranges=(List[TextElementSchema], Field(description="List of text elements with coordinates for numbered sections (like '7. PRICE', '9. PAYMENT', etc.)")),
            )
            
            prompt = """Extract the complete document content as markdown AND provide coordinates for all numbered section headers and their content blocks.

REQUIREMENTS:
1. Extract the complete document as markdown:
   - Readable and well-structured
   - In correct order (page by page, top to bottom, left to right)
   - Use proper markdown syntax (headers #, lists, tables |, bold **, etc.)
   - Preserve all text exactly as written
   - Maintain document structure and formatting

2. For each numbered section (e.g., "7. PRICE", "9. PAYMENT"):
   - Find the header text (e.g., "9. PAYMENT")
   - Find the COMPLETE content block that follows until the next numbered section
   - Determine ALL pages where this section appears (header + content may span multiple pages)
   - For EACH page, provide normalized bounding box coordinates (0.0-1.0)
   - Provide character indices (start_index, end_index) in the markdown text

CRITICAL:
- If text spans multiple pages (e.g., starts on page 2, continues on page 3), include coordinates for ALL pages
- For each page, provide separate bounding box coordinates in page_bboxes array
- Include the header text in the "text" field along with the complete content block
- Coordinates should cover the entire section from header to end of content block

Return both markdown_text and text_ranges with coordinates in one response."""
            
            logger.info("GeminiOCRService: extracting markdown and coordinates in ONE request...")
            response = await self.client.models.generate_content(
                model=self.MODEL,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=DocumentOutputSchema,
                )
            )
            
            if not response.text:
                logger.warning("GeminiOCRService: no response text")
                return None, None
            
            try:
                result_data = json.loads(response.text)
                markdown_text = result_data.get("markdown_text", "")
                
                if not markdown_text:
                    logger.warning("GeminiOCRService: no markdown_text in response")
                    return None, None
                
                logger.info("GeminiOCRService: markdown extracted successfully (length=%d)", len(markdown_text))
                
                # Process coordinates
                coordinates_map = None
                text_ranges = result_data.get("text_ranges", [])
                
                if text_ranges:
                    # Convert page_bboxes format to bboxes dict format for easier lookup
                    for text_range in text_ranges:
                        page_bboxes = text_range.get("page_bboxes", [])
                        # Convert to dict format: {"2": [x1, y1, x2, y2], "3": [x1, y1, x2, y2]}
                        bboxes_dict = {}
                        page_numbers = []
                        for page_bbox in page_bboxes:
                            page_num = page_bbox.get("page")
                            if page_num:
                                page_numbers.append(page_num)
                                bboxes_dict[str(page_num)] = [
                                    page_bbox.get("x1", 0.0),
                                    page_bbox.get("y1", 0.0),
                                    page_bbox.get("x2", 1.0),
                                    page_bbox.get("y2", 1.0),
                                ]
                        text_range["bboxes"] = bboxes_dict
                        text_range["page_numbers"] = page_numbers
                        # Remove page_bboxes to avoid duplication
                        if "page_bboxes" in text_range:
                            del text_range["page_bboxes"]
                    
                    coordinates_map = {"text_ranges": text_ranges}
                    logger.info(
                        "GeminiOCRService: extracted coordinates for %d text ranges",
                        len(text_ranges)
                    )
                else:
                    logger.warning("GeminiOCRService: no text_ranges in response")
                
                return markdown_text, coordinates_map
                
            except json.JSONDecodeError as e:
                logger.error("GeminiOCRService: failed to parse JSON response: %s", e)
                logger.debug("GeminiOCRService: raw response: %s", response.text[:500])
                return None, None
            except Exception as e:
                logger.error("GeminiOCRService: error processing response: %s", e, exc_info=True)
                return None, None
                
        except Exception as e:
            logger.error("GeminiOCRService: error extracting markdown with coordinates: %s", e, exc_info=True)
            return None, None
    
    async def improve_markdown_from_document_ai(
        self,
        document_ai_text: str,
        document_ai_coordinates: Dict[str, Any],
        temp_file_path: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Improve markdown structure using Gemini, keeping coordinates from Document AI.
        Hybrid approach: Document AI OCR for accurate coordinates, Gemini for better text structure.
        
        Args:
            document_ai_text: Raw text extracted from Document AI Layout Parser
            document_ai_coordinates: Coordinates map from Document AI (with text elements and bboxes)
            temp_file_path: Optional path to save temporary text file for upload
            
        Returns:
            Tuple of (improved_markdown_text, coordinates_map) where coordinates_map preserves Document AI coordinates
        """
        if not document_ai_text:
            logger.error("GeminiOCRService: document_ai_text is empty")
            return None, None
        
        try:
            # Save text to temporary file for upload
            import tempfile
            import os
            
            if temp_file_path:
                temp_path = temp_file_path
            else:
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
                temp_path = temp_file.name
                temp_file.close()
            
            # Write text to file
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(document_ai_text)
            
            logger.info("GeminiOCRService: uploaded Document AI text to temp file (length=%d)", len(document_ai_text))
            
            # Upload text file to Gemini
            uploaded_file = await self.client.files.upload(file=temp_path)
            logger.info("GeminiOCRService: file uploaded to Gemini: %s", uploaded_file.name)
            
            # Clean up temp file
            if not temp_file_path:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
            
            # Prompt Gemini to improve structure and create markdown
            prompt = """Improve the structure and formatting of this document text, converting it to well-structured markdown.

REQUIREMENTS:
- Convert to readable, well-structured markdown
- Use proper markdown syntax (headers #, lists, tables |, bold **, etc.)
- Preserve all text exactly as written (do not add or remove content)
- Maintain document structure and formatting
- Keep the order of content (page by page, top to bottom, left to right)
- For numbered sections (like "7. PRICE", "9. PAYMENT"), use proper markdown headers
- CRITICAL: Detect all tables and format them as proper Markdown tables (using | separator). Do not skip any rows or columns.

Output only the improved markdown text, nothing else."""
            
            logger.info("GeminiOCRService: improving markdown structure with Gemini...")
            response = await self.client.models.generate_content(
                model=self.MODEL,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(temperature=0.0)
            )
            
            if not response.text:
                logger.warning("GeminiOCRService: no improved markdown in response")
                return None, None
            
            improved_markdown = response.text.strip()
            logger.info("GeminiOCRService: markdown improved successfully (length=%d)", len(improved_markdown))
            
            # Map Document AI coordinates to improved markdown
            # We'll use text matching to find corresponding elements
            coordinates_map = self._map_document_ai_coordinates_to_markdown(
                document_ai_coordinates,
                document_ai_text,
                improved_markdown
            )
            
            return improved_markdown, coordinates_map
            
        except Exception as e:
            logger.error("GeminiOCRService: error improving markdown from Document AI: %s", e, exc_info=True)
            return None, None
    
    async def extract_markdown_from_image(self, file_path: str, mime_type: Optional[str] = None) -> Optional[str]:
        """
        Extract markdown from image using Gemini OCR.
        Simple method for converting image directly to markdown.
        
        Args:
            file_path: Path to image file
            mime_type: Optional MIME type (defaults to image/png)
            
        Returns:
            Markdown formatted text or None if error
        """
        if not Path(file_path).exists():
            logger.error("GeminiOCRService: file not found: %s", file_path)
            return None
        
        # Detect mime type if not provided
        if not mime_type:
            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = "image/png"
                logger.warning("GeminiOCRService: could not detect mime type, using default: %s", mime_type)
        
        try:
            logger.info("GeminiOCRService: uploading image for OCR: %s (mime_type=%s)", file_path, mime_type)
            uploaded_file = await self.client.files.upload(file=str(file_path))
            logger.info("GeminiOCRService: image uploaded: %s", uploaded_file.name)
            
            prompt = """Extract the complete text content from this image and convert it to well-structured markdown format.

REQUIREMENTS:
- Perform OCR to extract all text from the image
- Convert to readable, well-structured markdown
- Use proper markdown syntax (headers #, lists, tables |, bold **, etc.)
- Preserve all text exactly as written (do not add or remove content)
- Maintain document structure and formatting
- Keep the order of content (top to bottom, left to right)
- Detect tables and format them as proper Markdown tables (using | separator)
- For numbered sections, use proper markdown headers

Output only the markdown text, nothing else."""
            
            logger.info("GeminiOCRService: performing OCR and converting to markdown...")
            response = await self.client.models.generate_content(
                model=self.MODEL,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(temperature=0.0)
            )
            
            if not response.text:
                logger.warning("GeminiOCRService: no markdown in response")
                return None
            
            markdown = response.text.strip()
            logger.info("GeminiOCRService: OCR and markdown conversion completed (length=%d)", len(markdown))
            return markdown
            
        except Exception as e:
            logger.error("GeminiOCRService: error extracting markdown from image: %s", e, exc_info=True)
            return None
    
    async def convert_text_to_markdown(self, text: str) -> Optional[str]:
        """
        Convert plain text to well-structured markdown using Gemini.
        Simple method for converting OCR text to markdown.
        
        Args:
            text: Plain text to convert
            
        Returns:
            Markdown formatted text or None if error
        """
        if not text or not text.strip():
            logger.warning("GeminiOCRService: text is empty")
            return None
        
        try:
            import tempfile
            import os
            
            # Save text to temporary file for upload
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
            temp_path = temp_file.name
            temp_file.write(text)
            temp_file.close()
            
            try:
                # Upload text file to Gemini
                uploaded_file = await self.client.files.upload(file=temp_path)
                logger.info("GeminiOCRService: text file uploaded to Gemini: %s", uploaded_file.name)
                
                # Prompt Gemini to convert text to markdown
                prompt = """Convert this OCR text to well-structured markdown format.

REQUIREMENTS:
- Convert to readable, well-structured markdown
- Use proper markdown syntax (headers #, lists, tables |, bold **, etc.)
- Preserve all text exactly as written (do not add or remove content)
- Maintain document structure and formatting
- Keep the order of content (top to bottom, left to right)
- Detect tables and format them as proper Markdown tables (using | separator)
- For numbered sections, use proper markdown headers

Output only the markdown text, nothing else."""
                
                logger.info("GeminiOCRService: converting text to markdown with Gemini...")
                response = await self.client.models.generate_content(
                    model=self.MODEL,
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(temperature=0.0)
                )
                
                if not response.text:
                    logger.warning("GeminiOCRService: no markdown in response")
                    return None
                
                markdown = response.text.strip()
                logger.info("GeminiOCRService: text converted to markdown successfully (length=%d)", len(markdown))
                return markdown
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error("GeminiOCRService: error converting text to markdown: %s", e, exc_info=True)
            return None
    
    def _map_document_ai_coordinates_to_markdown(
        self,
        document_ai_coordinates: Dict[str, Any],
        original_text: str,
        improved_markdown: str
    ) -> Optional[Dict[str, Any]]:
        """
        Map Document AI coordinates to improved markdown text.
        Uses text matching to find corresponding elements.
        """
        if not document_ai_coordinates:
            return None
        
        # If coordinates are already in the expected format, return them
        if "text_ranges" in document_ai_coordinates:
            return document_ai_coordinates
        
        # Extract text elements from Document AI structure
        text_elements = document_ai_coordinates.get("text_elements", [])
        if not text_elements:
            return None
        
        # Group elements by numbered sections and map to markdown
        text_ranges = []
        current_section = None
        
        for elem in text_elements:
            text = elem.get("text", "").strip()
            if not text:
                continue
            
            # Check if this is a numbered section header
            import re
            if re.match(r"^\s*\d+\.\s+[A-Z]", text.upper()):
                # Save previous section if exists
                if current_section:
                    text_ranges.append(current_section)
                
                # Start new section
                current_section = {
                    "text": text,
                    "start_index": elem.get("start_index", 0),
                    "end_index": elem.get("end_index", len(text)),
                    "page_numbers": [elem.get("page", 1)],
                    "bboxes": {
                        str(elem.get("page", 1)): elem.get("bbox", [0.0, 0.0, 1.0, 1.0])
                    }
                }
            elif current_section:
                # Add to current section
                current_section["text"] += "\n\n" + text
                current_section["end_index"] = elem.get("end_index", current_section["end_index"])
                
                page_num = elem.get("page", 1)
                if page_num not in current_section["page_numbers"]:
                    current_section["page_numbers"].append(page_num)
                
                page_key = str(page_num)
                if page_key not in current_section["bboxes"]:
                    current_section["bboxes"][page_key] = elem.get("bbox", [0.0, 0.0, 1.0, 1.0])
        
        # Add last section
        if current_section:
            text_ranges.append(current_section)
        
        if text_ranges:
            logger.debug("GeminiOCRService: mapped %d text ranges from Document AI coordinates", len(text_ranges))
            return {"text_ranges": text_ranges}
        
        return None
    
    async def extract_markdown(self, file_path: str, mime_type: Optional[str] = None) -> Optional[str]:
        """
        Extract markdown content from a document file (backward compatibility).
        For coordinates, use extract_markdown_with_coordinates instead.
        """
        markdown_text, _ = await self.extract_markdown_with_coordinates(file_path, mime_type)
        return markdown_text
    
    def find_text_coordinates_in_map(
        self,
        text_to_find: str,
        keywords: List[str],
        coordinates_map: Optional[Dict[str, Any]],
        markdown_text: str
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        """
        Find coordinates for text using pre-extracted coordinates map.
        This avoids making additional API calls.
        
        Args:
            text_to_find: Text to find coordinates for (may include markdown headers like "## 2. BUYER")
            keywords: Keywords to help locate the text
            coordinates_map: Pre-extracted coordinates map from extract_markdown_with_coordinates
            markdown_text: Full markdown text for matching
            
        Returns:
            Tuple of (page_num_str, bbox_entries)
        """
        if not coordinates_map or not text_to_find:
            return None, None
        
        import re
        
        kw_upper = [kw.upper() for kw in keywords]
        text_to_find_upper = text_to_find.upper().strip()
        
        # Extract section number from text_to_find if it's a markdown header (e.g., "## 2. BUYER" -> "2")
        section_num_match = re.search(r"#+\s*(\d+)\.\s+", text_to_find_upper)
        section_num = section_num_match.group(1) if section_num_match else None
        
        # Extract main keyword from text_to_find (e.g., "## 2. BUYER" -> "BUYER")
        keyword_from_text = None
        if section_num:
            keyword_match = re.search(rf"#+\s*{re.escape(section_num)}\.\s+([A-Z][A-Z\s]+)", text_to_find_upper)
            if keyword_match:
                keyword_from_text = keyword_match.group(1).strip()
        
        # Search in coordinates map
        text_ranges = coordinates_map.get("text_ranges", [])
        
        best_match = None
        best_score = 0
        
        for text_range in text_ranges:
            range_text = text_range.get("text", "").upper()
            
            # Score based on multiple criteria
            score = 0
            
            # 1. Check section number match (highest priority)
            if section_num:
                if re.search(rf"#+\s*{re.escape(section_num)}\.\s+", range_text) or re.search(rf"^\s*{re.escape(section_num)}\.\s+", range_text):
                    score += 50
            
            # 2. Check keyword match
            for kw in kw_upper:
                if kw in range_text:
                    score += 15
                    # Bonus if keyword appears at the start (likely header)
                    if range_text.startswith(kw) or f" {kw}" in range_text[:100]:
                        score += 10
            
            # 3. Check keyword from text
            if keyword_from_text and keyword_from_text in range_text:
                score += 20
            
            # 4. Check if text_to_find matches range text (partial match)
            # Remove markdown formatting for comparison
            range_text_clean = re.sub(r"#+\s*", "", range_text)
            text_to_find_clean = re.sub(r"#+\s*", "", text_to_find_upper)
            
            # Check if significant portion of text matches
            if len(text_to_find_clean) > 20:
                # Check first 100 chars match
                if text_to_find_clean[:100] in range_text_clean:
                    score += 25
                # Check if key parts match
                words_to_find = set(text_to_find_clean.split()[:10])
                words_range = set(range_text_clean.split()[:20])
                common_words = words_to_find.intersection(words_range)
                if len(common_words) >= 3:
                    score += 15
            
            # 5. Prefer longer matches (but not too long - might be wrong section)
            if len(range_text) >= len(text_to_find_clean) * 0.5 and len(range_text) <= len(text_to_find_clean) * 2:
                score += 5
            
            if score > best_score:
                best_score = score
                best_match = text_range
        
        # Lower threshold for matching - we want to find coordinates even with partial matches
        if best_match and best_score >= 15:
            page_numbers = best_match.get("page_numbers", [])
            bboxes = best_match.get("bboxes", {})
            
            if page_numbers:
                # Format page number string
                if len(page_numbers) == 1:
                    page_num_str = str(page_numbers[0])
                elif len(page_numbers) == 2:
                    page_num_str = f"{page_numbers[0]},{page_numbers[1]}"
                else:
                    page_num_str = f"{page_numbers[0]}-{page_numbers[-1]}"
                
                # Build bbox entries
                bbox_entries = []
                for page_num in page_numbers:
                    page_key = str(page_num)
                    if page_key in bboxes:
                        coords = bboxes[page_key]
                        if len(coords) >= 4:
                            bbox_entries.append({
                                "page": page_key,
                                "coords": coords[:4]
                            })
                
                # Add combined bbox if multiple pages
                if len(bbox_entries) > 1:
                    all_coords = [entry["coords"] for entry in bbox_entries]
                    min_x = min(c[0] for c in all_coords)
                    min_y = min(c[1] for c in all_coords)
                    max_x = max(c[2] for c in all_coords)
                    max_y = max(c[3] for c in all_coords)
                    bbox_entries.append({
                        "page": "combined",
                        "coords": [min_x, min_y, max_x, max_y]
                    })
                
                logger.debug(
                    "GeminiOCRService: found coordinates in map - pages=%s, bbox_entries=%d, score=%d, keywords=%s",
                    page_num_str, len(bbox_entries), best_score, keywords
                )
                
                return page_num_str, bbox_entries
        
        logger.warning(
            "GeminiOCRService: no coordinates found in map for text. Best score: %d (threshold: 15), keywords: %s, text_preview: %s, available_ranges: %d",
            best_score, keywords, text_to_find[:100] if text_to_find else "", len(text_ranges)
        )
        return None, None

