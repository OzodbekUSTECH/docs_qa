# Utility for finding bounding boxes in PDF documents using word anchors.
# Used for Gemini AI structured extraction grounding.
# Matches prototype logic from bbox_finder.py and gemini_test.py

import logging
from typing import List, Dict, Optional, Any, Tuple

from app.utils.text_normalizer import normalize_for_search, fuzzy_contains

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison with Unicode handling."""
    return normalize_for_search(text)


def extract_pdf_text_with_bbox(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text with bounding boxes from a PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of dictionaries, each containing:
        - "text": the extracted line text (str)
        - "page": 1‑based page number (int)
        - "bbox": normalized coordinates [x1, y1, x2, y2] (list of floats, 0‑1 range)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) not installed. Cannot extract bboxes.")
        return []

    ocr_data: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_bbox = None
                        for span in line["spans"]:
                            line_text += span["text"] + " "
                            if line_bbox is None:
                                line_bbox = list(span["bbox"])
                            else:
                                line_bbox[0] = min(line_bbox[0], span["bbox"][0])
                                line_bbox[1] = min(line_bbox[1], span["bbox"][1])
                                line_bbox[2] = max(line_bbox[2], span["bbox"][2])
                                line_bbox[3] = max(line_bbox[3], span["bbox"][3])
                        if line_text.strip() and line_bbox:
                            normalized_bbox = [
                                line_bbox[0] / page_width,
                                line_bbox[1] / page_height,
                                line_bbox[2] / page_width,
                                line_bbox[3] / page_height,
                            ]
                            ocr_data.append({
                                "text": line_text.strip(),
                                "page": page_num + 1,  # 1‑indexed
                                "bbox": normalized_bbox,
                            })
        doc.close()
        return ocr_data
    except Exception as e:
        logger.error(f"Error extracting PDF text with PyMuPDF: {e}", exc_info=True)
        return []


def find_bbox_by_anchors(
    ocr_data: List[Dict[str, Any]],
    start_words: str,
    end_words: str,
    page_num: int,
) -> Optional[List[float]]:
    """
    Finds bbox coordinates by start_words and end_words in OCR data.
    Returns bbox coordinates as list [x1, y1, x2, y2] or None.
    Matches prototype logic from gemini_test.py (no continuation flag).
    """
    if not ocr_data or not start_words or not end_words:
        logger.debug("Missing ocr_data or start/end words for find_bbox_by_anchors")
        return None

    start_norm = normalize_text(start_words)
    end_norm = normalize_text(end_words)

    page_items = [item for item in ocr_data if item.get('page') == page_num]
    if not page_items:
        logger.debug(f"No text found on page {page_num}")
        return None

    # Find start index - matches prototype logic
    start_idx = None
    for i, item in enumerate(page_items):
        txt = normalize_text(item.get('text', ''))
        if start_norm in txt or fuzzy_contains(txt, start_norm, 0.7):
            start_idx = i
            break
    if start_idx is None:
        # Accumulate text
        acc = ""
        for i, item in enumerate(page_items):
            acc += " " + normalize_text(item.get('text', ''))
            if start_norm in acc or fuzzy_contains(acc, start_norm, 0.8):
                start_idx = i - min(3, i)
                break
    if start_idx is None:
        logger.debug(f"Start phrase not found: '{start_words[:60]}'")
        return None

    # Find end index - matches prototype logic
    end_idx = None
    for i in range(start_idx, len(page_items)):
        txt = normalize_text(page_items[i].get('text', ''))
        if end_norm in txt or fuzzy_contains(txt, end_norm, 0.7):
            end_idx = i
            break
    if end_idx is None:
        # Accumulate text
        acc = ""
        for i in range(start_idx, len(page_items)):
            acc += " " + normalize_text(page_items[i].get('text', ''))
            if end_norm in acc or fuzzy_contains(acc, end_norm, 0.8):
                end_idx = i
                break
    if end_idx is None:
        logger.debug(f"End phrase not found: '{end_words[:60]}'")
        # Use reasonable fallback (like prototype)
        end_idx = min(start_idx + 20, len(page_items) - 1)

    # Collect bboxes
    bboxes = []
    for i in range(start_idx, end_idx + 1):
        if 'bbox' in page_items[i]:
            bboxes.append(page_items[i]['bbox'])
    if not bboxes:
        return None

    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def sanitize_anchor_input(text: str) -> str:
    """Convert escaped control characters (e.g. '\\n') into spaces and collapse whitespace.
    For backward compatibility with extract_field_values.py
    """
    if not text:
        return ""
    import re
    _ESCAPE_SEQUENCE_RE = re.compile(r"\\[nrt]")
    text = _ESCAPE_SEQUENCE_RE.sub(" ", text)
    return " ".join(text.replace("\n", " ").replace("\r", " ").replace("\t", " ").split())


def normalize_anchor_text(text: str) -> str:
    """Normalize anchor text for comparison.
    For backward compatibility with extract_field_values.py
    """
    return normalize_text(text)


def find_bbox_by_plain_anchor(
    ocr_data: List[Dict],
    start_anchor: str,
    page_num: int,
    end_anchor: Optional[str] = None,
) -> Optional[List[float]]:
    """
    Find a bounding box using anchor strings.
    Wrapper around find_bbox_by_anchors for backward compatibility.
    """
    if not start_anchor:
        return None
    
    return find_bbox_by_anchors(
        ocr_data=ocr_data,
        start_words=start_anchor,
        end_words=end_anchor or start_anchor,
        page_num=page_num
    )