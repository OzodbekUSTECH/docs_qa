import json
import logging
import re
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path

from app.entities.extracion_fields import ExtractionField
from app.dto.field_extraction import ExtractedFieldValue
from app.utils.enums import ExtractionFieldType

logger = logging.getLogger("app.services.manual_field_extraction")


class ClauseExtractor:
    """
    Универсальный класс для извлечения клауз и параграфов
    из markdown/текстового контракта.
    Поддерживаем заголовки в стилях:
    - '1. SELLER'
    - '7. PRICE'
    - '23. ISPS CODE COMPLIANCE CLAUSES'
    - '### 7. Price'
    - 'SELLER' / 'BUYER' / 'TAXES AND DUTIES' + следующая строка '-----'
    - 'ATTACHMENT 1 – ULSD FRENCH SPEC'
    - 'ATTACHMENT 2 - ULSD TURKISH SPEC'
    """

    def __init__(self, text: str):
        self.text = text
        self.lines: List[str] = text.splitlines()
        # header_idx -> (normalized_header, raw_header)
        self._headers: Dict[int, Tuple[str, str]] = self._find_clause_start_indices()

    # ---------- Хелперы для заголовков ----------
    @staticmethod
    def _is_mostly_upper(s: str, threshold: float = 0.7) -> bool:
        """
        Проверяем, что строка в основном в верхнем регистре
        (для SELLER / TAXES AND DUTIES / CONTACTS).
        """
        letters = [ch for ch in s if ch.isalpha()]
        if not letters:
            return False
        upper = sum(1 for ch in letters if ch.isupper())
        return upper / len(letters) >= threshold

    def _normalize_header_at(self, i: int) -> Optional[Tuple[str, str]]:
        """
        Вернуть (normalized_header, raw_header) если строка на позиции i — заголовок клаузы.
        Иначе None.
        normalized_header — строка в UPPERCASE, по ней матчим keywords.
        raw_header       — исходный текст заголовка.
        """
        line = self.lines[i].rstrip("\n")

        # 1) Markdown-заголовок: '# ...', '## ...', '### ...'
        m = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if m:
            raw = m.group(1).strip()
            return raw.upper(), raw

        # 2) Нумерованный заголовок: '11. PRICE', '3. QUALITY'
        m = re.match(r"^\s*(\d+)\.\s+(.+?)\s*$", line)
        if m:
            rest = m.group(2).strip()
            # если нет строчных букв -> считаем заголовком ('PRICE', 'RISK AND PROPERTY')
            if not re.search(r"[a-z]", rest):
                raw = rest
                return raw.upper(), raw
            return None

        # 3) Attachment plain: 'ATTACHMENT 1 – ULSD FRENCH SPEC'
        m = re.match(
            r"^\s*ATTACHMENT\s+\d+\s*[–-]\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if m:
            raw = line.strip()
            return raw.upper(), raw

        # 4) UPPERCASE + подчёркивание:
        #    текущая строка в UPPERCASE,
        #    следующая строка — '-----'
        if i + 1 < len(self.lines):
            next_line = self.lines[i + 1]
            if re.match(r"^\s*-{3,}\s*$", next_line):
                if self._is_mostly_upper(line) and len(line.strip()) >= 2:
                    raw = line.strip()
                    return raw.upper(), raw

        return None

    def _find_clause_start_indices(self) -> Dict[int, Tuple[str, str]]:
        """
        Вернуть словарь:
            { line_index: (normalized_header, raw_header), ... }
        """
        result: Dict[int, Tuple[str, str]] = {}
        for i in range(len(self.lines)):
            h = self._normalize_header_at(i)
            if h:
                result[i] = h
        return result

    # ---------- Выделение блока клаузы ----------
    def _find_clause_block_by_keywords(self, keywords: List[str]) -> Optional[str]:
        """
        Найти клауза-блок по ключевым словам в normalized заголовке.
        Возвращает сырые строки от заголовка до следующего заголовка (или конца файла).
        """
        if not self._headers:
            return None

        kw_upper = [kw.upper() for kw in keywords]
        target_start_idx: Optional[int] = None
        sorted_indices = sorted(self._headers.keys())

        for idx in sorted_indices:
            normalized_header, _raw = self._headers[idx]
            if any(kw in normalized_header for kw in kw_upper):
                target_start_idx = idx
                break

        if target_start_idx is None:
            return None

        pos = sorted_indices.index(target_start_idx)
        if pos + 1 < len(sorted_indices):
            end_idx = sorted_indices[pos + 1]
        else:
            end_idx = len(self.lines)

        block_lines = self.lines[target_start_idx:end_idx]
        return "\n".join(block_lines).rstrip("\n")

    # ---------- Параграфы внутри блока ----------
    @staticmethod
    def _split_paragraphs(block: str) -> List[str]:
        """
        Разбиваем блок на параграфы:
        - пустая строка = разделитель параграфов
        - bullet ('- xxx', '* xxx', '• xxx') начинает новый параграф
        """
        lines = block.splitlines()
        paragraphs: List[str] = []
        current: List[str] = []

        def flush():
            if current:
                paragraph = " ".join(line.strip() for line in current if line.strip())
                if paragraph:
                    paragraphs.append(paragraph)
                current.clear()

        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                flush()
                continue

            if re.match(r"^\s*[-*•]\s+", stripped):
                flush()
                current.append(stripped)
                flush()
                continue

            current.append(stripped)

        flush()
        return paragraphs

    # ---------- Публичные методы ----------
    def extract_clause_raw(self, keywords: List[str]) -> Optional[str]:
        return self._find_clause_block_by_keywords(keywords)

    def extract_clause_with_paragraphs(
        self, keywords: List[str]
    ) -> Optional[Dict[str, Any]]:
        block = self._find_clause_block_by_keywords(keywords)
        if block is None:
            return None

        paragraphs = self._split_paragraphs(block)
        return {"block": block, "paragraphs": paragraphs}

    # совместимость со старым интерфейсом
    def extract_clause(self, keywords: List[str]) -> Optional[str]:
        return self.extract_clause_raw(keywords)


# ---------- EXTRACT TEXT FIELD ----------
def extract_text_field(
    text: str,
    keywords: List[str],
    pattern: Optional[str] = None,
) -> Optional[str]:
    """
    Извлечь значение типа Text по label'у:
    - Ищем строку, где есть keyword (CONTRACT NUMBER, CONTRACT NR., REGISTRATION NUMBER).
    - Сначала пытаемся вытащить значение на той же строке после ':' или '-'.
    - Если не вышло — берём следующую непустую строку как значение.
    - Если указан pattern — используем его как regex по всему тексту.
    """
    if pattern:
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        m = regex.search(text)
        if m:
            if "value" in m.groupdict():
                return m.group("value").strip()
            if m.groups():
                return m.group(1).strip()

    lines = text.splitlines()
    kw_upper = [kw.upper() for kw in keywords]

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        upper_line = line_stripped.upper()
        if not any(kw in upper_line for kw in kw_upper):
            continue

        m = re.search(r"[:\-]\s*(.+)$", line_stripped)
        if m:
            value = m.group(1).strip()
            if value:
                return value

        j = i + 1
        while j < len(lines):
            next_line = lines[j].strip()
            j += 1
            if not next_line:
                break
            return next_line

    return None


class ManualFieldExtractionService:
    """Сервис для ручного извлечения значений полей из markdown текста документа"""

    def __init__(self):
        pass

    @staticmethod
    def _derive_default_keywords(field: ExtractionField) -> List[str]:
        if field.keywords:
            return field.keywords

        base = re.sub(r"CLAUSE$", "", field.name or "", flags=re.IGNORECASE)
        base = base.upper().strip()
        return [base] if base else []

    @staticmethod
    def _load_manual_options(field: ExtractionField) -> Dict[str, Any]:
        """
        Попытаться загрузить JSON с настройками ручной экстракции.
        Предпочтение: prompt -> short_description.
        """
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

    def extract_by_schema(
        self,
        text: str,
        extraction_fields: List[ExtractionField],
    ) -> List[ExtractedFieldValue]:
        """
        Динамическая функция для извлечения полей по схеме.

        Args:
            text: markdown/текстовый контент документа
            extraction_fields: список полей для извлечения

        Returns:
            Список извлеченных значений полей
        """
        extractor = ClauseExtractor(text)
        results: List[ExtractedFieldValue] = []

        for field in extraction_fields:
            options = self._load_manual_options(field)
            keywords = options.get("keywords") or self._derive_default_keywords(field)
            if not keywords:
                logger.warning(
                    "ManualFieldExtractionService: no keywords defined for field_id=%d, name=%s",
                    field.id,
                    field.name,
                )
                continue

            value: Optional[str] = None

            if field.type == ExtractionFieldType.CLAUSE:
                mode = (options.get("mode") or "block").lower()
                clause_data = None

                if mode in {"paragraphs", "both"}:
                    clause_data = extractor.extract_clause_with_paragraphs(keywords)
                    if clause_data is None:
                        value = None
                    elif mode == "paragraphs":
                        separator = options.get("paragraph_joiner", "\n\n")
                        value = separator.join(clause_data["paragraphs"])
                    else:  # both
                        value = json.dumps(
                            clause_data,
                            ensure_ascii=False,
                        )
                else:
                    value = extractor.extract_clause_raw(keywords)

            elif field.type == ExtractionFieldType.TEXT:
                pattern = options.get("pattern")
                value = extract_text_field(text, keywords, pattern=pattern)

            else:
                logger.warning(
                    "ManualFieldExtractionService: unsupported field type %s for field_id=%d",
                    field.type.value,
                    field.id,
                )
                value = None

            extracted_value = ExtractedFieldValue(
                field_id=field.id,
                value=value or "",
                confidence=1.0 if value else 0.0,
                page_num=None,
                bbox=None,
            )
            results.append(extracted_value)

        logger.info(
            "ManualFieldExtractionService: extracted %d field values from %d manual fields",
            len([r for r in results if r.value]),
            len(extraction_fields),
        )

        return results

