import logging
import re
from typing import Dict, Optional, List, Any
from pathlib import Path

from app.entities.extracion_fields import ExtractionField
from app.dto.field_extraction import ExtractedFieldValue

logger = logging.getLogger("app.services.manual_field_extraction")


class ClauseExtractor:
    """
    Универсальный класс для извлечения клауз из markdown/текстового контракта.
    Поддерживаем такие типы заголовков:
    
    1) Markdown:
       '# Something'
       '## 24. ISPS Code Compliance Clauses'
       '### Attachment 1 – ULSD French Spec'
    
    2) Нумерованные клаузы:
       '1. SELLER'
       '11. PRICE'
       '23. ISPS CODE COMPLIANCE CLAUSES'
       (отсекаем строки, где после номера есть строчные буквы)
    
    3) Подчёркнутые заголовки:
       'SELLER'
       '-----'
    
    4) Attachment-строки без markdown:
       'ATTACHMENT 1 – ULSD FRENCH SPEC'
       'ATTACHMENT 2 - ULSD TURKISH SPEC'
    """
    
    def __init__(self, text: str):
        self.text = text
        self.lines: List[str] = text.splitlines()
    
    def _normalize_header_at(self, i: int) -> Optional[str]:
        """
        Нормализует заголовок на строке i.
        Возвращает текст заголовка или None, если это не заголовок.
        """
        line = self.lines[i]
        
        # 1) Markdown-заголовок: '# ...', '## ...', '### ...'
        m = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if m:
            header_text = m.group(1).strip()
            return header_text
        
        # 2) Нумерованный заголовок: '11. PRICE', '3. QUALITY'
        m = re.match(r"^\s*(\d+)\.\s+(.+?)\s*$", line)
        if m:
            rest = m.group(2).strip()
            # Если нет строчных букв — считаем заголовком ('PRICE', 'RISK AND PROPERTY')
            if not re.search(r"[a-z]", rest):
                return rest
            return None  # типа '1. From the date ...' — не заголовок клаузы
        
        # 3) Attachment plain: 'ATTACHMENT 1 – ULSD FRENCH SPEC'
        #    Поддерживаем обычный '-' и длинное тире '–'
        m = re.match(
            r"^\s*ATTACHMENT\s+\d+\s*[–-]\s*(.+?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if m:
            # Возвращаем нормализованный текст заголовка,
            # но с самим словом ATTACHMENT, чтобы по нему можно было матчить.
            # Пример: 'ATTACHMENT 1 – ULSD FRENCH SPEC'
            return line.strip()
        
        # 4) Подчёркнутый заголовок:
        #    Текущая строка — ALL CAPS текст,
        #    следующая строка — линия из дефисов.
        if i + 1 < len(self.lines):
            next_line = self.lines[i + 1]
            if re.match(r"^\s*-{3,}\s*$", next_line):
                # SELLER / BUYER / TAXES AND DUTIES / CONTACTS и т.п.
                if re.match(r"^\s*[A-Z0-9 ,&/'\-]+\s*$", line) and re.search(
                    r"[A-Z]", line
                ):
                    return line.strip()
        
        # Остальное — не заголовок
        return None
    
    def _find_clause_start_indices(self) -> Dict[int, str]:
        """
        Вернёт dict: {line_index: "HEADER TEXT"}.
        Например:
        { 30: "PRICE", 45: "PAYMENT", 5: "SELLER", 200: "ATTACHMENT 1 – ULSD FRENCH SPEC", ... }
        """
        result: Dict[int, str] = {}
        for i in range(len(self.lines)):
            header_text = self._normalize_header_at(i)
            if header_text:
                result[i] = header_text
        return result
    
    def _find_clause_block_by_keywords(self, keywords: List[str]) -> Optional[str]:
        """
        Найти клауза-блок по ключевым словам в заголовке (например ["PRICE"]).
        Возвращает весь блок текста (от заголовка до следующего заголовка или конца).
        Если не найдено — возвращает None.
        """
        clause_headers = self._find_clause_start_indices()
        if not clause_headers:
            return None
        
        target_start_idx: Optional[int] = None
        sorted_indices = sorted(clause_headers.keys())
        
        for idx in sorted_indices:
            header_text = clause_headers[idx]
            upper_header = header_text.upper()
            if any(kw.upper() in upper_header for kw in keywords):
                target_start_idx = idx
                break
        
        if target_start_idx is None:
            return None
        
        # Границы блока: от target_start_idx до следующего заголовка или конца файла
        start_pos_in_sorted = sorted_indices.index(target_start_idx)
        if start_pos_in_sorted + 1 < len(sorted_indices):
            end_idx = sorted_indices[start_pos_in_sorted + 1]
        else:
            end_idx = len(self.lines)
        
        block_lines = self.lines[target_start_idx:end_idx]
        return "\n".join(block_lines).rstrip("\n")
    
    def extract_clause(self, keywords: List[str]) -> Optional[str]:
        """Универсальный метод для извлечения клаузы по ключевым словам"""
        return self._find_clause_block_by_keywords(keywords)


class ManualFieldExtractionService:
    """Сервис для ручного извлечения значений полей из markdown текста документа"""
    
    def __init__(self):
        pass
    
    def extract_by_schema(
        self, 
        text: str, 
        extraction_fields: List[ExtractionField]
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
            # Получаем keywords из поля, если не указаны - используем name
            keywords = field.keywords if field.keywords else []
            
            # Если keywords не указаны — по умолчанию взять name в верхнем регистре
            if not keywords:
                # Например name = "PriceClause" → "PRICE"
                # или name = "Price" → "PRICE"
                base = re.sub(r'CLAUSE$', '', field.name, flags=re.IGNORECASE)
                base = base.upper()
                keywords = [base]
            
            # Извлекаем значение в зависимости от типа поля
            value: Optional[str] = None
            
            if field.type.value == "CLAUSE":
                value = extractor.extract_clause(keywords)
            else:
                # Для других типов пока возвращаем None
                # Можно расширить логику позже
                logger.warning(
                    "ManualFieldExtractionService: unsupported field type %s for field_id=%d",
                    field.type.value,
                    field.id
                )
                value = None
            
            # Создаем ExtractedFieldValue
            # Для ручной экстракции confidence всегда 1.0, так как мы точно знаем что нашли
            # page_num и bbox не применимы для markdown текста
            extracted_value = ExtractedFieldValue(
                field_id=field.id,
                value=value or "",
                confidence=1.0 if value else 0.0,
                page_num=None,
                bbox=None,
            )
            results.append(extracted_value)
        
        logger.info(
            "ManualFieldExtractionService: extracted %d field values from %d fields",
            len([r for r in results if r.value]),
            len(extraction_fields)
        )
        
        return results

