"""
Text normalization utilities for robust text matching.
Handles Unicode dashes, quotes, spaces, and other special characters.
"""

import re
import unicodedata
from typing import Optional


def normalize_unicode(text: str) -> str:
    """
    Нормализация Unicode символов для надежного сопоставления текста.
    
    Обрабатывает:
    - Различные виды тире (–, —, ‐, −) → обычный дефис (-)
    - Различные кавычки (" " ' ') → обычные ("' )
    - Неразрывные пробелы → обычные пробелы
    - Греческие и латинские похожие буквы → латинский вариант
    - Множественные пробелы → один пробел
    
    Args:
        text: Исходный текст
        
    Returns:
        Нормализованный текст
    """
    if not text:
        return ""
    
    # 1. Нормализуем все тире к обычному дефису
    # U+2013 (EN DASH), U+2014 (EM DASH), U+2212 (MINUS SIGN), 
    # U+2010 (HYPHEN), U+002D (HYPHEN-MINUS)
    dash_variants = [
        '\u2013',  # – EN DASH
        '\u2014',  # — EM DASH
        '\u2212',  # − MINUS SIGN
        '\u2010',  # ‐ HYPHEN
        '\u2011',  # ‑ NON-BREAKING HYPHEN
        '\u2043',  # ⁃ HYPHEN BULLET
        '\ufe58',  # ﹘ SMALL EM DASH
        '\ufe63',  # ﹣ SMALL HYPHEN-MINUS
        '\uff0d',  # － FULLWIDTH HYPHEN-MINUS
    ]
    for dash in dash_variants:
        text = text.replace(dash, '-')
    
    # 2. Нормализуем кавычки
    quote_variants = {
        '\u201c': '"',  # " LEFT DOUBLE QUOTATION MARK
        '\u201d': '"',  # " RIGHT DOUBLE QUOTATION MARK
        '\u2018': "'",  # ' LEFT SINGLE QUOTATION MARK
        '\u2019': "'",  # ' RIGHT SINGLE QUOTATION MARK
        '\u00ab': '"',  # « LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
        '\u00bb': '"',  # » RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
        '\u2039': "'",  # ‹ SINGLE LEFT-POINTING ANGLE QUOTATION MARK
        '\u203a': "'",  # › SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
    }
    for old, new in quote_variants.items():
        text = text.replace(old, new)
    
    # 3. Нормализуем пробелы
    space_variants = [
        '\u00a0',  # NO-BREAK SPACE
        '\u2000',  # EN QUAD
        '\u2001',  # EM QUAD
        '\u2002',  # EN SPACE
        '\u2003',  # EM SPACE
        '\u2004',  # THREE-PER-EM SPACE
        '\u2005',  # FOUR-PER-EM SPACE
        '\u2006',  # SIX-PER-EM SPACE
        '\u2007',  # FIGURE SPACE
        '\u2008',  # PUNCTUATION SPACE
        '\u2009',  # THIN SPACE
        '\u200a',  # HAIR SPACE
        '\u202f',  # NARROW NO-BREAK SPACE
        '\u205f',  # MEDIUM MATHEMATICAL SPACE
        '\u3000',  # IDEOGRAPHIC SPACE
    ]
    for space in space_variants:
        text = text.replace(space, ' ')
    
    # 4. Нормализуем похожие греческие/кириллические буквы на латинские
    # Часто OCR или копипаст могут спутать эти символы
    char_mapping = {
        # Греческие буквы похожие на латинские
        'Α': 'A',  # Greek Capital Letter Alpha
        'Β': 'B',  # Greek Capital Letter Beta
        'Ε': 'E',  # Greek Capital Letter Epsilon
        'Ζ': 'Z',  # Greek Capital Letter Zeta
        'Η': 'H',  # Greek Capital Letter Eta
        'Ι': 'I',  # Greek Capital Letter Iota
        'Κ': 'K',  # Greek Capital Letter Kappa
        'Μ': 'M',  # Greek Capital Letter Mu
        'Ν': 'N',  # Greek Capital Letter Nu
        'Ο': 'O',  # Greek Capital Letter Omicron
        'Ρ': 'P',  # Greek Capital Letter Rho
        'Τ': 'T',  # Greek Capital Letter Tau
        'Υ': 'Y',  # Greek Capital Letter Upsilon
        'Χ': 'X',  # Greek Capital Letter Chi
        # Кириллические буквы похожие на латинские
        'А': 'A',  # Cyrillic Capital Letter A
        'В': 'B',  # Cyrillic Capital Letter Ve
        'Е': 'E',  # Cyrillic Capital Letter Ie
        'К': 'K',  # Cyrillic Capital Letter Ka
        'М': 'M',  # Cyrillic Capital Letter Em
        'Н': 'H',  # Cyrillic Capital Letter En
        'О': 'O',  # Cyrillic Capital Letter O
        'Р': 'P',  # Cyrillic Capital Letter Er
        'С': 'C',  # Cyrillic Capital Letter Es
        'Т': 'T',  # Cyrillic Capital Letter Te
        'У': 'Y',  # Cyrillic Capital Letter U
        'Х': 'X',  # Cyrillic Capital Letter Ha
    }
    for old_char, new_char in char_mapping.items():
        text = text.replace(old_char, new_char)
    
    # 5. Удаляем невидимые символы (zero-width, soft hyphens, etc.)
    invisible_chars = [
        '\u200b',  # ZERO WIDTH SPACE
        '\u200c',  # ZERO WIDTH NON-JOINER
        '\u200d',  # ZERO WIDTH JOINER
        '\ufeff',  # ZERO WIDTH NO-BREAK SPACE (BOM)
        '\u00ad',  # SOFT HYPHEN
    ]
    for char in invisible_chars:
        text = text.replace(char, '')
    
    # 6. Нормализуем множественные пробелы в один
    text = re.sub(r'\s+', ' ', text)
    
    # 7. Убираем пробелы в начале и конце
    text = text.strip()
    
    return text


def normalize_for_search(text: str) -> str:
    """
    Полная нормализация для поиска: Unicode + uppercase + whitespace.
    
    Args:
        text: Исходный текст
        
    Returns:
        Нормализованный текст для сравнения
    """
    text = normalize_unicode(text)
    text = text.upper()
    text = " ".join(text.split())  # Normalize whitespace
    return text


def fuzzy_contains(haystack: str, needle: str, threshold: float = 0.8) -> bool:
    """
    Нечеткий поиск подстроки.
    
    Args:
        haystack: Текст для поиска
        needle: Искомая подстрока
        threshold: Минимальный порог совпадения (0.0 - 1.0)
        
    Returns:
        True если найдено совпадение выше порога
    """
    haystack_norm = normalize_for_search(haystack)
    needle_norm = normalize_for_search(needle)
    
    # Точное совпадение
    if needle_norm in haystack_norm:
        return True
    
    # Если искомая фраза короткая, требуем точное совпадение
    if len(needle_norm) < 10:
        return False
    
    # Разбиваем на слова и проверяем совпадение слов
    needle_words = needle_norm.split()
    haystack_words = haystack_norm.split()
    
    if not needle_words or not haystack_words:
        return False
    
    # Подсчитываем совпадающие слова
    matches = sum(1 for word in needle_words if word in haystack_words)
    score = matches / len(needle_words)
    
    return score >= threshold


def find_best_match_position(text: str, search_phrase: str) -> Optional[int]:
    """
    Находит лучшую позицию для поисковой фразы в тексте.
    Возвращает индекс символа начала совпадения или None.
    
    Args:
        text: Текст для поиска
        search_phrase: Искомая фраза
        
    Returns:
        Индекс начала совпадения или None
    """
    text_norm = normalize_for_search(text)
    phrase_norm = normalize_for_search(search_phrase)
    
    # Точное совпадение
    idx = text_norm.find(phrase_norm)
    if idx != -1:
        return idx
    
    # Попробуем найти по первым N словам
    phrase_words = phrase_norm.split()
    if len(phrase_words) >= 3:
        # Ищем первые 3 слова
        first_3_words = " ".join(phrase_words[:3])
        idx = text_norm.find(first_3_words)
        if idx != -1:
            return idx
    
    return None

