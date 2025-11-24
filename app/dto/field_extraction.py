"""DTO для извлечения значений полей из документов через OpenAI"""
from pydantic import BaseModel, Field
from pydantic import ConfigDict
from typing import List, Optional


class BoundingBoxEntry(BaseModel):
    """Единичная запись bounding box для конкретной страницы или агрегата."""

    model_config = ConfigDict(extra="forbid")

    page: str = Field(..., description="Номер страницы или специальный ключ (например, 'combined').")
    coords: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Список [x1, y1, x2, y2] с нормализованными координатами (0-1).",
    )


class ExtractedFieldValue(BaseModel):
    """Модель для одного извлеченного значения поля"""

    model_config = ConfigDict(extra="forbid")

    field_id: int = Field(..., description="ID поля извлечения (числовой идентификатор)")
    value: str = Field(..., description="Извлеченное значение поля")
    confidence: Optional[float] = Field(None, description="Уровень уверенности в извлечении (0.0-1.0)")
    page_num: Optional[str] = Field(
        None,
        description="Номер(а) страницы(ниц) документа, где найдено значение. Может быть '2' для одной страницы, '2,3' или '2-3' для многостраничных параграфов.",
    )
    bbox: Optional[List[BoundingBoxEntry]] = Field(
        None,
        description="Список координат bounding box, по одному на страницу или агрегатные ('combined').",
    )


class FieldExtractionOutput(BaseModel):
    """Структурированный вывод извлечения полей из документа"""

    model_config = ConfigDict(extra="forbid")

    extracted_values: List[ExtractedFieldValue] = Field(
        ...,
        description="Список извлеченных значений полей",
    )


class DocumentMarkdownOutput(BaseModel):
    """Структурированный вывод markdown текста документа"""

    model_config = ConfigDict(extra="forbid")

    markdown_text: str = Field(
        ...,
        description="Полный markdown текст документа с сохранением структуры, заголовков и форматирования. Включает все текстовое содержимое, включая заголовки клауз типа '7. PRICE', '9. PAYMENT' и т.п.",
    )

