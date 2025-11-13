"""DTO для извлечения значений полей из документов через OpenAI"""
from pydantic import BaseModel, Field
from typing import List, Optional


class ExtractedFieldValue(BaseModel):
    """Модель для одного извлеченного значения поля"""
    field_id: int = Field(..., description="ID поля извлечения (числовой идентификатор)")
    value: str = Field(..., description="Извлеченное значение поля")
    confidence: Optional[float] = Field(None, description="Уровень уверенности в извлечении (0.0-1.0)")
    page_num: Optional[int] = Field(
        None, 
        description="Номер страницы документа, где найдено значение. Должен быть целым числом >= 1, где 1 означает первую страницу документа. Если документ одностраничный, всегда используй 1."
    )
    bbox: Optional[List[float]] = Field(
        None, 
        description="Координаты bounding box в формате [x1, y1, x2, y2], где координаты нормализованы относительно размеров страницы (0.0-1.0). x1, y1 - левый верхний угол, x2, y2 - правый нижний угол области, где находится извлеченное значение. Все 4 значения должны быть в диапазоне [0.0, 1.0]."
    )


class FieldExtractionOutput(BaseModel):
    """Структурированный вывод извлечения полей из документа"""
    extracted_values: List[ExtractedFieldValue] = Field(
        ..., 
        description="Список извлеченных значений полей"
    )

