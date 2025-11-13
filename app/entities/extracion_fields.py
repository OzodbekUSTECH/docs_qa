from sqlalchemy import Text, Enum as SAEnum
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from app.entities.base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.entities.mixins.timestamp_mixin import TimestampMixin
from app.entities.mixins.id_mixin import IdMixin
from app.utils.enums import DocumentType, ExtractionFieldType
from typing import Optional


class ExtractionField(Base,IdMixin, TimestampMixin):
    __tablename__ = "extraction_fields"

    name: Mapped[str]
    identifier: Mapped[str] = mapped_column(index=True, unique=True)
    
    short_description: Mapped[Optional[str]]
    type: Mapped[ExtractionFieldType]
    document_types: Mapped[list[DocumentType]] = mapped_column(ARRAY(SAEnum(DocumentType)))
    
    prompt: Mapped[str]
    
    examples: Mapped[list[str]] = mapped_column(ARRAY(Text))
    
    