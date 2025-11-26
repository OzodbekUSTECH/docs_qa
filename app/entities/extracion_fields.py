from sqlalchemy import Text, Enum as SAEnum, text
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from app.entities.base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.entities.mixins.timestamp_mixin import TimestampMixin
from app.entities.mixins.id_mixin import IdMixin
from app.utils.enums import DocumentType, ExtractionFieldType, FieldOccurrence, ExtractionBy
from typing import Optional


class ExtractionField(Base,IdMixin, TimestampMixin):
    __tablename__ = "extraction_fields"

    name: Mapped[str]
    identifier: Mapped[Optional[str]] 
    occurrence: Mapped[FieldOccurrence] = mapped_column(
        SAEnum(FieldOccurrence, name="fieldoccurrence"),
        default=FieldOccurrence.OPTIONAL_ONCE,
        server_default=text("'OPTIONAL_ONCE'::fieldoccurrence")
    )
    
    short_description: Mapped[Optional[str]]
    type: Mapped[ExtractionFieldType]
    document_types: Mapped[list[DocumentType]] = mapped_column(ARRAY(SAEnum(DocumentType)))
    
    # for code base extraction
    keywords: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, server_default=text("'{}'::text[]"))
    
    # for ai
    use_ai: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    extraction_by: Mapped[ExtractionBy] = mapped_column(SAEnum(ExtractionBy, name="extractionby"), default=ExtractionBy.GEMINI_AI, server_default=text("'DOCUMENT_AI'::extractionby"))
    prompt: Mapped[Optional[str]]
    
    examples: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, server_default=text("'{}'::text[]"))
    
    