from sqlalchemy import ForeignKey, Index, Text, Float, text
from sqlalchemy.dialects.postgresql import TSVECTOR, ARRAY, JSONB
from app.entities.base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.entities.mixins.timestamp_mixin import TimestampMixin
from app.entities.mixins.id_mixin import IdMixin
from app.utils.enums import DocumentStatus, DocumentType
from pgvector.sqlalchemy import Vector
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.entities.extracion_fields import ExtractionField


class Document(Base,IdMixin, TimestampMixin):
    __tablename__ = "documents"


    filename: Mapped[str]
    file_path: Mapped[str]
    content_type: Mapped[str]
    file_hash: Mapped[str] = mapped_column(index=True, unique=True)
    status: Mapped[DocumentStatus] = mapped_column(default=DocumentStatus.PENDING)
    type: Mapped[DocumentType] = mapped_column(default=DocumentType.OTHER)
    content: Mapped[Optional[str]] = mapped_column(Text)  # Markdown text
    content_coordinates: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)  # Coordinates map for markdown elements
    
    summary: Mapped[Optional[str]] = mapped_column(Text)
    
    field_values: Mapped[list["DocumentFieldValue"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True, back_populates="document"
    )
    
    
    
    
# ----- Конкретное значение поля для документа -----
class DocumentFieldValue(Base, IdMixin, TimestampMixin):
    __tablename__ = "document_field_values"


    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    field_id: Mapped[Optional[int]] = mapped_column(ForeignKey("extraction_fields.id", ondelete="RESTRICT"), index=True, nullable=True)
    custom_field_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # For custom fields without field_id

    # Всё в одном
    value_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Поиск
    value_tsv: Mapped[Optional[str]] = mapped_column(TSVECTOR)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(1024))

    # доверие/страница/координаты/сырые источники (для дебага и UX)
    confidence: Mapped[Optional[float]]
    page_num: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Всегда одна страница (например "1", "2"), каждая страница = отдельная запись
    bbox: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)  # JSONB для хранения координат: {page_num: [x1, y1, x2, y2]}

    document: Mapped["Document"] = relationship(back_populates="field_values")
    field: Mapped[Optional["ExtractionField"]] = relationship()

    __table_args__ = (
        # Одно поле может встречаться несколько раз в документе на разных страницах
        # И даже несколько раз на одной странице - каждая запись уникальна по id
        # page_num всегда одна страница (не "2,3")
        # НЕТ уникальности по field_id + page_num - позволяем дубликаты
        # Note: PostgreSQL partial indexes require text() for WHERE clause
        # Индексы для быстрого поиска (не уникальные):
        Index("ix_dfv_doc_field_page", "document_id", "field_id", "page_num", postgresql_where=text("field_id IS NOT NULL")),
        Index("ix_dfv_doc_custom_page", "document_id", "custom_field_name", "page_num", postgresql_where=text("field_id IS NULL")),
        Index("ix_dfv_value_tsv", "value_tsv", postgresql_using="gin"),
        Index(
            "ix_dfv_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
        # триграммы полезны против OCR-шумов и LIKE '%...%'
        Index("ix_dfv_value_trgm", "value_text", postgresql_using="gin", postgresql_ops={"value_text": "gin_trgm_ops"}),
    )


