from sqlalchemy import ForeignKey, Index, Text, Float
from sqlalchemy.dialects.postgresql import TSVECTOR, ARRAY
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
    content: Mapped[Optional[str]] = mapped_column(Text)
    
    summary: Mapped[Optional[str]] = mapped_column(Text)
    
    field_values: Mapped[list["DocumentFieldValue"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True, back_populates="document"
    )
    
    
    
    
# ----- Конкретное значение поля для документа -----
class DocumentFieldValue(Base, IdMixin, TimestampMixin):
    __tablename__ = "document_field_values"


    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    field_id: Mapped[int] = mapped_column(ForeignKey("extraction_fields.id", ondelete="RESTRICT"), index=True)

    # Всё в одном
    value_text: Mapped[str] = mapped_column(Text)

    # Поиск
    value_tsv: Mapped[Optional[str]] = mapped_column(TSVECTOR)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(1024))

    # доверие/страница/координаты/сырые источники (для дебага и UX)
    confidence: Mapped[Optional[float]]
    page_num: Mapped[Optional[int]]
    bbox: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Float), nullable=True)  # [x1,y1,x2,y2] или подобное

    document: Mapped["Document"] = relationship(back_populates="field_values")
    field: Mapped["ExtractionField"] = relationship()

    __table_args__ = (
        # одно поле может встречаться несколько раз в документе, но каждая позиция уникальна
        Index("uq_dfv_doc_field_pos", "document_id", "field_id", unique=True),
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


