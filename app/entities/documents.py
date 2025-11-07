from sqlalchemy import Computed, ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import TSVECTOR
from app.entities.base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.entities.mixins.timestamp_mixin import TimestampMixin
from app.entities.mixins.id_mixin import IdMixin
from app.utils.enums import DocumentStatus, DocumentType
from pgvector.sqlalchemy import Vector
from typing import Optional


class Document(Base,IdMixin, TimestampMixin):
    __tablename__ = "documents"


    filename: Mapped[str]
    file_path: Mapped[str]
    content_type: Mapped[str]
    file_hash: Mapped[str] = mapped_column(index=True, unique=True)
    status: Mapped[DocumentStatus] = mapped_column(default=DocumentStatus.PENDING)
    content: Mapped[Optional[str]] = mapped_column(Text)
    type: Mapped[DocumentType] = mapped_column(default=DocumentType.OTHER)
    
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    
class DocumentChunk(Base,IdMixin):
    __tablename__ = "document_chunks"

    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    is_important: Mapped[bool] = mapped_column(default=False, index=True)
    # FTS 
    content_tsv: Mapped[Optional[str]] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple', unaccent(content))",
            persisted=True
        ),
        nullable=True
    )
    
    # 1024 под text-embedding-3-small
    embedding: Mapped[list[float]] = mapped_column(Vector(1024))

    __table_args__ = (
        Index("uq_chunk_doc_idx", "document_id", "chunk_index", unique=True),
        Index("ix_chunk_tsv", "content_tsv", postgresql_using="gin"),
        # HNSW-индекс (если хочешь держать прямо в модели; альтернативно сделай это в миграции)
        Index(
            "ix_chunk_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
        # Триграммы для fuzzy/LIKE (опционально, но очень полезно при OCR)
        Index(
            "ix_chunk_content_trgm",
            "content",
            postgresql_using="gin",
            postgresql_ops={"content": "gin_trgm_ops"},
        ),
    )
    
    