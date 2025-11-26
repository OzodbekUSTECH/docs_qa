from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.entities.base import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow, onupdate=datetime.utcnow
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(Text)  # "user" or "model"
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[dict] = mapped_column(JSONB, nullable=True)  # Store citation metadata
    search_results: Mapped[dict] = mapped_column(JSONB, nullable=True)  # Store all search results from hybrid_search
    thinking_process: Mapped[dict] = mapped_column(JSONB, nullable=True)  # Store thinking process data
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
