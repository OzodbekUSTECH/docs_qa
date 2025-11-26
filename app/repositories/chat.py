from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload

from app.entities.chat import ChatSession, ChatMessage
from app.repositories.base import BaseRepository


class ChatRepository(BaseRepository):
    async def create_session(self, title: str | None = None) -> ChatSession:
        session = ChatSession(title=title)
        self.session.add(session)
        await self.session.flush()
        await self.session.commit()
        return session

    async def get_session(self, session_id: UUID) -> ChatSession | None:
        stmt = (
            select(ChatSession)
            .options(selectinload(ChatSession.messages))
            .where(ChatSession.id == session_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_sessions(self) -> list[ChatSession]:
        stmt = select(ChatSession).order_by(ChatSession.updated_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete_session(self, session_id: UUID) -> None:
        stmt = delete(ChatSession).where(ChatSession.id == session_id)
        await self.session.execute(stmt)
        await self.session.commit()

    async def add_message(self, session_id: UUID, role: str, content: str, citations: dict | None = None, search_results: dict | None = None, thinking_process: dict | None = None) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id, 
            role=role, 
            content=content, 
            citations=citations,
            search_results=search_results,
            thinking_process=thinking_process
        )
        self.session.add(message)
        
        # Update session updated_at
        chat_session = await self.get_session(session_id)
        if chat_session:
            # If it's the first user message and title is empty, set title
            if not chat_session.title and role == "user":
                # Truncate title to 50 chars
                chat_session.title = content[:50] + ("..." if len(content) > 50 else "")
            
            # Force update updated_at
            from datetime import datetime
            chat_session.updated_at = datetime.utcnow()
            self.session.add(chat_session)
            
        await self.session.flush()
        await self.session.commit()
        return message
