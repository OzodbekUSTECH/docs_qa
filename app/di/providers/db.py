from typing import AsyncIterable

from dishka import Provider, Scope, provide
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


class DBProvider(Provider):
    
    @provide(scope=Scope.APP)
    def provide_engine(self) -> AsyncEngine:
        return create_async_engine(  # для FastAPI
            settings.database_url,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_pre_ping=True,
            pool_recycle=1800,
        )

    @provide(scope=Scope.APP)
    def provide_session_maker(
        self, engine: AsyncEngine
    ) -> async_sessionmaker[AsyncSession]:
        return async_sessionmaker(bind=engine, expire_on_commit=False)

    @provide(scope=Scope.REQUEST)
    async def provide_session(
        self, session_maker: async_sessionmaker[AsyncSession]
    ) -> AsyncIterable[AsyncSession]:
        async with session_maker() as session:
            yield session
