from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import exists, select, update, delete, func, and_, Select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.dto.pagination import PaginationRequest, InfiniteScrollRequest, BaseRequest

from typing import TypeVar, Generic, Type
from app.entities.base import Base
from app.utils.joins_config import JoinConfig

EntityType = TypeVar("EntityType", bound=Base)


def chunked(data: list[dict], chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


class BaseRepository(Generic[EntityType]):
    def __init__(self, session: AsyncSession, entity: Type[EntityType]):
        self.session = session
        self.entity = entity

    async def bulk_upsert(
        self,
        data: list[dict],
        conflict_columns: list[str],
        chunk_size: int = 1000,
        where_conditions: list[Any] = None,
        exclude_update_columns: list[str] = None,
    ) -> None:
        if not data:
            return

        for chunk in chunked(data, chunk_size):
            insert_stmt = pg_insert(self.entity).values(chunk)

            present_keys = set().union(*(row.keys() for row in chunk))
            exclude_update_columns = set(exclude_update_columns or [])

            update_dict = {
                col.name: insert_stmt.excluded[col.name]
                for col in self.entity.__table__.columns
                if col.name in present_keys
                and col.name not in conflict_columns
                and col.name not in exclude_update_columns
                and not col.primary_key
            }

            if not update_dict:
                # Просто вставляем новые записи, если конфликты — ничего не делаем
                upsert_stmt = insert_stmt.on_conflict_do_nothing()
            else:
                upsert_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=conflict_columns,
                    set_=update_dict,
                )
                if where_conditions:
                    upsert_stmt = upsert_stmt.where(*where_conditions)

            await self.session.execute(upsert_stmt)

    async def bulk_create(self, entities: list[EntityType]) -> None:
        """
        Create multiple entities in the database without committing.
        """
        self.session.add_all(entities)
        await self.session.flush()

    async def create(self, entity: EntityType) -> None:
        """
        Create a new entity in the database without committing.
        """
        self.session.add(entity)
        await self.session.flush()

    async def update(self, id: UUID, data: dict) -> None:
        """
        Update an entity by its ID without committing.
        """
        if not data:
            return

        query = update(self.entity).where(self.entity.id == id).values(data)
        await self.session.execute(query)

    async def bulk_update(
        self,
        updates: list[dict],
    ) -> None:
        """
        Bulk update через SQLAlchemy 2.0-style.

        :param updates: Список словарей, каждый должен содержать pk_column и обновляемые значения.
        """
        if not updates:
            return

        stmt = update(self.entity)
        await self.session.execute(stmt, updates)

    async def delete(self, id: UUID) -> None:
        """
        Delete an entity by its ID without committing.
        """
        query = delete(self.entity).where(self.entity.id == id)
        await self.session.execute(query)

    async def soft_delete(self, id: UUID) -> None:
        """
        Delete an entity by its ID without committing.
        """
        query = (
            update(self.entity)
            .where(self.entity.id == id)
            .values(deleted_at=datetime.now(timezone.utc))
        )
        await self.session.execute(query)

    async def bulk_delete(
        self, ids: list[UUID], returning: Optional[list] = None
    ) -> list | None:
        """
        Delete multiple entities by their IDs without committing.
        """
        query = delete(self.entity).where(self.entity.id.in_(ids))
        if returning:
            query = query.returning(*returning)

        result = await self.session.execute(query)
        if returning:
            return result.scalars().all()

    async def soft_bulk_delete(
        self,
        ids: list[UUID],
        returning: Optional[list] = None,
    ) -> list | None:
        """
        Soft-delete multiple entities by setting deleted_at without committing.
        """
        query = (
            update(self.entity)
            .where(self.entity.id.in_(ids))
            .values(deleted_at=datetime.now(timezone.utc))
        )
        if returning:
            query = query.returning(*returning)

        result = await self.session.execute(query)
        if returning:
            return result.scalars().all()

    async def get_one(
        self,
        id: Optional[UUID] = None,
        where: Optional[list] = None,
        options: Optional[list] = None,
        joins: Optional[list[JoinConfig]] = None,
    ) -> Optional[EntityType]:
        """
        Retrieve a single entity by specific filters. By default, searches by ID if provided.
        Uses Limit(1) to ensure only one result is returned.
        DEFAULT WHERE CLAUSE: deleted_at IS NULL if there is such column.
        """
        stmt = select(self.entity)

        if hasattr(self.entity, "deleted_at"):
            stmt = stmt.where(self.entity.deleted_at.is_(None))

        if joins:
            for join in joins:
                stmt = stmt.join(
                    join.target, join.on_clause, isouter=join.isouter, full=join.full
                )

        # Используем options, если они переданы
        if options:
            stmt = stmt.options(*options)

        # Если entity_id указан, ищем по ID
        if id:
            stmt = stmt.where(self.entity.id == id)
        if where:
            stmt = stmt.where(
                and_(*where)
            )  # Если entity_id не указан, использует условия из where

        stmt = stmt.limit(1)  # Добавляем лимит

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(
        self,
        request_query: Optional[BaseRequest] = None,
        where: Optional[list] = None,
        options: Optional[list] = None,
        joins: Optional[list[JoinConfig]] = None,
        with_deleted: bool = False,
    ) -> list[EntityType]:
        """
        Retrieve all entities where deleted_at is None if there is such column.
        """
        stmt = select(self.entity)

        if hasattr(self.entity, "deleted_at") and not with_deleted:
            stmt = stmt.where(getattr(self.entity, "deleted_at").is_(None))

        if joins:
            for join in joins:
                stmt = stmt.join(
                    join.target, join.on_clause, isouter=join.isouter, full=join.full
                )

        if options:
            stmt = stmt.options(*options)

        if request_query:
            filters = request_query.build_filters()
            if filters:
                stmt = stmt.where(and_(*filters))

        if where:
            stmt = stmt.where(and_(*where))

        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_all_and_count(
        self,
        request_query: Optional[PaginationRequest] = None,
        where: Optional[list] = None,
        options: Optional[list] = None,
        joins: Optional[list[JoinConfig]] = None,
        extra_orderable_fields: Optional[dict[str, Any]] = None,
    ) -> tuple[list[EntityType], int]:
        """
        Retrieve all entities of the specified type with optional filters.
        """
        stmt = select(self.entity)

        # --- фильтрация по soft delete ---
        if hasattr(self.entity, "deleted_at"):
            stmt = stmt.where(getattr(self.entity, "deleted_at").is_(None))

        if joins:
            for join in joins:
                stmt = stmt.join(
                    join.target, join.on_clause, isouter=join.isouter, full=join.full
                )

        if options:
            stmt = stmt.options(*options)

        if request_query:
            filters = request_query.build_filters()
            if filters:
                stmt = stmt.where(and_(*filters))

        if where:
            stmt = stmt.where(and_(*where))

        total = await self._count(stmt)  # Получаем общее количество записей

        if request_query:
            order_by_clauses = request_query.build_order_by(
                extra_orderable_fields=extra_orderable_fields
            )
            if order_by_clauses:
                stmt = stmt.order_by(*order_by_clauses)

        if not request_query.disable_pagination:
            stmt = self._paginate_stmt(stmt, request_query.page, request_query.size)

        # Выполняем запрос
        result = await self.session.execute(stmt)
        items = result.scalars().all()

        return items, total

    async def get_all_with_more(
        self,
        request_query: InfiniteScrollRequest,
        where: Optional[list] = None,
        options: Optional[list] = None,
    ) -> tuple[list[EntityType], bool]:
        """
        Retrieve items with pagination based on the infinite scroll approach
        and determine if there are more items to load.
        """
        stmt = select(self.entity)

        if hasattr(self.entity, "deleted_at"):
            stmt = stmt.where(getattr(self.entity, "deleted_at").is_(None))

        if options:
            stmt = stmt.options(*options)

        filters = request_query.build_filters()
        if filters:
            stmt = stmt.where(and_(*filters))

        if where:
            stmt = stmt.where(and_(*where))

        # Получаем limit + 1 записи для определения наличия следующих страниц
        stmt = stmt.limit(request_query.limit + 1).offset(request_query.offset)

        total_count = await self._count(stmt)

        order_by_clauses = request_query.build_order_by()
        if order_by_clauses:
            stmt = stmt.order_by(*order_by_clauses)

        # Выполняем запрос и получаем элементы
        result = await self.session.execute(stmt)
        items = result.scalars().all()

        # Определяем, есть ли еще элементы
        has_more = total_count > request_query.limit

        if has_more:
            items = items[:-1]
        return items, has_more

    async def _count(self, stmt: Select) -> int:
        count_stmt = stmt.with_only_columns(func.count(self.entity.id))

        total_result = await self.session.execute(count_stmt)
        total = total_result.scalar() or 0  # ✅ Если total = None, заменяем на 0
        return total

    def _paginate_stmt(
        self,
        stmt: Select,
        page: int,
        size: int,
    ) -> Select:
        # Применяем пагинацию
        offset = page - 1 if page == 1 else (page - 1) * size
        paginated_query = stmt.limit(size).offset(offset)
        return paginated_query

    async def exists(
        self,
        id: Optional[UUID] = None,
        where: Optional[list] = None,
    ) -> bool:
        """
        Efficiently check if an entity with the given ID or where clause exists using EXISTS.
        """
        stmt = select(exists(self.entity))

        if hasattr(self.entity, "deleted_at"):
            stmt = stmt.where(getattr(self.entity, "deleted_at").is_(None))

        if id:
            stmt = stmt.where(self.entity.id == id)
        if where:
            stmt = stmt.where(and_(*where))

        if hasattr(self.entity, "deleted_at"):
            stmt = stmt.where(self.entity.deleted_at.is_(None))

        result = await self.session.execute(stmt)
        return result.scalar()
