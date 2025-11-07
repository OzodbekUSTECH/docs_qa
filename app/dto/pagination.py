from pydantic import model_validator, Field
from typing import Any, List, Optional, Type
from sqlalchemy import or_
from pydantic import BaseModel


class BaseRequest(BaseModel):
    search: Optional[str] = Field(None, min_length=3)
    order_by: Optional[str] = None  # Например: "-first_name, last_name, -created_at"

    class Constants:
        filter_map = {}
        searchable_fields = []
        orderable_fields = {}

    def build_filters(self) -> List:
        filters = []
        filter_map = {}

        # Собираем filter_map из иерархии классов
        for cls in reversed(self.__class__.mro()):
            config: Optional[Type] = getattr(cls, "Constants", None)
            if config and hasattr(config, "filter_map"):
                filter_map.update(config.filter_map)

        # Добавляем фильтры по конкретным полям
        for field, builder in filter_map.items():
            value = getattr(self, field, None)
            if value is not None:
                filters.append(builder(value))

        # Добавляем фильтр поиска, если задан
        if self.search:
            searchable_fields = []
            for cls in reversed(self.__class__.mro()):
                config = getattr(cls, "Constants", None)
                if config and hasattr(config, "searchable_fields"):
                    searchable_fields.extend(config.searchable_fields)
            if searchable_fields:
                search_filter_clauses = []
                for field in searchable_fields:
                    if callable(field):
                        # поле — функция: вызовем с self.search
                        search_filter_clauses.append(field(self.search))
                    else:
                        # обычное поле: применим ILIKE
                        search_filter_clauses.append(field.ilike(f"%{self.search}%"))

                filters.append(or_(*search_filter_clauses))

        return filters

    def get_orderable_fields(self) -> dict[str, Any]:
        """
        Возвращает словарь доступных полей для сортировки.
        Может быть переопределён или расширен внешне.
        """
        orderable_fields = {}
        for cls in reversed(self.__class__.mro()):
            config = getattr(cls, "Constants", None)
            if config and hasattr(config, "orderable_fields"):
                orderable_fields.update(config.orderable_fields)
        return orderable_fields

    def build_order_by(
        self, extra_orderable_fields: Optional[dict[str, Any]] = None
    ) -> Optional[list[Any]]:
        """
        Создаёт order_by выражения, с учётом внешнего расширения.
        """
        if not self.order_by:
            return None

        orderable_fields = self.get_orderable_fields()
        if extra_orderable_fields:
            orderable_fields.update(extra_orderable_fields)

        clauses = []
        for raw_field in map(str.strip, self.order_by.split(",")):
            if not raw_field:
                continue
            desc_order = raw_field.startswith("-")
            field_name = raw_field[1:] if desc_order else raw_field

            column = orderable_fields.get(field_name)
            if column is None:
                continue
            clauses.append(column.desc() if desc_order else column.asc())

        return clauses if clauses else None


class PaginationRequest(BaseRequest):
    page: int = Field(1, ge=1)
    size: int = Field(50, ge=1, le=100)
    disable_pagination: bool = False


class InfiniteScrollRequest(BaseRequest):
    offset: int = Field(0, ge=0)
    limit: int = Field(20, ge=1, le=100)


# --- RESPONSES ---


class PaginatedResponse[ResponseModel](BaseModel):
    items: list[ResponseModel]
    total: int = Field(0, description="Total number of items")
    page: int = Field(1, ge=1, description="Current page number")
    size: int = Field(1, ge=1, description="Items per page")
    pages: int = Field(0, description="Total number of pages")

    @model_validator(mode="after")
    def compute_fields(self) -> "PaginatedResponse":
        self.pages = (self.total + self.size - 1) // self.size if self.size > 0 else 0
        return self


class InfiniteScrollResponse[ResponseModel](BaseModel):
    items: list[ResponseModel]
    has_more: bool
