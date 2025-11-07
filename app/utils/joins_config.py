from dataclasses import dataclass
from typing import Optional, Type
from sqlalchemy import ClauseElement

from app.entities.base import Base


@dataclass
class JoinConfig:
    target: Type[Base]  # модель для join
    on_clause: Optional[ClauseElement] = None
    isouter: bool = False  # внешний или внутренний join
    full: bool = False  # полный внешний join (редко)
