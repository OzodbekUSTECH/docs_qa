from decimal import Decimal
from datetime import datetime, time
from sqlalchemy import DateTime, Numeric, Time
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    __abstract__ = True

    type_annotation_map = {
        datetime: DateTime(timezone=True),
        time: Time(timezone=True),
        dict: JSONB,
        Decimal: Numeric(20, 3),
    }
