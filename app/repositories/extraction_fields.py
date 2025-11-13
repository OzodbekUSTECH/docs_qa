from sqlalchemy.ext.asyncio import AsyncSession

from app.entities import ExtractionField
from app.repositories.base import BaseRepository

class ExtractionFieldsRepository(BaseRepository[ExtractionField]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, entity=ExtractionField)