from app.repositories.uow import UnitOfWork

from app.repositories.extraction_fields import ExtractionFieldsRepository
from app.dto.common import BaseResponse
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages

class DeleteExtractionFieldInteractor:
    def __init__(self, uow: UnitOfWork, extraction_fields_repo: ExtractionFieldsRepository):
        self.uow = uow
        self.extraction_fields_repo = extraction_fields_repo
        
        
    async def execute(self, id: int) -> BaseResponse:
        extraction_field = await self.extraction_fields_repo.get_one(id=id)
        if not extraction_field:
            raise AppError(404, ErrorMessages.EXTRACTION_FIELD_NOT_FOUND)
        await self.extraction_fields_repo.soft_delete(id)
        await self.uow.commit()
        return BaseResponse()