from app.repositories.uow import UnitOfWork
from app.repositories.extraction_fields import ExtractionFieldsRepository
from app.dto.extraction_fields import UpdateExtractionFieldRequest
from app.dto.common import BaseModelResponse
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from app.entities.extracion_fields import ExtractionField



class UpdateExtractionFieldInteractor:
    def __init__(self, uow: UnitOfWork, extraction_fields_repo: ExtractionFieldsRepository):
        self.uow = uow
        self.extraction_fields_repo = extraction_fields_repo
        
    async def execute(self, id: int, request: UpdateExtractionFieldRequest) -> BaseModelResponse:
        extraction_field = await self.extraction_fields_repo.get_one(id=id)
        if not extraction_field:
            raise AppError(404, ErrorMessages.EXTRACTION_FIELD_NOT_FOUND)
        await self.extraction_fields_repo.update(id, request.model_dump(exclude_unset=True))
        await self.uow.commit()
        return BaseModelResponse.model_validate(extraction_field)