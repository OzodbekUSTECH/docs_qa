from app.entities.extracion_fields import ExtractionField
from app.repositories.extraction_fields import ExtractionFieldsRepository
from app.dto.extraction_fields import CreateExtractionFieldRequest, ExtractionFieldResponse
from app.dto.common import BaseModelResponse
from app.repositories.uow import UnitOfWork

class CreateExtractionFieldInteractor:
    def __init__(self, uow: UnitOfWork, extraction_fields_repo: ExtractionFieldsRepository):
        self.uow = uow
        self.extraction_fields_repo = extraction_fields_repo
        
    async def execute(self, request: CreateExtractionFieldRequest) -> BaseModelResponse:
        extraction_field = ExtractionField(**request.model_dump())
        await self.extraction_fields_repo.create(extraction_field)
        await self.uow.commit()
        return BaseModelResponse.model_validate(extraction_field)