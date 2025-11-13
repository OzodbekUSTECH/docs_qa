
from app.repositories.extraction_fields import ExtractionFieldsRepository
from app.dto.extraction_fields import GetExtractionFieldsParams, ExtractionFieldListResponse, ExtractionFieldResponse
from app.dto.pagination import PaginatedResponse
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages



class GetAllExtractionFieldsInteractor:
    def __init__(self, extraction_fields_repo: ExtractionFieldsRepository):
        self.extraction_fields_repo = extraction_fields_repo
        
    async def execute(self, params: GetExtractionFieldsParams) -> PaginatedResponse[ExtractionFieldListResponse]:
        extraction_fields, total = await self.extraction_fields_repo.get_all_and_count(
            request_query=params,
        )
        return PaginatedResponse[ExtractionFieldListResponse](
            items=[ExtractionFieldListResponse.model_validate(extraction_field) for extraction_field in extraction_fields],
            total=total,
            page=params.page,
            size=params.size
        )
        
        

class GetExtractionFieldByIdInteractor:
    def __init__(self, extraction_fields_repo: ExtractionFieldsRepository):
        self.extraction_fields_repo = extraction_fields_repo
        
    async def execute(self, id: int) -> ExtractionFieldResponse:
        extraction_field = await self.extraction_fields_repo.get_one(id=id)
        if not extraction_field:
            raise AppError(404, ErrorMessages.EXTRACTION_FIELD_NOT_FOUND)
        return ExtractionFieldResponse.model_validate(extraction_field)