from dishka import Provider, Scope, provide_all

from app.interactors.extraction_fields.create import CreateExtractionFieldInteractor
from app.interactors.extraction_fields.get import GetAllExtractionFieldsInteractor, GetExtractionFieldByIdInteractor
from app.interactors.extraction_fields.update import UpdateExtractionFieldInteractor
from app.interactors.extraction_fields.delete import DeleteExtractionFieldInteractor


class ExtractionFieldsInteractorProvider(Provider):

    scope = Scope.REQUEST

    interactors = provide_all(
        CreateExtractionFieldInteractor,
        GetAllExtractionFieldsInteractor,
        GetExtractionFieldByIdInteractor,
        UpdateExtractionFieldInteractor,
        DeleteExtractionFieldInteractor,
    )
