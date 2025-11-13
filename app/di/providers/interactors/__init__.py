from app.di.providers.interactors.chat import ChatInteractorProvider
from app.di.providers.interactors.documents import DocumentsInteractorProvider
from app.di.providers.interactors.extraction_fields import ExtractionFieldsInteractorProvider

all_interactors = [
    DocumentsInteractorProvider(),
    ChatInteractorProvider(),
    ExtractionFieldsInteractorProvider(),
]
