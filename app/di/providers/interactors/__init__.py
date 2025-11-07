from app.di.providers.interactors.chat import ChatInteractorProvider
from app.di.providers.interactors.documents import DocumentsInteractorProvider

all_interactors = [
    DocumentsInteractorProvider(),
    ChatInteractorProvider(),
]
