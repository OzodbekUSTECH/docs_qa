from dishka import Provider, Scope, provide_all

from app.interactors.chat.generate import GenerateChatResponseInteractor
from app.interactors.documents.create import CreateDocumentInteractor
from app.interactors.documents.get import GetAllDocumentsInteractor, GetDocumentByIdInteractor
from app.interactors.documents.delete import DeleteDocumentInteractor


class DocumentsInteractorProvider(Provider):

    scope = Scope.REQUEST

    interactors = provide_all(
        CreateDocumentInteractor,
        GetAllDocumentsInteractor,
        GetDocumentByIdInteractor,
        DeleteDocumentInteractor,
    )
