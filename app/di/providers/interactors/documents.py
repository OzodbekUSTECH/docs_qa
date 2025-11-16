from dishka import Provider, Scope, provide_all

from app.interactors.documents.create import CreateDocumentInteractor
from app.interactors.documents.get import GetAllDocumentsInteractor, GetDocumentByIdInteractor
from app.interactors.documents.delete import DeleteDocumentInteractor
from app.interactors.documents.update import UpdateDocumentPartiallyInteractor


class DocumentsInteractorProvider(Provider):

    scope = Scope.REQUEST

    interactors = provide_all(
        CreateDocumentInteractor,
        GetAllDocumentsInteractor,
        GetDocumentByIdInteractor,
        DeleteDocumentInteractor,
        UpdateDocumentPartiallyInteractor
    )
