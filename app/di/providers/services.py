from dishka import Provider, Scope, provide, provide_all
from app.services.file_service import FileService
from app.services.document_chunk_service import DocumentChunkService
from app.repositories.documents import DocumentChunksRepository


class ServicesProvider(Provider):
    
    scope = Scope.APP
    
    services = provide_all(
        FileService,
    )
    
    document_chunk_service = provide(DocumentChunkService, scope=Scope.REQUEST)

