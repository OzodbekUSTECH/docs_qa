from dishka import Provider, Scope, provide, provide_all
from app.services.extract_field_values import ExtractDocumentFieldValuesService
from app.services.file_service import FileService


class ServicesProvider(Provider):
    
    scope = Scope.APP
    
    services = provide_all(
        FileService,
    )
    
    extract_docs_fields = provide(ExtractDocumentFieldValuesService, scope=Scope.REQUEST)

