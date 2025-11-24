from dishka import Provider, Scope, provide, provide_all
from app.interactors.chat.gemini_chat import GeminiChatInteractor
from app.interactors.documents.gemini_file_upload import GeminiFileUploadInteractor
from app.services.extract_field_values import ExtractDocumentFieldValuesService
from app.services.manual_field_extraction import ManualFieldExtractionService
from app.services.file_service import FileService
from app.services.document_ai_service import DocumentAIService
from app.services.ocr_field_extraction import OCRFieldExtractionService
from app.services.gemini_ocr_service import GeminiOCRService


class ServicesProvider(Provider):
    
    scope = Scope.APP
    
    services = provide_all(
        FileService,
        ManualFieldExtractionService,
        DocumentAIService,
        OCRFieldExtractionService,
        GeminiOCRService,
        
        
        GeminiChatInteractor,
        GeminiFileUploadInteractor,
    )
    
    extract_docs_fields = provide(ExtractDocumentFieldValuesService, scope=Scope.REQUEST)

