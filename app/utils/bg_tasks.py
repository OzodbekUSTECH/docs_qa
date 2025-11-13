from app.di.containers import app_container
from app.services.extract_field_values import ExtractDocumentFieldValuesService

async def process_document(document_id: int, extraction_field_ids: list[int]):
    try:
        print(f"Processing document {document_id}")
        async with app_container() as container:
            document_chunk_service = await container.get(ExtractDocumentFieldValuesService)
            await document_chunk_service.execute(document_id, extraction_field_ids)
    except Exception as e:
        print(f"Error processing document {document_id}: {e}")
        