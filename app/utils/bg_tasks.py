from app.di.containers import app_container
from app.services.document_chunk_service import DocumentChunkService

async def process_document(document_id: int):
    try:
        print(f"Processing document {document_id}")
        async with app_container() as container:
            document_chunk_service = await container.get(DocumentChunkService)
            await document_chunk_service.execute(document_id)
    except Exception as e:
        print(f"Error processing document {document_id}: {e}")
        