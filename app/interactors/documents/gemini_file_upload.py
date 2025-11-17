import asyncio
import logging
from typing import Optional, List
from pathlib import Path
from google.genai.client import AsyncClient

# Common constant for File Search store name
DEFAULT_FILE_SEARCH_STORE_NAME = "documents"

logger = logging.getLogger(__name__)


class GeminiFileUploadInteractor:
    """
    Interactor for uploading files to Gemini File Search store.
    Enables RAG (Retrieval Augmented Generation) by indexing documents
    for semantic search.
    """
    
    def __init__(self, client: AsyncClient):
        """
        Initialize the interactor with a Gemini async client.
        
        Args:
            client: Google Gemini API async client
        """
        self.client = client
        self._file_search_store = None
        self._file_search_store_name = DEFAULT_FILE_SEARCH_STORE_NAME
    
    async def _get_or_create_file_search_store(self):
        """Lazy initialization of File Search store (async)."""
        if self._file_search_store is None:
            # First, try to find existing store by display_name
            target_display_name = self._file_search_store_name or DEFAULT_FILE_SEARCH_STORE_NAME
            
            try:
                # List all stores and find one with matching display_name
                # Prefer store with documents, or the most recent one
                pager = await self.client.file_search_stores.list()
                candidate_stores = []
                best_store = None
                
                async for store in pager:
                    if getattr(store, 'display_name', None) == target_display_name:
                        candidate_stores.append(store)
                        # Prefer store with active documents
                        active_count = getattr(store, 'active_documents_count', 0) or 0
                        if active_count > 0:
                            best_store = store
                
                # If we found stores, use the best one (with documents) or the first one
                if candidate_stores:
                    self._file_search_store = best_store if best_store else candidate_stores[0]
                
                # If not found, create a new one
                # Note: In case of race condition, this might create duplicate stores,
                # but the next call will find the existing one
                if self._file_search_store is None:
                    try:
                        self._file_search_store = await self.client.file_search_stores.create(
                            config={'display_name': target_display_name}
                        )
                    except Exception as create_error:
                        # If creation fails (e.g., duplicate created by another instance),
                        # try to find it again
                        pager = await self.client.file_search_stores.list()
                        async for store in pager:
                            if getattr(store, 'display_name', None) == target_display_name:
                                self._file_search_store = store
                                break
                        if self._file_search_store is None:
                            raise create_error
            except Exception:
                # If listing fails, try to create a new one
                self._file_search_store = await self.client.file_search_stores.create(
                    config={'display_name': target_display_name}
                )
        return self._file_search_store
    
    async def upload_to_file_search_store(
        self,
        file_path: str | Path,
        display_name: Optional[str] = None,
        chunking_config: Optional[dict] = None,
        custom_metadata: Optional[list[dict]] = None
    ) -> str:
        """
        Directly upload a file to the File Search store.
        This method uploads and imports the file in one operation.
        
        Args:
            file_path: Path to the file to upload
            display_name: Optional display name for the file (visible in citations)
            chunking_config: Optional chunking configuration dict with:
                - white_space_config: dict with max_tokens_per_chunk and max_overlap_tokens
            custom_metadata: Optional list of metadata dicts with keys:
                - key: str
                - string_value: str (for string metadata)
                - numeric_value: float/int (for numeric metadata)
        
        Returns:
            The name of the uploaded file resource
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        config = {}
        if display_name:
            config['display_name'] = display_name
        if chunking_config:
            config['chunking_config'] = chunking_config
        if custom_metadata:
            config['custom_metadata'] = custom_metadata
        
        # Ensure file search store is initialized
        store = await self._get_or_create_file_search_store()
        
        # Verify store still exists (in case it was deleted)
        try:
            store = await self.client.file_search_stores.get(name=store.name)
        except Exception:
            # Store was deleted, clear cache and recreate
            self._file_search_store = None
            store = await self._get_or_create_file_search_store()
        
        # Upload and import file into File Search store (async)
        operation = await self.client.file_search_stores.upload_to_file_search_store(
            file=str(file_path),
            file_search_store_name=store.name,
            config=config if config else None
        )
        
        # Wait until import is complete
        while not operation.done:
            await asyncio.sleep(5)
            operation = await self.client.operations.get(operation)
        
        # Return the file name (operation result should contain file info)
        return operation.name if hasattr(operation, 'name') else str(operation)
    
    async def upload_and_import_file(
        self,
        file_path: str | Path,
        display_name: Optional[str] = None,
        chunking_config: Optional[dict] = None,
        custom_metadata: Optional[list[dict]] = None
    ) -> str:
        """
        Upload a file using the Files API and then import it to the File Search store.
        This is a two-step process: upload first, then import.
        
        Args:
            file_path: Path to the file to upload
            display_name: Optional display name for the file (visible in citations)
            chunking_config: Optional chunking configuration dict
            custom_metadata: Optional list of metadata dicts
        
        Returns:
            The name of the uploaded file resource
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Step 1: Upload the file using the Files API (async)
        upload_config = {}
        if display_name:
            upload_config['name'] = display_name
        
        sample_file = await self.client.files.upload(
            file=str(file_path),
            config=upload_config if upload_config else None
        )
        
        # Step 2: Import the file into the File Search store
        import_config = {}
        if chunking_config:
            import_config['chunking_config'] = chunking_config
        if custom_metadata:
            import_config['custom_metadata'] = custom_metadata
        
        # Ensure file search store is initialized
        store = await self._get_or_create_file_search_store()
        
        # Verify store still exists (in case it was deleted)
        try:
            store = await self.client.file_search_stores.get(name=store.name)
        except Exception:
            # Store was deleted, clear cache and recreate
            self._file_search_store = None
            store = await self._get_or_create_file_search_store()
        
        operation = await self.client.file_search_stores.import_file(
            file_search_store_name=store.name,
            file_name=sample_file.name,
            config=import_config if import_config else None
        )
        
        # Wait until import is complete
        while not operation.done:
            await asyncio.sleep(5)
            operation = await self.client.operations.get(operation)
        
        return sample_file.name
    
    async def get_file_search_store_name(self) -> str:
        """Get the name of the File Search store."""
        store = await self._get_or_create_file_search_store()
        return store.name
    
    async def list_file_search_stores(self):
        """List all File Search stores."""
        stores = []
        pager = await self.client.file_search_stores.list()
        async for store in pager:
            stores.append(store)
        return stores
    
    async def delete_file_search_store(self, force: bool = True):
        """Delete the current File Search store."""
        if self._file_search_store:
            store_name = self._file_search_store.name
            await self.client.file_search_stores.delete(
                name=store_name,
                config={'force': force}
            )
            self._file_search_store = None
    
    # File operations (Files API)
    async def list_files(self) -> List:
        """List all files in the Files API."""
        files = []
        pager = await self.client.files.list()
        async for file in pager:
            files.append(file)
        return files
    
    async def get_file(self, file_name: str):
        """Get file information by name."""
        return await self.client.files.get(name=file_name)
    
    async def delete_file(self, file_name: str):
        """Delete a file by name."""
        await self.client.files.delete(name=file_name)
    
    # File Search Store Documents operations
    async def list_documents_in_store(self):
        """List all documents in the File Search store."""
        store = await self._get_or_create_file_search_store()
        documents = []
        pager = await self.client.file_search_stores.documents.list(
            parent=store.name
        )
        async for doc in pager:
            documents.append(doc)
        return documents
    
    async def get_document_in_store(self, document_name: str):
        """Get document information from File Search store."""
        return await self.client.file_search_stores.documents.get(
            name=document_name
        )
    
    async def delete_document_from_store(self, document_name: str, force: bool = True):
        """
        Delete a document from File Search store.
        
        Args:
            document_name: Full document name
            force: If True, delete document even if it contains chunks (default: True)
        """
        logger.info(f"Deleting document: {document_name}, force={force}")
        
        # According to Gemini API, if document contains chunks, we need to use force=True
        # or delete chunks first. We'll use force=True by default.
        try:
            await self.client.file_search_stores.documents.delete(
                name=document_name,
                config={'force': force} if force else None
            )
            logger.info(f"Successfully deleted document: {document_name}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error deleting document {document_name}: {error_msg}")
            
            # Check if it's a FAILED_PRECONDITION error (non-empty document)
            if "FAILED_PRECONDITION" in error_msg or "non-empty" in error_msg.lower():
                if not force:
                    # Try with force=True
                    logger.info("Document contains chunks, retrying with force=True...")
                    try:
                        await self.client.file_search_stores.documents.delete(
                            name=document_name,
                            config={'force': True}
                        )
                        logger.info(f"Successfully deleted document with force: {document_name}")
                        return
                    except Exception as e2:
                        logger.error(f"Error deleting document with force: {e2}")
                        raise Exception(f"Cannot delete document: it contains chunks. Error: {error_msg}")
                else:
                    raise Exception(f"Cannot delete document even with force=True. Error: {error_msg}")
            
            raise
    
    async def get_chunk_content(self, chunk_name: str) -> Optional[dict]:
        """
        Get chunk content from File Search store.
        
        Args:
            chunk_name: Full chunk name (e.g., 'documents/{doc_name}/chunks/{chunk_id}')
            
        Returns:
            Dict with chunk content and metadata, or None if not found
        """
        try:
            # Try to get chunk using chunks API
            chunk = await self.client.file_search_stores.documents.chunks.get(
                name=chunk_name
            )
            return {
                'name': chunk.name,
                'data': getattr(chunk, 'data', None),
                'custom_metadata': getattr(chunk, 'custom_metadata', None),
            }
        except Exception as e:
            logger.warning(f"Could not get chunk content for {chunk_name}: {e}")
            return None