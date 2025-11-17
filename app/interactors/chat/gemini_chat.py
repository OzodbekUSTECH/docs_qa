from typing import AsyncIterator, Optional
import json
import logging
import re

from google.genai.client import AsyncClient
from google.genai import types

from app.dto.chat import GenerateChatRequest
from app.interactors.documents.gemini_file_upload import DEFAULT_FILE_SEARCH_STORE_NAME

logger = logging.getLogger(__name__)


class GeminiChatInteractor:
    """
    Interactor for chat with Gemini using File Search for RAG.
    Enables document-based conversations using Gemini's File Search tool.
    """
    
    def __init__(
        self,
        client: AsyncClient,
    ):
        """
        Initialize the Gemini chat interactor.
        
        Args:
            client: Google Gemini API async client
            file_search_store_name: Optional name of existing File Search store.
                                   If not provided, uses "documents" as default
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
    
    async def stream(self, request: GenerateChatRequest) -> AsyncIterator[bytes]:
        """
        Stream chat response using Gemini with File Search.
        
        Args:
            request: Chat request with prompt, document_ids, model, and session_id
            
        Yields:
            JSON-encoded chunks of the streaming response
        """
        # Get or create File Search store
        store = await self._get_or_create_file_search_store()
        
        # Verify store still exists (in case it was deleted)
        try:
            store = await self.client.file_search_stores.get(name=store.name)
        except Exception:
            # Store was deleted, clear cache and recreate
            self._file_search_store = None
            store = await self._get_or_create_file_search_store()
        
        file_search_store_name = store.name
        
        # Log documents in store for debugging
        try:
            pager = await self.client.file_search_stores.documents.list(parent=store.name)
            doc_count = 0
            async for doc in pager:
                doc_count += 1
                logger.info(f"Document in store: {doc.name}, display_name: {getattr(doc, 'display_name', None)}")
            logger.info(f"Total documents in store: {doc_count}")
        except Exception as e:
            logger.warning(f"Could not list documents in store: {e}")
        
        # Build contents from chat history if provided
        contents = []
        if hasattr(request, 'history') and request.history:
            # Convert history to Gemini Content format
            for msg in request.history:
                role = msg.get('role', 'user')
                content_text = msg.get('content', '')
                # Strip HTML tags from assistant messages for API
                if role == 'assistant' and content_text:
                    # Remove HTML tags but keep text
                    content_text = re.sub(r'<[^>]+>', '', content_text)
                    # Remove citations section if present
                    content_text = re.sub(r'<div class="citations-section".*?</div>', '', content_text, flags=re.DOTALL)
                    content_text = content_text.strip()
                
                if content_text:
                    if role == 'user':
                        contents.append(types.UserContent(parts=[types.Part.from_text(text=content_text)]))
                    elif role == 'assistant':
                        contents.append(types.ModelContent(parts=[types.Part.from_text(text=content_text)]))
        
        # Add current prompt
        if not contents:
            # If no history, just use prompt as string (will be converted to UserContent automatically)
            contents = request.prompt
        else:
            # Add current user message
            contents.append(types.UserContent(parts=[types.Part.from_text(text=request.prompt)]))
        
        # Configure File Search tool
        # Note: Gemini models that support File Search: gemini-2.5-pro, gemini-2.5-flash
        model = request.model if request.model.startswith("gemini") else "gemini-2.5-flash"
        
        # Create tool configuration with File Search
        tools_config = [
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_search_store_name],
                )
            )
        ]
        
        # Generate content with streaming
        try:
            response_stream = await self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=tools_config,
                )
            )
            
            # Collect citations from all chunks
            citations_map = {}
            chunk_citations_map = {}  # Map for in-text citations
            full_text = ""
            all_chunks = []  # Store all chunks to check final one
            
            # Stream the response and collect all chunks
            async for chunk in response_stream:
                all_chunks.append(chunk)  # Store chunk
                
                # Extract text from chunk
                if hasattr(chunk, 'text') and chunk.text:
                    full_text += chunk.text
                    # Send text delta
                    event_data = {
                        "type": "response.output_text.delta",
                        "data": {
                            "type": "response.output_text.delta",
                            "delta": chunk.text
                        }
                    }
                    yield f"data: {json.dumps(event_data)}\n\n".encode()
            
            # After streaming completes, extract citations from all chunks
            # Check all chunks for grounding_metadata - it's usually in the last chunk
            logger.info(f"Checking {len(all_chunks)} chunks for citations")
            for idx, chunk in enumerate(reversed(all_chunks)):  # Check from last to first
                # Debug: log chunk structure
                if idx == 0:  # Only log first (last) chunk
                    logger.info(f"Last chunk type: {type(chunk)}, has candidates: {hasattr(chunk, 'candidates')}")
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        logger.info(f"Last chunk has {len(chunk.candidates)} candidates")
                        for i, cand in enumerate(chunk.candidates):
                            logger.info(f"Candidate {i} has grounding_metadata: {hasattr(cand, 'grounding_metadata')}")
                            if hasattr(cand, 'grounding_metadata'):
                                gm = cand.grounding_metadata
                                logger.info(f"Grounding metadata attributes: {[x for x in dir(gm) if not x.startswith('_')]}")
                
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'grounding_metadata'):
                            grounding = candidate.grounding_metadata
                            logger.info(f"Found grounding_metadata, has grounding_supports: {hasattr(grounding, 'grounding_supports')}")
                            
                            # Check retrieval_metadata for document information
                            if hasattr(grounding, 'retrieval_metadata'):
                                retrieval = grounding.retrieval_metadata
                                logger.info(f"Found retrieval_metadata, type: {type(retrieval)}, dir: {[x for x in dir(retrieval) if not x.startswith('_')]}")
                                if hasattr(retrieval, 'chunks') and retrieval.chunks:
                                    logger.info(f"Retrieval metadata has {len(retrieval.chunks)} chunks")
                                    for i, ret_chunk in enumerate(retrieval.chunks):
                                        logger.info(f"Retrieval chunk {i} type: {type(ret_chunk)}, dir: {[x for x in dir(ret_chunk) if not x.startswith('_')][:10]}")
                                        if hasattr(ret_chunk, 'chunk'):
                                            logger.info(f"Retrieval chunk {i} chunk: {ret_chunk.chunk}")
                            
                            # Extract chunk citations for in-text citations and build index map
                            # First, build a map of chunk indices to chunk names from grounding_chunks
                            chunk_index_to_name = {}
                            chunk_index_to_info = {}  # Store full info for each chunk
                            
                            if hasattr(grounding, 'grounding_chunks') and grounding.grounding_chunks:
                                logger.info(f"Found {len(grounding.grounding_chunks)} grounding_chunks")
                                for idx, gc in enumerate(grounding.grounding_chunks):
                                    logger.info(f"GroundingChunk {idx} type: {type(gc)}, dir: {[x for x in dir(gc) if not x.startswith('_')]}")
                                    
                                    chunk_name = None
                                    chunk_index = None
                                    
                                    # Try different ways to get chunk name and index
                                    # Check all possible attributes
                                    if hasattr(gc, 'chunk'):
                                        chunk_name = gc.chunk
                                    elif hasattr(gc, 'chunk_name'):
                                        chunk_name = gc.chunk_name
                                    elif hasattr(gc, 'name'):
                                        chunk_name = gc.name
                                    
                                    if hasattr(gc, 'chunk_index'):
                                        chunk_index = gc.chunk_index
                                    elif hasattr(gc, 'index'):
                                        chunk_index = gc.index
                                    
                                    # Check retrieved_context - this might contain chunk information
                                    if hasattr(gc, 'retrieved_context'):
                                        retrieved_context = gc.retrieved_context
                                        logger.info(f"GroundingChunk {idx} retrieved_context type: {type(retrieved_context)}, dir: {[x for x in dir(retrieved_context) if not x.startswith('_')] if hasattr(retrieved_context, '__dict__') else 'N/A'}")
                                        if retrieved_context:
                                            # Try to get chunk name from retrieved_context
                                            if hasattr(retrieved_context, 'chunk'):
                                                chunk_name = retrieved_context.chunk
                                            elif hasattr(retrieved_context, 'chunk_name'):
                                                chunk_name = retrieved_context.chunk_name
                                            elif hasattr(retrieved_context, 'name'):
                                                chunk_name = retrieved_context.name
                                            elif isinstance(retrieved_context, dict):
                                                chunk_name = retrieved_context.get('chunk') or retrieved_context.get('chunk_name') or retrieved_context.get('name')
                                    
                                    # Check maps if available
                                    if not chunk_name and hasattr(gc, 'maps'):
                                        maps = gc.maps
                                        if maps:
                                            logger.info(f"GroundingChunk {idx} maps: {maps}")
                                            # Maps might contain chunk information
                                            if isinstance(maps, dict):
                                                chunk_name = maps.get('chunk') or maps.get('chunk_name')
                                                chunk_index = maps.get('chunk_index') or maps.get('index')
                                    
                                    # If chunk_index is None but we're iterating, use index as chunk_index
                                    if chunk_index is None:
                                        chunk_index = idx
                                    
                                    logger.info(f"GroundingChunk {idx}: chunk_name={chunk_name}, chunk_index={chunk_index}")
                                    
                                    # If we have chunk_name, process it
                                    if chunk_name and '/chunks/' in chunk_name:
                                        doc_name = chunk_name.split('/chunks/')[0].replace('documents/', '')
                                        chunk_id = chunk_name.split('/chunks/')[-1] if '/chunks/' in chunk_name else None
                                        
                                        chunk_index_to_name[chunk_index] = chunk_name
                                        chunk_index_to_info[chunk_index] = {
                                            'document': doc_name,
                                            'display_name': doc_name.split('/')[-1] if '/' in doc_name else doc_name,
                                            'chunk_id': chunk_id,
                                            'chunk_name': chunk_name
                                        }
                                        chunk_citations_map[chunk_index] = chunk_index_to_info[chunk_index]
                                    elif chunk_index is not None:
                                        # Even without chunk_name, store the index for later
                                        logger.warning(f"GroundingChunk {idx} has index {chunk_index} but no chunk_name")
                            else:
                                logger.warning("No grounding_chunks found in grounding_metadata")
                            
                            # Extract grounding_supports for references - this is the main source
                            # Note: The attribute is called 'grounding_supports', not 'chunk_support'
                            # GroundingSupport has 'grounding_chunk_indices' and 'confidence_scores'
                            # We need to use grounding_chunks to get chunk names from indices
                            
                            if hasattr(grounding, 'grounding_supports') and grounding.grounding_supports:
                                logger.info(f"Found {len(grounding.grounding_supports)} grounding_supports items")
                                logger.info(f"Chunk index map has {len(chunk_index_to_name)} entries")
                                
                                for idx, support in enumerate(grounding.grounding_supports):
                                    # Get chunk indices from support
                                    chunk_indices = []
                                    if hasattr(support, 'grounding_chunk_indices'):
                                        chunk_indices = support.grounding_chunk_indices or []
                                    
                                    # Get confidence scores
                                    confidence_scores = []
                                    if hasattr(support, 'confidence_scores'):
                                        confidence_scores = support.confidence_scores or []
                                    
                                    logger.info(f"Support {idx} has {len(chunk_indices)} chunk indices, {len(confidence_scores)} confidence scores")
                                    
                                    # Process each chunk index
                                    for i, chunk_index in enumerate(chunk_indices):
                                        chunk_name = chunk_index_to_name.get(chunk_index)
                                        
                                        # If we don't have chunk_name from grounding_chunks, try to get it from retrieval_metadata
                                        if not chunk_name and hasattr(grounding, 'retrieval_metadata'):
                                            retrieval = grounding.retrieval_metadata
                                            if hasattr(retrieval, 'chunks') and retrieval.chunks:
                                                if chunk_index < len(retrieval.chunks):
                                                    retrieved_chunk = retrieval.chunks[chunk_index]
                                                    if hasattr(retrieved_chunk, 'chunk'):
                                                        chunk_name = retrieved_chunk.chunk
                                                        logger.info(f"Got chunk_name from retrieval_metadata: {chunk_name}")
                                        
                                        if chunk_name and '/chunks/' in chunk_name:
                                            doc_name = chunk_name.split('/chunks/')[0].replace('documents/', '')
                                            if doc_name not in citations_map:
                                                citations_map[doc_name] = {
                                                    'chunks': [],
                                                    'confidence': 0.0,
                                                    'display_name': doc_name.split('/')[-1] if '/' in doc_name else doc_name
                                                }
                                            
                                            # Get confidence for this chunk (use first score if available, or 0.0)
                                            confidence = confidence_scores[i] if i < len(confidence_scores) else 0.0
                                            
                                            # Store chunk info with chunk_name for later retrieval
                                            chunk_info = {
                                                'chunk_id': chunk_name.split('/chunks/')[-1] if '/chunks/' in chunk_name else None,
                                                'chunk_name': chunk_name,
                                                'confidence': confidence
                                            }
                                            
                                            # Also store chunk index info if available
                                            if chunk_index in chunk_index_to_info:
                                                chunk_info.update(chunk_index_to_info[chunk_index])
                                            
                                            citations_map[doc_name]['chunks'].append(chunk_info)
                                            if confidence > citations_map[doc_name]['confidence']:
                                                citations_map[doc_name]['confidence'] = confidence
                                            
                                            logger.info(f"Added citation: doc={doc_name}, chunk={chunk_name}, confidence={confidence}")
                                        elif chunk_index is not None:
                                            logger.warning(f"Chunk index {chunk_index} not found in chunk_index_to_name map, trying alternative methods...")
                                            
                                            # Last resort: try to infer document from grounding_supports or other metadata
                                            # This is a fallback - we'll add a generic citation
                                            if i == 0:  # Only log once per support
                                                logger.warning(f"Could not extract chunk_name for indices: {chunk_indices}")
                
                # Also check chunk directly for grounding_metadata (some SDKs expose it differently)
                if hasattr(chunk, 'grounding_metadata'):
                    grounding = chunk.grounding_metadata
                    if hasattr(grounding, 'grounding_supports') and grounding.grounding_supports:
                        for support in grounding.grounding_supports:
                            if hasattr(support, 'chunk'):
                                chunk_name = support.chunk
                                if '/chunks/' in chunk_name:
                                    doc_name = chunk_name.split('/chunks/')[0].replace('documents/', '')
                                    if doc_name not in citations_map:
                                        citations_map[doc_name] = {
                                            'chunks': [],
                                            'confidence': 0.0,
                                            'display_name': doc_name.split('/')[-1] if '/' in doc_name else doc_name
                                        }
                                    confidence = getattr(support, 'confidence_score', 0.0) if hasattr(support, 'confidence_score') else 0.0
                                    citations_map[doc_name]['chunks'].append({
                                        'chunk_id': chunk_name.split('/chunks/')[-1] if '/chunks/' in chunk_name else None,
                                        'confidence': confidence
                                    })
                                    if confidence > citations_map[doc_name]['confidence']:
                                        citations_map[doc_name]['confidence'] = confidence
            
            logger.info(f"Extracted citations: {len(citations_map)} documents, chunk_citations: {len(chunk_citations_map)}")
            
            # Send chunk citations map for in-text citations
            if chunk_citations_map:
                chunk_citations_event = {
                    "type": "response.chunk_citations",
                    "data": {
                        "type": "response.chunk_citations",
                        "chunk_citations": chunk_citations_map
                    }
                }
                yield f"data: {json.dumps(chunk_citations_event)}\n\n".encode()
            
            # Send citations if any
            if citations_map:
                citations_list = []
                for doc_name, info in citations_map.items():
                    # Include chunk names for fetching content
                    chunk_names = [chunk.get('chunk_name') for chunk in info['chunks'] if chunk.get('chunk_name')]
                    citations_list.append({
                        'document': doc_name,
                        'display_name': info.get('display_name', doc_name.split('/')[-1] if '/' in doc_name else doc_name),
                        'confidence': info['confidence'],
                        'chunks_count': len(info['chunks']),
                        'chunk_names': chunk_names[:5]  # Limit to first 5 chunks for preview
                    })
                
                citations_event = {
                    "type": "response.citations",
                    "data": {
                        "type": "response.citations",
                        "citations": citations_list
                    }
                }
                yield f"data: {json.dumps(citations_event)}\n\n".encode()
            
            # Send final done event
            done_event = {
                "type": "response.done",
                "data": {
                    "type": "response.done"
                }
            }
            yield f"data: {json.dumps(done_event)}\n\n".encode()
            
        except Exception as e:
            # Send error event
            error_event = {
                "type": "error",
                "data": {
                    "type": "error",
                    "error": str(e)
                }
            }
            yield f"data: {json.dumps(error_event)}\n\n".encode()
    
    async def generate(self, request: GenerateChatRequest) -> str:
        """
        Generate a single chat response (non-streaming) using Gemini with File Search.
        
        Args:
            request: Chat request with prompt, document_ids, model, and session_id
            
        Returns:
            Complete response text
        """
        # Get or create File Search store
        store = await self._get_or_create_file_search_store()
        
        # Verify store still exists (in case it was deleted)
        try:
            store = await self.client.file_search_stores.get(name=store.name)
        except Exception:
            # Store was deleted, clear cache and recreate
            self._file_search_store = None
            store = await self._get_or_create_file_search_store()
        
        file_search_store_name = store.name
        
        # Build the prompt
        contents = request.prompt
        
        # Configure File Search tool
        model = request.model if request.model.startswith("gemini") else "gemini-2.5-flash"
        
        tools_config = [
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_search_store_name],
                )
            )
        ]
        
        # Generate content
        response = await self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=tools_config,
            )
        )
        
        # Extract text from response (response has .text attribute directly)
        result_text = ""
        if hasattr(response, 'text') and response.text:
            result_text = response.text
        
        # Extract citations from grounding_metadata
        citations_list = []
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'grounding_metadata'):
                    grounding = candidate.grounding_metadata
                    if hasattr(grounding, 'grounding_supports') and grounding.grounding_supports:
                        citations_map = {}
                        for support in grounding.grounding_supports:
                            if hasattr(support, 'chunk') and hasattr(support, 'confidence_score'):
                                chunk_name = support.chunk
                                # Extract document name from chunk
                                if '/chunks/' in chunk_name:
                                    doc_name = chunk_name.split('/chunks/')[0].replace('documents/', '')
                                    if doc_name not in citations_map:
                                        citations_map[doc_name] = {
                                            'chunks': [],
                                            'confidence': 0.0
                                        }
                                    citations_map[doc_name]['chunks'].append({
                                        'chunk_id': chunk_name.split('/chunks/')[-1] if '/chunks/' in chunk_name else None,
                                        'confidence': getattr(support, 'confidence_score', 0.0)
                                    })
                                    if support.confidence_score > citations_map[doc_name]['confidence']:
                                        citations_map[doc_name]['confidence'] = support.confidence_score
                        
                        # Format citations
                        for doc_name, info in citations_map.items():
                            citations_list.append({
                                'document': doc_name,
                                'display_name': doc_name.split('/')[-1] if '/' in doc_name else doc_name,
                                'confidence': info['confidence'],
                                'chunks_count': len(info['chunks'])
                            })
        
        # Append citations to text if any
        if citations_list:
            result_text += "\n\n**References:**\n"
            for i, citation in enumerate(citations_list, 1):
                result_text += f"{i}. {citation['display_name']} (confidence: {citation['confidence']:.2f})\n"
        
        return result_text

