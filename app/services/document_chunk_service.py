import asyncio
import logging
import re
from typing import List
from pathlib import Path

from docling_core.types.doc.base import ImageRefMode
from app.entities.documents import DocumentChunk
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from app.repositories.documents import DocumentChunksRepository, DocumentsRepository
from docling.document_converter import DocumentConverter
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.transforms.chunker.hierarchical_chunker import TripletTableSerializer
from openai import AsyncOpenAI
from sqlalchemy import func, text

from app.utils.enums import DocumentStatus
from app.repositories.uow import UnitOfWork

logger = logging.getLogger("app.services.document_chunk_service")

class DocumentChunkService:
    """Сервис для работы с chunks документов"""
    
    # Размер чанка в символах (примерно 500-1000 токенов)
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200  # Перекрытие между чанками для контекста
    
    def __init__(
        self,
        uow: UnitOfWork,
        document_repository: DocumentsRepository,
        chunks_repository: DocumentChunksRepository,
        document_converter: DocumentConverter,
        openai_client: AsyncOpenAI,
    ):
        self.uow = uow
        self.document_repository = document_repository
        self.chunks_repository = chunks_repository
        self.document_converter = document_converter
        self.openai_client = openai_client

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Разделяет текст на чанки с перекрытием.
        Разбивает по предложениям для сохранения контекста.
        """
        if not text or not text.strip():
            return []
        
        # Очищаем текст от лишних пробелов
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Разбиваем по предложениям (точка, восклицательный, вопросительный знак)
        sentences = re.split(r'([.!?]\s+)', text)
        
        # Объединяем предложения обратно с разделителями
        sentences_with_delimiters = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentences_with_delimiters.append(sentences[i] + sentences[i + 1])
            else:
                sentences_with_delimiters.append(sentences[i])
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences_with_delimiters:
            # Если добавление предложения не превышает размер чанка
            if len(current_chunk) + len(sentence) <= self.CHUNK_SIZE:
                current_chunk += sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Начинаем новый чанк с перекрытием
                if chunks and self.CHUNK_OVERLAP > 0:
                    # Берем последние N символов предыдущего чанка для контекста
                    overlap_text = chunks[-1][-self.CHUNK_OVERLAP:]
                    current_chunk = overlap_text + sentence
                else:
                    current_chunk = sentence
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    async def _create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Создает embeddings для списка текстов (батчинг для эффективности)"""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
                dimensions=1024
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise AppError(
                status_code=500,
                message=f"Failed to create embeddings: {str(e)}"
            )

    async def execute(self, document_id: int):
        from app.entities.documents import Document
        
        logger.info("DocumentChunkService: starting processing for document_id=%d", document_id)
        
        document = await self.document_repository.get_one(
            where=[Document.id == document_id]
        )
        if not document:
            raise AppError(status_code=404, message=ErrorMessages.DOCUMENT_NOT_FOUND)
        
        document.status = DocumentStatus.PROCESSING
        try:
            file_path = document.file_path
            logger.info("DocumentChunkService: converting document from file_path=%s", file_path)

            # 1) Конвертируем документ через Docling
            loop = asyncio.get_running_loop()
            dl_doc = await loop.run_in_executor(
                None,
                lambda: self.document_converter.convert(source=file_path).document
            )
            if not dl_doc:
                raise AppError(status_code=500, message=ErrorMessages.DOCUMENT_CONVERSION_FAILED)
            
            logger.info("DocumentChunkService: document converted successfully")
            
            # 2) Сериализуем в Markdown + триплеты
            serializer = MarkdownDocSerializer(
                doc=dl_doc,
                table_serializer=TripletTableSerializer(),
                params=MarkdownParams(
                    image_mode=ImageRefMode.PLACEHOLDER,
                    image_placeholder="",
                ),
            )
            md_text = serializer.serialize().text
            logger.info("DocumentChunkService: document serialized to markdown, length=%d chars", len(md_text))
            
            # Сохраняем полный контент в документ
            document.content = md_text
            
            # 3) Разделяем на чанки
            chunks_text = self._split_text_into_chunks(md_text)
            logger.info("DocumentChunkService: document split into %d chunks", len(chunks_text))
            
            if not chunks_text:
                raise AppError(status_code=500, message="Failed to split document into chunks")
            
            # 4) Создаем embeddings для всех чанков (батчинг)
            logger.info("DocumentChunkService: creating embeddings for %d chunks", len(chunks_text))
            embeddings = await self._create_embeddings_batch(chunks_text)
            logger.info("DocumentChunkService: embeddings created successfully")
            
            # 5) Создаем записи DocumentChunk
            chunk_entities = []
            for i, (chunk_text, embedding_vector) in enumerate(zip(chunks_text, embeddings)):
                # Создаем чанк - content_tsv будет обновлен через SQL запрос ниже
                # Используем пустую строку, PostgreSQL преобразует её в пустой tsvector
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    content=chunk_text,
                    embedding=embedding_vector,
                )
                chunk_entities.append(chunk)
            
            # 6) Сохраняем чанки в базу через bulk_create
            logger.info("DocumentChunkService: creating %d chunks for document_id=%d", len(chunk_entities), document_id)
            await self.chunks_repository.bulk_create(chunk_entities)
            
            # 7) Явно обновляем content_tsv для всех созданных chunks
            # PostgreSQL computed колонки могут не обновляться автоматически при bulk insert
            logger.info("DocumentChunkService: updating content_tsv for all chunks")
            try:
                update_tsv_query = text("""
                    UPDATE document_chunks 
                    SET content_tsv = to_tsvector('simple', unaccent(content))
                    WHERE document_id = :document_id
                """)
                result = await self.chunks_repository.session.execute(
                    update_tsv_query, 
                    {"document_id": document_id}
                )
                updated_count = result.rowcount
                logger.info("DocumentChunkService: updated content_tsv for %d chunks", updated_count)
            except Exception as e:
                logger.error(
                    "DocumentChunkService: error updating content_tsv: %s. "
                    "Make sure 'unaccent' extension is installed: CREATE EXTENSION IF NOT EXISTS unaccent;",
                    str(e),
                    exc_info=True
                )
                # Пробуем без unaccent как fallback
                try:
                    update_tsv_fallback_query = text("""
                        UPDATE document_chunks 
                        SET content_tsv = to_tsvector('simple', content)
                        WHERE document_id = :document_id
                    """)
                    result = await self.chunks_repository.session.execute(
                        update_tsv_fallback_query,
                        {"document_id": document_id}
                    )
                    logger.info("DocumentChunkService: updated content_tsv (without unaccent) for %d chunks", result.rowcount)
                except Exception as fallback_error:
                    logger.error("DocumentChunkService: fallback update also failed: %s", str(fallback_error))
                    raise
            
            # Проверяем что content_tsv заполнен
            check_tsv_query = text("""
                SELECT COUNT(*) 
                FROM document_chunks 
                WHERE document_id = :document_id 
                  AND (content_tsv IS NULL OR content_tsv = ''::tsvector)
            """)
            check_result = await self.chunks_repository.session.execute(
                check_tsv_query,
                {"document_id": document_id}
            )
            empty_tsv_count = check_result.scalar() or 0
            if empty_tsv_count > 0:
                logger.warning(
                    "DocumentChunkService: %d chunks still have empty content_tsv after update",
                    empty_tsv_count
                )
            else:
                logger.info("DocumentChunkService: all chunks have content_tsv filled")
            
            # 8) Обновляем статус документа
            document.status = DocumentStatus.COMPLETED
            
        except Exception as e:
            logger.error("DocumentChunkService: error processing document_id=%d: %s", document_id, str(e), exc_info=True)
            document.status = DocumentStatus.FAILED
        finally:
            await self.uow.commit()
        
