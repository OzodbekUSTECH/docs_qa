import logging
from typing import List, Tuple

from sqlalchemy import any_
from app.entities.extracion_fields import ExtractionField
from app.entities.documents import Document, DocumentFieldValue
from app.repositories.documents import DocumentFieldValuesRepository
from app.repositories.uow import UnitOfWork
from openai import AsyncOpenAI
from app.repositories.extraction_fields import ExtractionFieldsRepository
from app.repositories.documents import DocumentsRepository
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from app.utils.enums import DocumentStatus
from app.dto.field_extraction import FieldExtractionOutput, ExtractedFieldValue, DocumentMarkdownOutput
from app.services.manual_field_extraction import ManualFieldExtractionService

logger = logging.getLogger("app.services.extract_field_values")


class ExtractDocumentFieldValuesService:
    """Сервис для извлечения значений полей из документов через OpenAI Vision API и ручную экстракцию"""
    
    VISION_MODEL = "gpt-4o-2024-08-06"  # Более быстрая модель с structured outputs
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1024
    
    def __init__(
        self,
        uow: UnitOfWork,
        document_field_values_repository: DocumentFieldValuesRepository,
        openai_client: AsyncOpenAI,
        extraction_fields_repository: ExtractionFieldsRepository,
        document_repository: DocumentsRepository,
        manual_extraction_service: ManualFieldExtractionService,
    ):
        self.uow = uow
        self.document_field_values_repository = document_field_values_repository
        self.openai_client = openai_client
        self.extraction_fields_repository = extraction_fields_repository
        self.document_repository = document_repository
        self.manual_extraction_service = manual_extraction_service

    async def _upload_file_to_openai(self, file_path: str) -> str:
        """Загружает файл в OpenAI через Files API и возвращает file_id"""
        try:
            with open(file_path, "rb") as file_content:
                file = await self.openai_client.files.create(
                    file=file_content,
                    purpose="user_data",  # Используем user_data для поддержки PDF и других форматов
                )
                logger.info("ExtractDocumentFieldValuesService: file uploaded to OpenAI, file_id=%s, status=%s", file.id, getattr(file, 'status', 'unknown'))
                return file.id
        except Exception as e:
            raise AppError(
                status_code=500,
                message=f"Failed to upload file to OpenAI: {str(e)}"
            )
    
    async def _wait_for_file_ready(self, file_id: str, max_wait_seconds: int = 60) -> bool:
        """Ожидает готовности файла в OpenAI (для больших файлов может потребоваться время)"""
        import asyncio
        import time
        start_time = time.time()
        
        while True:
            try:
                file = await self.openai_client.files.retrieve(file_id)
                status = getattr(file, 'status', 'unknown')
                
                if status == 'processed':
                    logger.info("ExtractDocumentFieldValuesService: file %s is ready (status: %s)", file_id, status)
                    return True
                elif status == 'error':
                    logger.error("ExtractDocumentFieldValuesService: file %s processing failed (status: %s)", file_id, status)
                    return False
                elif status in ('pending', 'uploading'):
                    elapsed = time.time() - start_time
                    if elapsed > max_wait_seconds:
                        logger.warning("ExtractDocumentFieldValuesService: file %s still processing after %d seconds, proceeding anyway", file_id, max_wait_seconds)
                        return True  # Продолжаем, возможно файл готов, но статус не обновился
                    logger.info("ExtractDocumentFieldValuesService: file %s status: %s, waiting... (elapsed: %.1fs)", file_id, status, elapsed)
                    await asyncio.sleep(2)
                else:
                    # Неизвестный статус, продолжаем
                    logger.info("ExtractDocumentFieldValuesService: file %s status: %s, proceeding", file_id, status)
                    return True
            except Exception as e:
                logger.warning("ExtractDocumentFieldValuesService: error checking file status: %s, proceeding anyway", str(e))
                return True  # Продолжаем, если не можем проверить статус

    async def _delete_openai_file(self, file_id: str):
        """Удаляет файл из OpenAI после обработки"""
        try:
            await self.openai_client.files.delete(file_id)
            logger.info("ExtractDocumentFieldValuesService: file deleted from OpenAI, file_id=%s", file_id)
        except Exception as e:
            logger.warning("ExtractDocumentFieldValuesService: failed to delete OpenAI file %s: %s", file_id, str(e))

    def _build_extraction_prompt(self, extraction_fields: List[ExtractionField]) -> str:
        """Строит промпт для извлечения полей на основе конфигурации полей"""
        prompt_parts = [
            "Extract the following fields from the document:",
            ""
        ]
        
        for field in extraction_fields:
            prompt_parts.append(f"Field ID {field.id} - '{field.name}':")
            if field.short_description:
                prompt_parts.append(f"  Description: {field.short_description}")
            prompt_parts.append(f"  Type: {field.type.value}")
            if field.prompt:
                prompt_parts.append(f"  Instructions: {field.prompt}")
            
            if field.examples:
                prompt_parts.append("  Examples:")
                for example in field.examples:
                    prompt_parts.append(f"    - {example}")
            
            prompt_parts.append("")
        
        prompt_parts.extend([
            "CRITICAL REQUIREMENTS:",
            "",
            "1. FIELD ID (field_id):",
            "   - Use the exact numeric field_id shown above for each field",
            "   - This is the unique identifier for the field (e.g., 1, 2, 3, etc.)",
            "",
            "2. PAGE NUMBER (page_num):",
            "   - Must be an integer >= 1",
            "   - 1 = first page of the document, 2 = second page, etc.",
            "   - For single-page documents, always use 1",
            "   - Count pages from the beginning of the document",
            "",
            "3. BOUNDING BOX (bbox):",
            "   - Format: [x1, y1, x2, y2] - exactly 4 float values",
            "   - All coordinates MUST be normalized (0.0 to 1.0) relative to the PAGE dimensions",
            "   - x1, y1 = top-left corner of the text region (x1 < x2, y1 < y2)",
            "   - x2, y2 = bottom-right corner of the text region",
            "   - Coordinate system: (0,0) is top-left, (1,1) is bottom-right of the page",
            "   - The bbox should tightly enclose the actual text that contains the extracted value",
            "   - Example: [0.1, 0.2, 0.5, 0.25] means text spans from 10% to 50% horizontally and 20% to 25% vertically",
            "",
            "4. EXTRACTION:",
            "   - Extract values based on field instructions and examples",
            "   - Provide confidence (0.0-1.0) for each extraction",
            "   - If field not found, omit it or set confidence to 0.0",
            "",
            "Return only the extracted field values in structured format with correct field_id for each value."
        ])
        
        return "\n".join(prompt_parts)

    async def _extract_fields_with_openai(
        self, 
        markdown_content: str, 
        extraction_fields: List[ExtractionField],
        timeout_seconds: int = 300
    ) -> FieldExtractionOutput:
        """Извлекает значения полей через OpenAI API с structured output, используя markdown контент"""
        import asyncio
        import time
        
        system_prompt = self._build_extraction_prompt(extraction_fields)
        
        user_prompt = f"Extract the specified field values from this document:\n\n{markdown_content}"

        try:
            logger.info(
                "ExtractDocumentFieldValuesService: extracting %d fields with AI (timeout=%ds, content_length=%d chars)",
                len(extraction_fields),
                timeout_seconds,
                len(markdown_content)
            )
            start_time = time.time()
            
            try:
                response = await asyncio.wait_for(
                    self.openai_client.responses.parse(
                        model=self.VISION_MODEL,
                        input=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        text_format=FieldExtractionOutput,
                    ),
                    timeout=timeout_seconds
                )
                
                elapsed = time.time() - start_time
                logger.info("ExtractDocumentFieldValuesService: received response from OpenAI after %.1f seconds", elapsed)
                
                result = response.output_parsed
                logger.info(
                    "ExtractDocumentFieldValuesService: extracted %d field values (total time: %.1fs)",
                    len(result.extracted_values),
                    time.time() - start_time
                )
                return result
                
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.error(
                    "ExtractDocumentFieldValuesService: OpenAI field extraction timed out after %.1f seconds (timeout=%ds)",
                    elapsed,
                    timeout_seconds
                )
                raise AppError(
                    status_code=500,
                    message=f"OpenAI field extraction timed out after {timeout_seconds} seconds. The document may be too large."
                )
                
        except AppError:
            raise
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(
                "ExtractDocumentFieldValuesService: error extracting fields after %.1fs: %s",
                elapsed,
                str(e),
                exc_info=True
            )
            raise AppError(
                status_code=500,
                message=f"Failed to extract fields with OpenAI: {str(e)}"
            )

    async def _get_document_markdown(self, file_id: str, timeout_seconds: int = 300) -> str:
        """
        Получает полный markdown текст документа через GPT с использованием structured output.
        Используется для ручной экстракции полей.
        
        Args:
            file_id: ID файла в OpenAI
            timeout_seconds: Максимальное время ожидания ответа (по умолчанию 5 минут)
        """
        import asyncio
        import time
        
        try:
            logger.info("ExtractDocumentFieldValuesService: requesting markdown text from document (file_id=%s, timeout=%ds)", file_id, timeout_seconds)
            
            # Улучшенный промпт для OCR - более детальные инструкции
            system_prompt = (
                "You are an expert OCR and document extraction system. Your task is to extract ALL text content "
                "from the document with maximum accuracy.\n\n"
                
                "CRITICAL OCR REQUIREMENTS:\n"
                "1. For scanned documents or images, carefully read EVERY character, number, and symbol\n"
                "2. Pay special attention to:\n"
                "   - Numbers (prices, dates, quantities, measurements)\n"
                "   - Technical terms and abbreviations\n"
                "   - Table data (preserve exact values, do not approximate)\n"
                "   - Headers and section titles\n"
                "   - Legal clauses and contract terms\n"
                "3. If text is unclear, make your best interpretation but preserve the original structure\n"
                "4. Do NOT skip any text, even if it seems redundant\n"
                "5. Preserve exact spacing and line breaks where they are meaningful\n\n"
                
                "DOCUMENT STRUCTURE PRESERVATION:\n"
                "1. Maintain original document hierarchy:\n"
                "   - Use markdown headers (# ## ###) for section titles\n"
                "   - Preserve numbered clauses (e.g., '7. PRICE', '9. PAYMENT')\n"
                "   - Keep attachment headers (e.g., 'Attachment 1 – ULSD French Spec')\n"
                "2. Tables:\n"
                "   - Convert to proper markdown table format with pipes (|)\n"
                "   - Preserve all table headers and data cells\n"
                "   - Maintain column alignment\n"
                "   - Include separator row (|---|---|)\n"
                "3. Lists:\n"
                "   - Preserve bullet points and numbered lists\n"
                "   - Maintain indentation levels\n"
                "4. Formatting:\n"
                "   - Use **bold** for emphasized text\n"
                "   - Preserve line breaks between paragraphs\n"
                "   - Keep special characters and symbols as they appear\n\n"
                
                "OUTPUT FORMAT:\n"
                "- Return ONLY the markdown text content\n"
                "- No explanations, no comments, no metadata\n"
                "- Start directly with the document content\n"
                "- Ensure the markdown is valid and properly formatted"
            )
            
            user_prompt = (
                "Extract ALL text content from this document as markdown.\n\n"
                "If this is a scanned document or image:\n"
                "- Perform careful OCR to read every character accurately\n"
                "- Pay special attention to numbers, technical terms, and table data\n"
                "- Preserve the exact document structure including headers, tables, and formatting\n"
                "- Do not skip any text, even if it appears unclear\n\n"
                "Return the complete markdown representation of the document content."
            )
            
            logger.info("ExtractDocumentFieldValuesService: sending request to OpenAI for markdown extraction")
            start_time = time.time()
            
            # Обертываем запрос в таймаут
            try:
                response = await asyncio.wait_for(
                    self.openai_client.responses.parse(
                        model=self.VISION_MODEL,
                        input=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_file",
                                        "file_id": file_id,
                                    },
                                    {
                                        "type": "input_text",
                                        "text": user_prompt,
                                    },
                                ],
                            },
                        ],
                        text_format=DocumentMarkdownOutput,
                    ),
                    timeout=timeout_seconds
                )
                
                elapsed = time.time() - start_time
                logger.info("ExtractDocumentFieldValuesService: received response from OpenAI after %.1f seconds, parsing...", elapsed)
                
                result = response.output_parsed
                markdown_text = result.markdown_text
                
                logger.info(
                    "ExtractDocumentFieldValuesService: markdown extraction completed, length=%d characters (total time: %.1fs)",
                    len(markdown_text),
                    time.time() - start_time
                )
                return markdown_text
                
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.error(
                    "ExtractDocumentFieldValuesService: OpenAI request timed out after %.1f seconds (timeout=%ds)",
                    elapsed,
                    timeout_seconds
                )
                raise AppError(
                    status_code=500,
                    message=f"OpenAI request timed out after {timeout_seconds} seconds. The document may be too large or complex."
                )
                
        except AppError:
            raise
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(
                "ExtractDocumentFieldValuesService: error getting markdown after %.1fs: %s",
                elapsed,
                str(e),
                exc_info=True
            )
            raise AppError(
                status_code=500,
                message=f"Failed to get document markdown: {str(e)}"
            )

    async def _create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Создает embeddings для списка текстов (батчинг для эффективности)"""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=texts,
                dimensions=self.EMBEDDING_DIMENSIONS
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise AppError(
                status_code=500,
                message=f"Failed to create embeddings: {str(e)}"
            )
    
    def _split_fields_by_extraction_method(
        self, 
        extraction_fields: List[ExtractionField]
    ) -> Tuple[List[ExtractionField], List[ExtractionField]]:
        """
        Разделяет поля на те, что извлекаются через AI и те, что извлекаются вручную.
        
        Returns:
            Tuple[ai_fields, manual_fields]
        """
        ai_fields = []
        manual_fields = []
        
        for field in extraction_fields:
            if field.use_ai:
                ai_fields.append(field)
            else:
                manual_fields.append(field)
        
        logger.info(
            "ExtractDocumentFieldValuesService: split fields - AI: %d, Manual: %d",
            len(ai_fields),
            len(manual_fields)
        )
        
        return ai_fields, manual_fields

    async def execute(self, document_id: int, extraction_field_ids: list[int]):
        """Основной метод извлечения значений полей из документа"""
        logger.info("ExtractDocumentFieldValuesService: starting extraction for document_id=%d", document_id)
        
        document = await self.document_repository.get_one(
            where=[Document.id == document_id]
        )
        if not document:
            raise AppError(status_code=404, message=ErrorMessages.DOCUMENT_NOT_FOUND)
        
        # Обновляем статус на PROCESSING
        document.status = DocumentStatus.PROCESSING
        await self.uow.commit()
        
        # Получаем поля извлечения, подходящие для типа документа
        # Используем оператор @> (contains) для проверки наличия значения в массиве
        extraction_fields = await self.extraction_fields_repository.get_all(
            where=[document.type == any_(ExtractionField.document_types), ExtractionField.id.in_(extraction_field_ids)]
        )
        if not extraction_fields:
            logger.warning("ExtractDocumentFieldValuesService: no extraction fields found for document type %s", document.type)
            document.status = DocumentStatus.COMPLETED
            await self.uow.commit()
            return
        
        # Удаляем ВСЕ старые значения полей перед повторной экстракцией
        # Это необходимо, чтобы итоговый набор значений соответствовал выбранным полям
        from sqlalchemy import delete as sql_delete
        from app.entities.documents import DocumentFieldValue
        
        logger.info(
            "ExtractDocumentFieldValuesService: deleting ALL existing field values for document_id=%d",
            document_id
        )
        delete_stmt = sql_delete(DocumentFieldValue).where(
            DocumentFieldValue.document_id == document_id
        )
        await self.document_field_values_repository.session.execute(delete_stmt)
        await self.uow.commit()
        logger.info("ExtractDocumentFieldValuesService: deleted all existing field values for document_id=%d", document_id)
        
        # Создаем словарь для быстрого поиска полей по ID
        fields_by_id = {field.id: field for field in extraction_fields}
        
        # Разделяем поля на AI и ручные
        ai_fields, manual_fields = self._split_fields_by_extraction_method(extraction_fields)
        
        file_id = None
        try:
            file_path = document.file_path
            logger.info("ExtractDocumentFieldValuesService: processing document from file_path=%s", file_path)

            # 1) Проверяем, есть ли уже сохраненный markdown контент
            if document.content:
                logger.info(
                    "ExtractDocumentFieldValuesService: using existing document.content (length=%d characters), skipping markdown extraction",
                    len(document.content)
                )
                markdown_content = document.content
            else:
                # 1a) Загружаем файл в OpenAI через Files API и извлекаем markdown
                logger.info("ExtractDocumentFieldValuesService: document.content not found, uploading file to OpenAI and extracting markdown")
                file_id = await self._upload_file_to_openai(file_path)
                
                # 1b) Ожидаем готовности файла (для больших файлов может потребоваться время)
                logger.info("ExtractDocumentFieldValuesService: checking if file is ready for processing")
                await self._wait_for_file_ready(file_id)
                
                # 2) Извлекаем markdown текст документа и сохраняем в document.content
                logger.info("ExtractDocumentFieldValuesService: starting markdown extraction")
                markdown_content = await self._get_document_markdown(file_id)
                document.content = markdown_content
                await self.uow.commit()
                logger.info(
                    "ExtractDocumentFieldValuesService: markdown extracted and saved to document.content, length=%d characters",
                    len(markdown_content)
                )
            
            # 3) Извлекаем значения полей используя сохраненный markdown контент
            all_extracted_values: List[ExtractedFieldValue] = []
            
            # 3a) AI экстракция - используем markdown контент вместо file_id
            if ai_fields:
                logger.info("ExtractDocumentFieldValuesService: extracting %d fields with AI using markdown content", len(ai_fields))
                ai_extraction_result = await self._extract_fields_with_openai(markdown_content, ai_fields)
                all_extracted_values.extend(ai_extraction_result.extracted_values)
            
            # 3b) Ручная экстракция - используем сохраненный markdown контент
            if manual_fields:
                logger.info("ExtractDocumentFieldValuesService: extracting %d fields manually using markdown content", len(manual_fields))
                manual_extracted_values = self.manual_extraction_service.extract_by_schema(
                    markdown_content,
                    manual_fields
                )
                all_extracted_values.extend(manual_extracted_values)
            
            # 4) Создаем embeddings для всех извлеченных значений (батчинг)
            extracted_texts = [ev.value for ev in all_extracted_values if ev.value]
            if extracted_texts:
                logger.info("ExtractDocumentFieldValuesService: creating embeddings for %d values", len(extracted_texts))
                embeddings = await self._create_embeddings_batch(extracted_texts)
                logger.info("ExtractDocumentFieldValuesService: embeddings created successfully")
            else:
                embeddings = []
                logger.warning("ExtractDocumentFieldValuesService: no values to create embeddings for")
            
            # 5) Создаем записи DocumentFieldValue
            field_value_entities = []
            embedding_idx = 0
            
            for extracted_value in all_extracted_values:
                field = fields_by_id.get(extracted_value.field_id)
                if not field:
                    logger.warning(
                        "ExtractDocumentFieldValuesService: field with id %d not found, skipping",
                        extracted_value.field_id
                    )
                    continue
                
                # Создаем embedding только если есть значение
                embedding_vector = None
                if extracted_value.value and embedding_idx < len(embeddings):
                    embedding_vector = embeddings[embedding_idx]
                    embedding_idx += 1
                
                field_value = DocumentFieldValue(
                    document_id=document_id,
                    field_id=field.id,
                    value_text=extracted_value.value or "",
                    confidence=extracted_value.confidence,
                    page_num=extracted_value.page_num,
                    bbox=extracted_value.bbox,
                    embedding=embedding_vector,
                )
                field_value_entities.append(field_value)
            
            # 6) Сохраняем значения полей в базу через bulk_create
            if field_value_entities:
                logger.info(
                    "ExtractDocumentFieldValuesService: creating %d field values for document_id=%d",
                    len(field_value_entities),
                    document_id
                )
                await self.document_field_values_repository.bulk_create(field_value_entities)
                
                # 7) Обновляем value_tsv для всех созданных значений
                logger.info("ExtractDocumentFieldValuesService: updating value_tsv for all field values")
                try:
                    from sqlalchemy import text
                    update_tsv_query = text("""
                        UPDATE document_field_values 
                        SET value_tsv = to_tsvector('simple', unaccent(value_text))
                        WHERE document_id = :document_id
                    """)
                    result = await self.document_field_values_repository.session.execute(
                        update_tsv_query, 
                        {"document_id": document_id}
                    )
                    logger.info("ExtractDocumentFieldValuesService: updated value_tsv for %d field values", result.rowcount)
                except Exception as e:
                    logger.warning("ExtractDocumentFieldValuesService: error updating value_tsv with unaccent: %s", str(e))
                    # Fallback без unaccent
                    try:
                        update_tsv_fallback_query = text("""
                            UPDATE document_field_values 
                            SET value_tsv = to_tsvector('simple', value_text)
                            WHERE document_id = :document_id
                        """)
                        result = await self.document_field_values_repository.session.execute(
                            update_tsv_fallback_query,
                            {"document_id": document_id}
                        )
                        logger.info("ExtractDocumentFieldValuesService: updated value_tsv (without unaccent) for %d field values", result.rowcount)
                    except Exception as fallback_error:
                        logger.error("ExtractDocumentFieldValuesService: fallback update also failed: %s", str(fallback_error))
            else:
                logger.warning("ExtractDocumentFieldValuesService: no field values to save")
            
            # 8) Обновляем статус документа
            document.status = DocumentStatus.COMPLETED
            logger.info(
                "ExtractDocumentFieldValuesService: field extraction completed successfully. "
                "Extracted %d field values",
                len(field_value_entities)
            )
            
        except Exception as e:
            logger.error("ExtractDocumentFieldValuesService: error processing document_id=%d: %s", document_id, str(e), exc_info=True)
            document.status = DocumentStatus.FAILED
        finally:
            # Удаляем файл из OpenAI после обработки
            if file_id:
                await self._delete_openai_file(file_id)
            await self.uow.commit()
