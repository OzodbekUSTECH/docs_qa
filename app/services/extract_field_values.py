import logging
from typing import List

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
from app.dto.field_extraction import FieldExtractionOutput

logger = logging.getLogger("app.services.extract_field_values")


class ExtractDocumentFieldValuesService:
    """Сервис для извлечения значений полей из документов через OpenAI Vision API"""
    
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
    ):
        self.uow = uow
        self.document_field_values_repository = document_field_values_repository
        self.openai_client = openai_client
        self.extraction_fields_repository = extraction_fields_repository
        self.document_repository = document_repository

    async def _upload_file_to_openai(self, file_path: str) -> str:
        """Загружает файл в OpenAI через Files API и возвращает file_id"""
        try:
            with open(file_path, "rb") as file_content:
                file = await self.openai_client.files.create(
                    file=file_content,
                    purpose="user_data",  # Используем user_data для поддержки PDF и других форматов
                )
                logger.info("ExtractDocumentFieldValuesService: file uploaded to OpenAI, file_id=%s", file.id)
                return file.id
        except Exception as e:
            raise AppError(
                status_code=500,
                message=f"Failed to upload file to OpenAI: {str(e)}"
            )

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
        file_id: str, 
        extraction_fields: List[ExtractionField]
    ) -> FieldExtractionOutput:
        """Извлекает значения полей через OpenAI Vision API с structured output"""
        system_prompt = self._build_extraction_prompt(extraction_fields)
        
        user_prompt = "Extract the specified field values from this document."

        try:
            response = await self.openai_client.responses.parse(
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
                text_format=FieldExtractionOutput,
            )
            
            result = response.output_parsed
            logger.info(
                "ExtractDocumentFieldValuesService: extracted %d field values",
                len(result.extracted_values)
            )
            return result
        except Exception as e:
            logger.error("ExtractDocumentFieldValuesService: error extracting fields: %s", str(e), exc_info=True)
            raise AppError(
                status_code=500,
                message=f"Failed to extract fields with OpenAI: {str(e)}"
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
        
        # Создаем словарь для быстрого поиска полей по ID
        fields_by_id = {field.id: field for field in extraction_fields}
        
        file_id = None
        try:
            file_path = document.file_path
            logger.info("ExtractDocumentFieldValuesService: processing document from file_path=%s", file_path)

            # 1) Загружаем файл в OpenAI через Files API
            logger.info("ExtractDocumentFieldValuesService: uploading file to OpenAI")
            file_id = await self._upload_file_to_openai(file_path)
            
            # 2) Извлекаем значения полей через OpenAI Vision API с structured output
            logger.info("ExtractDocumentFieldValuesService: extracting fields with OpenAI Vision API")
            extraction_result = await self._extract_fields_with_openai(file_id, extraction_fields)
            
            # 3) Создаем embeddings для всех извлеченных значений (батчинг)
            extracted_texts = [ev.value for ev in extraction_result.extracted_values if ev.value]
            if extracted_texts:
                logger.info("ExtractDocumentFieldValuesService: creating embeddings for %d values", len(extracted_texts))
                embeddings = await self._create_embeddings_batch(extracted_texts)
                logger.info("ExtractDocumentFieldValuesService: embeddings created successfully")
            else:
                embeddings = []
                logger.warning("ExtractDocumentFieldValuesService: no values to create embeddings for")
            
            # 4) Создаем записи DocumentFieldValue
            field_value_entities = []
            embedding_idx = 0
            
            for extracted_value in extraction_result.extracted_values:
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
            
            # 5) Сохраняем значения полей в базу через bulk_create
            if field_value_entities:
                logger.info(
                    "ExtractDocumentFieldValuesService: creating %d field values for document_id=%d",
                    len(field_value_entities),
                    document_id
                )
                await self.document_field_values_repository.bulk_create(field_value_entities)
                
                # 6) Обновляем value_tsv для всех созданных значений
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
            
            # 7) Обновляем статус документа
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
