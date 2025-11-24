import logging
from agents import function_tool
from app.dto.chat import (
    FieldValueSearchResponse,
    DocumentSearchResultResponse,
    MatchedFieldValueResponse,
    OtherFieldValueResponse,
)
from app.utils.enums import DocumentType

logger = logging.getLogger("app.tools.hybrid_search")


# @function_tool

async def hybrid_search(
    query: str,
    document_ids: list[int] | None = None,
    exclude_document_ids: list[int] | None = None,
    field_ids: list[int] | None = None,
    exclude_field_ids: list[int] | None = None,
    document_types: list[str] | None = None,
    min_confidence: float | None = None,
    limit: int = 15,
    text_weight: float = 0.6,
    vector_weight: float = 0.4,
    k_text: int = 64,
    k_vector: int = 64,
) -> dict:
    """
    Выполнить мощный гибридный поиск по извлеченным значениям полей документов.

    Механизм: Независимо выполняет полнотекстовый (FTS) и векторный поиск по document_field_values,
    затем группирует результаты по документам и возвращает документы с найденными полями.

    Аргументы:
        query (str): Пользовательский поисковый запрос.
        document_ids (list[int], опционально): Ограничить поиск определенными документами (по их ID).
        exclude_document_ids (list[int], опционально): Исключить определенные документы (по их ID).
        field_ids (list[int], опционально): Ограничить поиск определенными полями извлечения (по их ID).
        exclude_field_ids (list[int], опционально): Исключить определенные поля извлечения (по их ID).
        document_types (list[str], опционально): Фильтр по типам документов. Доступные типы (регистр не важен):
            - "CONTRACT" - контракты
            - "INVOICE" - счета-фактуры
            - "FINANCIAL" - финансовые документы
            - "COO" - сертификат происхождения (certificate of origin)
            - "COA" - сертификат анализа (certificate of analysis)
            - "COW" - сертификат веса (certificate of weight)
            - "COQ" - сертификат качества (certificate of quality)
            - "BL" - коносамент (bill of lading)
            - "LC" - аккредитив (letter of credit)
            - "OTHER" - другие документы
            Пример: ["CONTRACT", "INVOICE"] или ["contract", "invoice"] (регистр не важен).
        min_confidence (float, опционально): Минимальный уровень уверенности для значений полей (0.0-1.0).
        limit (int, опционально): Максимальное количество возвращаемых документов (по умолчанию 15).
        text_weight (float, опционально): Вес текстового скоринга при ранжировании (по умолчанию 0.6).
        vector_weight (float, опционально): Вес векторного скоринга при ранжировании (по умолчанию 0.4).
        k_text (int, опционально): Сколько top-k кандидатов брать из FTS (по умолчанию 64).
        k_vector (int, опционально): Сколько top-k кандидатов брать из векторного поиска (по умолчанию 64).

    Returns:
        dict: Список документов с найденными полями и дополнительной информацией о других полях документа.
            Структура ответа:
            {
                "documents": [
                    {
                        "document_id": int,
                        "filename": str,
                        "file_path": str,
                        "content_type": str,
                        "document_type": str,
                        "status": str,
                        "summary": str | None,
                        "matched_fields": [
                            {
                                "id": int,
                                "field_id": int,
                                "field_name": str,
                                "value_text": str,
                                "confidence": float | None,
                                "page_num": int | None,
                                "bbox": [float, float, float, float] | None,
                                "text_rank": float,
                                "vector_score": float,
                                "hybrid_score": float
                            }
                        ],
                        "other_fields": [
                            {
                                "id": int,
                                "field_id": int,
                                "field_name": str,
                                "short_description": str | None
                            }
                        ],
                        "max_hybrid_score": float
                    }
                ]
            }
            
            Примечание: 
            - matched_fields содержат полную информацию о найденных полях (значения, координаты, оценки)
            - other_fields содержат только метаданные (id, field_id, name, short_description) для понимания доступных полей
            - Если нужно получить полную информацию о поле из other_fields, используй его id или field_id для запроса
    """
    from app.di.containers import app_container
    from app.repositories.documents import DocumentFieldValuesRepository, DocumentsRepository
    from app.entities.documents import Document, DocumentFieldValue
    from app.entities.extracion_fields import ExtractionField
    from openai import AsyncOpenAI
    from sqlalchemy.orm import selectinload, joinedload

    logger.info(
        "╔═══════════════════════════════════════════════════════════════╗\n"
        "║ HYBRID SEARCH (FIELD VALUES)                                  ║\n"
        "╠═══════════════════════════════════════════════════════════════╣\n"
        "║ Query: %-55s ║\n"
        "║ Document IDs: %-48s ║\n"
        "║ Document Types: %-45s ║\n"
        "║ Field IDs: %-51s ║\n"
        "║ Limit: %-54d ║\n"
        "║ Weights: text=%.2f, vector=%.2f                              ║\n"
        "║ K values: text=%d, vector=%d                                  ║\n"
        "╚═══════════════════════════════════════════════════════════════╝",
        query[:55] if query else "None",
        str(document_ids)[:48] if document_ids else "None",
        str(document_types)[:45] if document_types else "None",
        str(field_ids)[:51] if field_ids else "None",
        limit,
        text_weight,
        vector_weight,
        k_text,
        k_vector,
    )

    if not (query and query.strip()):
        logger.warning("hybrid_search: empty query provided, returning empty documents")
        return {"documents": []}

    # Преобразуем document_types в enum если указаны (case-insensitive)
    document_type_enums = None
    if document_types:
        valid_types = []
        for dt in document_types:
            if not dt:
                continue
            # Пробуем прямое преобразование
            try:
                valid_types.append(DocumentType(dt))
            except ValueError:
                # Пробуем case-insensitive поиск
                dt_upper = dt.upper()
                found = None
                for doc_type in DocumentType:
                    if doc_type.value.upper() == dt_upper:
                        found = doc_type
                        break
                if found:
                    valid_types.append(found)
                    logger.debug("hybrid_search: converted document type '%s' to '%s'", dt, found.value)
                else:
                    logger.warning("hybrid_search: invalid document type '%s', ignoring. Valid types: %s", dt, [t.value for t in DocumentType])
        if valid_types:
            document_type_enums = valid_types
        else:
            logger.warning("hybrid_search: no valid document types found after filtering")

    async with app_container() as container:
        field_values_repo = await container.get(DocumentFieldValuesRepository)
        documents_repo = await container.get(DocumentsRepository)
        openai_client = await container.get(AsyncOpenAI)

        logger.info("hybrid_search: creating embedding for query")
        try:
            emb = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
                dimensions=1024,
            )
            query_embedding = emb.data[0].embedding
            logger.info("hybrid_search: embedding created, length=%d", len(query_embedding))
        except Exception as e:
            logger.error("hybrid_search: failed to create embedding: %s", str(e))
            return {"documents": []}

        try:
            logger.info("hybrid_search: calling independent_hybrid_search on field values")
            hits = await field_values_repo.independent_hybrid_search(
                query_text=query,
                query_embedding=query_embedding,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                field_ids=field_ids,
                exclude_field_ids=exclude_field_ids,
                document_types=document_type_enums,
                min_confidence=min_confidence,
                limit=limit * 10,  # Берем больше результатов, чтобы потом сгруппировать по документам
                text_weight=text_weight,
                vector_weight=vector_weight,
                k_text=k_text,
                k_vector=k_vector,
            )
            logger.info("hybrid_search: found %d field value hits", len(hits))

            if not hits:
                logger.info("hybrid_search: no hits found, returning empty documents")
                return {"documents": []}

            # Группируем результаты по документам
            documents_map = {}  # document_id -> {matched_fields: [], max_score: float, seen_field_ids: set}
            matched_field_value_ids = set()

            # Собираем все field_value_ids для загрузки field'ов
            field_value_ids = [hit.field_value.id for hit in hits]
            
            # Загружаем все field values с их fields одним запросом
            from sqlalchemy import select
            from sqlalchemy.orm import joinedload
            stmt = (
                select(DocumentFieldValue)
                .options(joinedload(DocumentFieldValue.field))
                .where(DocumentFieldValue.id.in_(field_value_ids))
            )
            result = await field_values_repo.session.execute(stmt)
            field_values_with_fields = {fv.id: fv for fv in result.scalars().unique().all()}

            for hit in hits:
                doc_id = hit.field_value.document_id
                if doc_id not in documents_map:
                    documents_map[doc_id] = {
                        "matched_fields": [],
                        "max_score": 0.0,
                        "seen_field_ids": set(),  # Для дедупликации полей в рамках документа
                    }
                
                # Получаем field_value с загруженным field
                field_value = field_values_with_fields.get(hit.field_value.id, hit.field_value)
                
                # Дедупликация: пропускаем если это поле уже было добавлено для этого документа
                if field_value.id in documents_map[doc_id]["seen_field_ids"]:
                    # Обновляем max_score если новый hit имеет больший score
                    documents_map[doc_id]["max_score"] = max(
                        documents_map[doc_id]["max_score"],
                        hit.hybrid_score
                    )
                    continue
                
                matched_field_value_ids.add(field_value.id)
                documents_map[doc_id]["seen_field_ids"].add(field_value.id)
                documents_map[doc_id]["matched_fields"].append({
                    "hit": hit,
                    "field_value": field_value,
                })
                documents_map[doc_id]["max_score"] = max(
                    documents_map[doc_id]["max_score"],
                    hit.hybrid_score
                )

            # Сортируем документы по максимальному score и берем top limit
            sorted_doc_ids = sorted(
                documents_map.keys(),
                key=lambda d_id: documents_map[d_id]["max_score"],
                reverse=True
            )[:limit]

            # Загружаем полную информацию о документах и всех их полях
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload, joinedload

            stmt = (
                select(Document)
                .options(
                    selectinload(Document.field_values).options(
                        joinedload(DocumentFieldValue.field)
                    )
                )
                .where(Document.id.in_(sorted_doc_ids))
            )
            result = await documents_repo.session.execute(stmt)
            documents = result.scalars().unique().all()

            # Формируем ответ
            document_results = []
            for doc in documents:
                doc_data = documents_map[doc.id]
                
                # Формируем matched_fields
                matched_fields = []
                for item in doc_data["matched_fields"]:
                    hit = item["hit"]
                    fv = item["field_value"]
                    matched_fields.append(
                        MatchedFieldValueResponse(
                            id=fv.id,
                            field_id=fv.field_id,
                            field_name=fv.field.name if fv.field else "Unknown",
                            value_text=fv.value_text,
                            confidence=fv.confidence,
                            page_num=fv.page_num,
                            bbox=fv.bbox,
                            text_rank=round(hit.text_rank, 4),
                            vector_score=round(hit.vec_score, 4),
                            hybrid_score=round(hit.hybrid_score, 4),
                        )
                    )

                # Формируем other_fields (все остальные поля документа, не найденные в поиске)
                # Только метаданные для понимания доступных полей
                other_fields = []
                for fv in doc.field_values:
                    if fv.id not in matched_field_value_ids:
                        other_fields.append(
                            OtherFieldValueResponse(
                                id=fv.id,
                                field_id=fv.field_id,
                                field_name=fv.field.name if fv.field else "Unknown",
                                short_description=fv.field.short_description if fv.field else None,
                            )
                        )

                # Нормализуем file_path (заменяем обратные слеши на прямые для веб-URL)
                normalized_file_path = doc.file_path.replace('\\', '/') if doc.file_path else doc.file_path
                
                document_results.append(
                    DocumentSearchResultResponse(
                        document_id=doc.id,
                        filename=doc.filename,
                        file_path=normalized_file_path,
                        content_type=doc.content_type,
                        document_type=doc.type.value,
                        status=doc.status.value,
                        summary=doc.summary,
                        matched_fields=matched_fields,
                        other_fields=other_fields,
                        max_hybrid_score=round(doc_data["max_score"], 4),
                    )
                )

            result = FieldValueSearchResponse(documents=document_results)
            logger.info(
                "╔═══════════════════════════════════════════════════════════════╗\n"
                "║ SEARCH RESULTS                                                ║\n"
                "╠═══════════════════════════════════════════════════════════════╣\n"
                "║ Documents found: %-47d ║\n"
                "║ Total field value hits: %-40d ║\n"
                "╚═══════════════════════════════════════════════════════════════╝",
                len(document_results),
                len(hits),
            )
            return result.model_dump()
        except Exception as e:
            logger.error("hybrid_search: error in independent_hybrid_search: %s", str(e), exc_info=True)
            return {"documents": []}
