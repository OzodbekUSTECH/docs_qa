import logging
from agents import function_tool
from app.dto.chat import HybridSearchResponse, SearchChunkResponse

logger = logging.getLogger("app.tools.hybrid_search")


@function_tool
async def hybrid_search(
    query: str,
    document_ids: list[int] | None = None,
    exclude_document_ids: list[int] | None = None,
    include_chunk_ids: list[int] | None = None,
    exclude_chunk_ids: list[int] | None = None,
    limit: int = 15,
    text_weight: float = 0.6,
    vector_weight: float = 0.4,
    k_text: int = 64,
    k_vector: int = 64,
) -> dict:
    """
    Выполнить мощный гибридный поиск по фрагментам документов.

    Механизм: Независимо выполняет полнотекстовый (FTS) и векторный поиск по chunks, затем объединяет, ранжирует и возвращает наиболее релевантные результаты.

    Аргументы:
        query (str): Пользовательский поисковый запрос.
        document_ids (list[int], опционально): Ограничить поиск определенными документами (по их ID).
        exclude_document_ids (list[int], опционально): Исключить определенные документы (по их ID).
        include_chunk_ids (list[int], опционально): Ограничить поиск определёнными chunk'ами (по их ID).
        exclude_chunk_ids (list[int], опционально): Исключить определённые chunk'и (по их ID).
        limit (int, опционально): Максимальное количество возвращаемых результатов (по умолчанию 15).
        text_weight (float, опционально): Вес текстового скоринга при ранжировании (по умолчанию 0.6).
        vector_weight (float, опционально): Вес векторного скоринга при ранжировании (по умолчанию 0.4).
        k_text (int, опционально): Сколько top-k кандидатов брать из FTS (по умолчанию 64).
        k_vector (int, опционально): Сколько top-k кандидатов брать из векторного поиска (по умолчанию 64).

    Returns:
        dict: Список наиболее релевантных chunks документов с их оценками. Пример:
            {"chunks": [{"chunk_id": int, "document_id": int, "score": float, ...}, ...]}
    """
    from app.di.containers import app_container
    from app.repositories.documents import DocumentChunksRepository
    from openai import AsyncOpenAI

    logger.info(
        "╔═══════════════════════════════════════════════════════════════╗\n"
        "║ HYBRID SEARCH                                                  ║\n"
        "╠═══════════════════════════════════════════════════════════════╣\n"
        "║ Query: %-55s ║\n"
        "║ Document IDs: %-48s ║\n"
        "║ Limit: %-54d ║\n"
        "║ Weights: text=%.2f, vector=%.2f                              ║\n"
        "║ K values: text=%d, vector=%d                                  ║\n"
        "╚═══════════════════════════════════════════════════════════════╝",
        query[:55] if query else "None",
        str(document_ids)[:48] if document_ids else "None",
        limit,
        text_weight,
        vector_weight,
        k_text,
        k_vector,
    )

    if not (query and query.strip()):
        logger.warning("hybrid_search: empty query provided, returning empty chunks")
        return {"chunks": []}

    async with app_container() as container:
        repo = await container.get(DocumentChunksRepository)
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
            return {"chunks": []}

        try:
            logger.info("hybrid_search: calling independent_hybrid_search")
            hits = await repo.independent_hybrid_search(
                query_text=query,
                query_embedding=query_embedding,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                include_chunk_ids=include_chunk_ids,
                exclude_chunk_ids=exclude_chunk_ids,
                limit=limit,
                text_weight=text_weight,
                vector_weight=vector_weight,
                k_text=k_text,
                k_vector=k_vector,
            )
            logger.info(
                "╔═══════════════════════════════════════════════════════════════╗\n"
                "║ SEARCH RESULTS                                                ║\n"
                "╠═══════════════════════════════════════════════════════════════╣\n"
                "║ Total hits: %-51d ║",
                len(hits)
            )
            
            # Логируем статистику по скорам
            if hits:
                text_ranks = [h.text_rank for h in hits if h.text_rank > 0]
                vec_scores = [h.vec_score for h in hits if h.vec_score > 0]
                logger.info(
                    "║ FTS matches: %-48d ║\n"
                    "║ Vector matches: %-46d ║\n"
                    "║ Score ranges: FTS max=%.4f, Vector max=%.4f            ║\n"
                    "║ Hybrid scores: min=%.4f, max=%.4f                        ║\n"
                    "╚═══════════════════════════════════════════════════════════════╝",
                    len(text_ranks),
                    len(vec_scores),
                    max(text_ranks) if text_ranks else 0,
                    max(vec_scores) if vec_scores else 0,
                    min(h.hybrid_score for h in hits),
                    max(h.hybrid_score for h in hits),
                )

            chunks = [
                SearchChunkResponse(
                    id=h.chunk.id,
                    document_id=h.chunk.document_id,
                    chunk_index=h.chunk.chunk_index,
                    content=h.chunk.content,
                    text_rank=round(h.text_rank, 4),
                    vector_score=round(h.vec_score, 4),
                    hybrid_score=round(h.hybrid_score, 4),
                )
                for h in hits
            ]
            
            result = HybridSearchResponse(chunks=chunks)
            logger.info("✓ Successfully returned %d chunks", len(result.chunks))
            return result.model_dump()
        except Exception as e:
            logger.error("hybrid_search: error in independent_hybrid_search: %s", str(e), exc_info=True)
            return {"chunks": []}
