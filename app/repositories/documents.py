import logging
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import Select, func, literal, select, text, union_all
from sqlalchemy.ext.asyncio import AsyncSession

from app.entities import Document, DocumentChunk
from app.repositories.base import BaseRepository

logger = logging.getLogger("app.repositories.documents")


# ---- cross-version helper for cosine distance ----
# работает с: col.cosine_distance(vec) ИЛИ оператором <=> ИЛИ func.cosine_distance
def cos_dist(col, vec):
    try:
        return col.cosine_distance(vec)  # у новых pgvector колонка имеет метод
    except Exception:
        pass
    try:
        return col.op("<=>")(vec)        # оператор pgvector (самый совместимый)
    except Exception:
        pass
    return func.cosine_distance(col, vec)  # запасной вариант


@dataclass
class SearchHit:
    chunk: DocumentChunk
    text_rank: float
    vec_score: float
    hybrid_score: float


class DocumentsRepository(BaseRepository[Document]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, entity=Document)


class DocumentChunksRepository(BaseRepository[DocumentChunk]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, entity=DocumentChunk)

    # ---------- filters ----------
    def _apply_filters(
        self,
        stmt: Select,
        *,
        document_ids: Sequence[int] | None,
        exclude_document_ids: Sequence[int] | None,
        include_chunk_ids: Sequence[int] | None,
        exclude_chunk_ids: Sequence[int] | None,
    ) -> Select:
        if document_ids:
            stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
        if exclude_document_ids:
            stmt = stmt.where(~DocumentChunk.document_id.in_(exclude_document_ids))
        if include_chunk_ids:
            stmt = stmt.where(DocumentChunk.id.in_(include_chunk_ids))
        if exclude_chunk_ids:
            stmt = stmt.where(~DocumentChunk.id.in_(exclude_chunk_ids))
        return stmt

    # ---------- pure FTS ----------
    async def fts_search(
        self,
        *,
        query_text: str,
        limit: int = 20,
        document_ids: Sequence[int] | None = None,
        exclude_document_ids: Sequence[int] | None = None,
        include_chunk_ids: Sequence[int] | None = None,
        exclude_chunk_ids: Sequence[int] | None = None,
    ) -> list[SearchHit]:
        if not (query_text and query_text.strip()):
            logger.warning("fts_search: empty query_text, returning empty list")
            return []

        logger.info("fts_search: searching with query_text='%s', limit=%d", query_text, limit)
        
        try:
            # Используем встроенные возможности PostgreSQL tsvector
            # Строим tsquery через SQL функции PostgreSQL для правильной обработки
            # Используем replace для преобразования пробелов в OR операторы
            ts_query = func.to_tsquery(
                "simple",
                func.replace(func.unaccent(query_text), " ", " | ")
            )
            text_rank = func.ts_rank_cd(DocumentChunk.content_tsv, ts_query)

            stmt: Select = (
                select(
                    DocumentChunk,
                    text_rank.label("text_rank"),
                    literal(0.0).label("vec_score"),
                    text_rank.label("hybrid_score"),
                )
                .where(DocumentChunk.content_tsv.op("@@")(ts_query))
            )
            stmt = self._apply_filters(
                stmt,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                include_chunk_ids=include_chunk_ids,
                exclude_chunk_ids=exclude_chunk_ids,
            ).order_by(text_rank.desc()).limit(limit)

            res = await self.session.execute(stmt)
            rows = res.all()
            hits = [
                SearchHit(
                    chunk=r[0],
                    text_rank=float(r[1] or 0),
                    vec_score=float(r[2] or 0),
                    hybrid_score=float(r[3] or 0),
                )
                for r in rows
            ]
            logger.info("fts_search: found %d hits", len(hits))
            return hits
        except Exception as e:
            logger.error("fts_search: error executing search: %s", str(e), exc_info=True)
            return []

    # ---------- pure vector ----------
    async def vector_search(
        self,
        *,
        query_embedding: Sequence[float],
        limit: int = 20,
        document_ids: Sequence[int] | None = None,
        exclude_document_ids: Sequence[int] | None = None,
        include_chunk_ids: Sequence[int] | None = None,
        exclude_chunk_ids: Sequence[int] | None = None,
    ) -> list[SearchHit]:
        if not query_embedding:
            logger.warning("vector_search: empty query_embedding, returning empty list")
            return []

        logger.info("vector_search: searching with embedding length=%d, limit=%d", len(query_embedding), limit)
        
        try:
            dist = cos_dist(DocumentChunk.embedding, query_embedding)
            vec_score = literal(1.0) - dist

            stmt: Select = select(
                DocumentChunk,
                literal(0.0).label("text_rank"),
                vec_score.label("vec_score"),
                vec_score.label("hybrid_score"),
            )
            stmt = self._apply_filters(
                stmt,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                include_chunk_ids=include_chunk_ids,
                exclude_chunk_ids=exclude_chunk_ids,
            ).order_by(vec_score.desc()).limit(limit)

            res = await self.session.execute(stmt)
            rows = res.all()
            hits = [
                SearchHit(
                    chunk=r[0],
                    text_rank=float(r[1] or 0),
                    vec_score=float(r[2] or 0),
                    hybrid_score=float(r[3] or 0),
                )
                for r in rows
            ]
            logger.info("vector_search: found %d hits", len(hits))
            return hits
        except Exception as e:
            logger.error("vector_search: error executing search: %s", str(e), exc_info=True)
            return []

    # ---------- independent union + re-rank ----------
    async def independent_hybrid_search(
        self,
        *,
        query_text: str,
        query_embedding: Sequence[float],
        document_ids: Sequence[int] | None = None,
        exclude_document_ids: Sequence[int] | None = None,
        include_chunk_ids: Sequence[int] | None = None,
        exclude_chunk_ids: Sequence[int] | None = None,
        limit: int = 8,
        k_text: int = 64,
        k_vector: int = 64,
        text_weight: float = 0.6,
        vector_weight: float = 0.4,
    ) -> list[SearchHit]:
        logger.info(
            "╔═══════════════════════════════════════════════════════════════╗\n"
            "║ INDEPENDENT HYBRID SEARCH                                    ║\n"
            "╠═══════════════════════════════════════════════════════════════╣\n"
            "║ Query: %-55s ║\n"
            "║ Embedding length: %-45d ║\n"
            "║ Limit: %-54d ║\n"
            "║ K values: text=%d, vector=%d                                ║\n"
            "║ Weights: text=%.2f, vector=%.2f                              ║\n"
            "╚═══════════════════════════════════════════════════════════════╝",
            (query_text[:55] if query_text else "None"),
            len(query_embedding) if query_embedding else 0,
            limit,
            k_text,
            k_vector,
            text_weight,
            vector_weight,
        )

        has_text = query_text and query_text.strip()
        has_vector = query_embedding and len(query_embedding) > 0

        if not has_text and not has_vector:
            logger.warning("independent_hybrid_search: no query_text or query_embedding, returning empty list")
            return []

        # Проверяем наличие данных в базе
        total_chunks_stmt = select(func.count(DocumentChunk.id))
        if document_ids:
            total_chunks_stmt = total_chunks_stmt.where(DocumentChunk.document_id.in_(document_ids))
        total_chunks_result = await self.session.execute(total_chunks_stmt)
        total_chunks_count = total_chunks_result.scalar() or 0
        logger.info("📊 Total chunks in database: %d", total_chunks_count)
        
        if total_chunks_count == 0:
            logger.warning("independent_hybrid_search: no chunks found in database")
            return []

        # Если есть только текст или только вектор, используем простой поиск
        if has_text and not has_vector:
            logger.info("independent_hybrid_search: text-only search, using fts_search")
            return await self.fts_search(
                query_text=query_text,
                limit=limit,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                include_chunk_ids=include_chunk_ids,
                exclude_chunk_ids=exclude_chunk_ids,
            )
        
        if has_vector and not has_text:
            logger.info("independent_hybrid_search: vector-only search, using vector_search")
            return await self.vector_search(
                query_embedding=query_embedding,
                limit=limit,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                include_chunk_ids=include_chunk_ids,
                exclude_chunk_ids=exclude_chunk_ids,
            )

        # Гибридный поиск: объединяем результаты FTS и Vector
            logger.info("🔄 Performing hybrid search...")
        
        try:
            # Используем встроенные возможности PostgreSQL tsvector
            # Строим tsquery через SQL функции PostgreSQL для правильной обработки
            # Используем replace для преобразования пробелов в OR операторы
            ts_query = func.to_tsquery(
                "simple",
                func.replace(func.unaccent(query_text), " ", " | ")
            )
            text_rank = func.ts_rank_cd(DocumentChunk.content_tsv, ts_query)
            dist = cos_dist(DocumentChunk.embedding, query_embedding)
            vec_score = literal(1.0) - dist

            # Проверяем значение ts_query (для отладки)
            try:
                ts_query_test_stmt = select(ts_query.label("ts_query_value"))
                ts_query_test_result = await self.session.execute(ts_query_test_stmt)
                ts_query_value = ts_query_test_result.scalar()
                logger.info("independent_hybrid_search: ts_query result='%s'", ts_query_value)
                if not ts_query_value or str(ts_query_value).strip() == '':
                    logger.warning(
                        "independent_hybrid_search: ts_query is empty for query '%s'. "
                        "FTS search will not work properly.",
                        query_text
                    )
            except Exception as e:
                logger.debug("independent_hybrid_search: could not test ts_query: %s", str(e))

            # Проверяем сколько chunks соответствуют FTS условию
            fts_count_stmt = select(func.count(DocumentChunk.id)).where(
                DocumentChunk.content_tsv.op("@@")(ts_query)
            )
            fts_count_stmt = self._apply_filters(
                fts_count_stmt,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                include_chunk_ids=include_chunk_ids,
                exclude_chunk_ids=exclude_chunk_ids,
            )
            fts_count_result = await self.session.execute(fts_count_stmt)
            fts_count = fts_count_result.scalar() or 0
            logger.info("independent_hybrid_search: FTS matches found=%d (query='%s')", fts_count, query_text)
            
            # Проверяем сколько chunks имеют заполненный content_tsv
            tsv_check_stmt = select(func.count(DocumentChunk.id)).where(
                DocumentChunk.content_tsv.isnot(None)
            )
            if document_ids:
                tsv_check_stmt = tsv_check_stmt.where(DocumentChunk.document_id.in_(document_ids))
            tsv_check_result = await self.session.execute(tsv_check_stmt)
            tsv_count = tsv_check_result.scalar() or 0
            logger.info("independent_hybrid_search: chunks with content_tsv=%d/%d", tsv_count, total_chunks_count)
            
            # ДИАГНОСТИКА: Проверяем что реально находится в content и content_tsv
            try:
                if document_ids:
                    debug_stmt = text("""
                        SELECT 
                            id,
                            LEFT(content, 200) as content_preview,
                            content_tsv::text as tsv_text,
                            to_tsvector('simple', unaccent(content))::text as computed_tsv
                        FROM document_chunks
                        WHERE document_id = ANY(:doc_ids)
                        LIMIT 3
                    """)
                    debug_result = await self.session.execute(debug_stmt, {"doc_ids": list(document_ids)})
                else:
                    debug_stmt = text("""
                        SELECT 
                            id,
                            LEFT(content, 200) as content_preview,
                            content_tsv::text as tsv_text,
                            to_tsvector('simple', unaccent(content))::text as computed_tsv
                        FROM document_chunks
                        LIMIT 3
                    """)
                    debug_result = await self.session.execute(debug_stmt)
                debug_rows = debug_result.all()
                logger.info("independent_hybrid_search: DEBUG - Sample chunks content and tsv:")
                for row in debug_rows:
                    logger.info(
                        "  Chunk ID=%d | content_preview='%s' | content_tsv='%s' | computed_tsv='%s'",
                        row[0],
                        row[1][:100] if row[1] else None,
                        row[2] if row[2] else None,
                        row[3] if row[3] else None,
                    )
                
                # Проверяем какие слова из запроса могут быть найдены
                query_words = query_text.lower().split()
                logger.info("independent_hybrid_search: DEBUG - Query words: %s", query_words)
                
                # Проверяем наличие каждого слова отдельно
                for word in query_words:
                    word_clean = word.strip('.,!?;:()[]{}"\'')
                    if not word_clean:
                        continue
                    word_query = func.plainto_tsquery("simple", func.unaccent(word_clean))
                    word_count_stmt = select(func.count(DocumentChunk.id)).where(
                        DocumentChunk.content_tsv.op("@@")(word_query)
                    )
                    if document_ids:
                        word_count_stmt = word_count_stmt.where(DocumentChunk.document_id.in_(document_ids))
                    word_count_result = await self.session.execute(word_count_stmt)
                    word_count = word_count_result.scalar() or 0
                    logger.info("independent_hybrid_search: DEBUG - Word '%s' found in %d chunks", word_clean, word_count)
            except Exception as debug_error:
                logger.debug("independent_hybrid_search: debug query failed: %s", str(debug_error))

            # FTS candidates
            fts_sel: Select = select(
                DocumentChunk.id.label("chunk_id"),
                text_rank.label("text_rank"),
                literal(0.0).label("vec_score"),
            ).where(DocumentChunk.content_tsv.op("@@")(ts_query))

            # Vector candidates
            vec_sel: Select = select(
                DocumentChunk.id.label("chunk_id"),
                literal(0.0).label("text_rank"),
                vec_score.label("vec_score"),
            )

            fts_sel = self._apply_filters(
                fts_sel,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                include_chunk_ids=include_chunk_ids,
                exclude_chunk_ids=exclude_chunk_ids,
            ).order_by(text_rank.desc()).limit(k_text)

            vec_sel = self._apply_filters(
                vec_sel,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                include_chunk_ids=include_chunk_ids,
                exclude_chunk_ids=exclude_chunk_ids,
            ).order_by(vec_score.desc()).limit(k_vector)

            # Проверяем количество результатов FTS и Vector отдельно
            fts_count_sel = select(func.count()).select_from(fts_sel.subquery())
            vec_count_sel = select(func.count()).select_from(vec_sel.subquery())
            fts_sel_count_result = await self.session.execute(fts_count_sel)
            vec_sel_count_result = await self.session.execute(vec_count_sel)
            fts_sel_count = fts_sel_count_result.scalar() or 0
            vec_sel_count = vec_sel_count_result.scalar() or 0
            logger.info(
                "independent_hybrid_search: FTS candidates=%d | Vector candidates=%d",
                fts_sel_count,
                vec_sel_count,
            )

            # Объединяем результаты
            union_cte = union_all(fts_sel, vec_sel).cte("u")

            # Агрегируем по chunk_id (берем максимальные значения)
            agg = select(
                union_cte.c.chunk_id.label("chunk_id"),
                func.max(union_cte.c.text_rank).label("text_rank"),
                func.max(union_cte.c.vec_score).label("vec_score"),
            ).group_by(union_cte.c.chunk_id).subquery("agg")

            # Вычисляем гибридный скор
            hybrid = (literal(text_weight) * agg.c.text_rank) + (literal(vector_weight) * agg.c.vec_score)

            stmt: Select = (
                select(
                    DocumentChunk,
                    agg.c.text_rank,
                    agg.c.vec_score,
                    hybrid.label("hybrid_score"),
                )
                .join(agg, agg.c.chunk_id == DocumentChunk.id)
                .order_by(hybrid.desc())
                .limit(limit)
            )

            res = await self.session.execute(stmt)
            rows = res.all()
            hits = [
                SearchHit(
                    chunk=row[0],
                    text_rank=float(row[1] or 0),
                    vec_score=float(row[2] or 0),
                    hybrid_score=float(row[3] or 0),
                )
                for row in rows
            ]
            
            # Детальная статистика по результатам
            fts_hits_count = sum(1 for h in hits if h.text_rank > 0)
            vec_hits_count = sum(1 for h in hits if h.vec_score > 0)
            logger.info(
                "╔═══════════════════════════════════════════════════════════════╗\n"
                "║ SEARCH COMPLETED                                              ║\n"
                "╠═══════════════════════════════════════════════════════════════╣\n"
                "║ Total hits: %-51d ║\n"
                "║ FTS hits: %-53d ║\n"
                "║ Vector hits: %-50d ║\n"
                "║ Both methods: %-47d ║\n"
                "╚═══════════════════════════════════════════════════════════════╝",
                len(hits),
                fts_hits_count,
                vec_hits_count,
                sum(1 for h in hits if h.text_rank > 0 and h.vec_score > 0),
            )
            
            # Если FTS не нашел результатов, логируем предупреждение
            if fts_hits_count == 0 and fts_count > 0:
                logger.warning(
                    "independent_hybrid_search: FTS found %d matches but text_rank is 0 in results. "
                    "This may indicate an issue with ts_rank_cd calculation.",
                    fts_count
                )
            elif fts_count == 0:
                logger.warning(
                    "independent_hybrid_search: FTS found 0 matches for query '%s'. "
                    "Check if content_tsv is properly indexed or query needs adjustment.",
                    query_text
                )
            
            # Если гибридный поиск не дал результатов, пробуем fallback
            if len(hits) == 0:
                logger.info("independent_hybrid_search: hybrid search returned 0 results, trying fallback")
                # Пробуем отдельно FTS и Vector
                fts_hits = await self.fts_search(
                    query_text=query_text,
                    limit=limit,
                    document_ids=document_ids,
                    exclude_document_ids=exclude_document_ids,
                    include_chunk_ids=include_chunk_ids,
                    exclude_chunk_ids=exclude_chunk_ids,
                )
                vec_hits = await self.vector_search(
                    query_embedding=query_embedding,
                    limit=limit,
                    document_ids=document_ids,
                    exclude_document_ids=exclude_document_ids,
                    include_chunk_ids=include_chunk_ids,
                    exclude_chunk_ids=exclude_chunk_ids,
                )
                
                # Объединяем и переранжируем в памяти
                chunk_map = {}
                for hit in fts_hits + vec_hits:
                    chunk_id = hit.chunk.id
                    if chunk_id not in chunk_map:
                        chunk_map[chunk_id] = hit
                    else:
                        # Объединяем скоры
                        existing = chunk_map[chunk_id]
                        chunk_map[chunk_id] = SearchHit(
                            chunk=hit.chunk,
                            text_rank=max(existing.text_rank, hit.text_rank),
                            vec_score=max(existing.vec_score, hit.vec_score),
                            hybrid_score=(text_weight * max(existing.text_rank, hit.text_rank) + 
                                        vector_weight * max(existing.vec_score, hit.vec_score)),
                        )
                
                hits = sorted(chunk_map.values(), key=lambda h: h.hybrid_score, reverse=True)[:limit]
                logger.info("independent_hybrid_search: fallback found %d hits", len(hits))
            
            return hits
        except Exception as e:
            logger.error("independent_hybrid_search: error executing hybrid search: %s", str(e), exc_info=True)
            # Fallback на отдельные поиски
            logger.info("independent_hybrid_search: trying fallback to separate searches")
            try:
                if has_text:
                    fts_hits = await self.fts_search(
                        query_text=query_text,
                        limit=limit,
                        document_ids=document_ids,
                        exclude_document_ids=exclude_document_ids,
                        include_chunk_ids=include_chunk_ids,
                        exclude_chunk_ids=exclude_chunk_ids,
                    )
                    if fts_hits:
                        return fts_hits
                
                if has_vector:
                    vec_hits = await self.vector_search(
                        query_embedding=query_embedding,
                        limit=limit,
                        document_ids=document_ids,
                        exclude_document_ids=exclude_document_ids,
                        include_chunk_ids=include_chunk_ids,
                        exclude_chunk_ids=exclude_chunk_ids,
                    )
                    if vec_hits:
                        return vec_hits
            except Exception as fallback_error:
                logger.error("independent_hybrid_search: fallback also failed: %s", str(fallback_error))
            
            return []
