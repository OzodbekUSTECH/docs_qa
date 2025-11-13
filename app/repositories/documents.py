import logging
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import Select, func, literal, select, text, union_all
from sqlalchemy.ext.asyncio import AsyncSession

from app.entities import Document
from app.entities.documents import DocumentFieldValue
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
class FieldValueSearchHit:
    field_value: DocumentFieldValue
    text_rank: float
    vec_score: float
    hybrid_score: float


class DocumentsRepository(BaseRepository[Document]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, entity=Document)





class DocumentFieldValuesRepository(BaseRepository[DocumentFieldValue]):
    def __init__(self, session: AsyncSession):
        super().__init__(session, DocumentFieldValue)

    def _apply_filters(
        self,
        stmt: Select,
        *,
        document_ids: Sequence[int] | None,
        exclude_document_ids: Sequence[int] | None,
        field_ids: Sequence[int] | None,
        exclude_field_ids: Sequence[int] | None,
        document_types: Sequence | None,
        min_confidence: float | None,
    ) -> Select:
        # Проверяем, был ли уже сделан join с Document
        has_document_join = False
        if document_types:
            # Делаем join только если нужно фильтровать по типам документов
            stmt = stmt.join(Document, DocumentFieldValue.document_id == Document.id)
            has_document_join = True
            stmt = stmt.where(Document.type.in_(document_types))
        
        if document_ids:
            stmt = stmt.where(DocumentFieldValue.document_id.in_(document_ids))
        if exclude_document_ids:
            stmt = stmt.where(~DocumentFieldValue.document_id.in_(exclude_document_ids))
        if field_ids:
            stmt = stmt.where(DocumentFieldValue.field_id.in_(field_ids))
        if exclude_field_ids:
            stmt = stmt.where(~DocumentFieldValue.field_id.in_(exclude_field_ids))
        if min_confidence is not None:
            stmt = stmt.where(DocumentFieldValue.confidence >= min_confidence)
        return stmt

    async def fts_search(
        self,
        *,
        query_text: str,
        limit: int = 20,
        document_ids: Sequence[int] | None = None,
        exclude_document_ids: Sequence[int] | None = None,
        field_ids: Sequence[int] | None = None,
        exclude_field_ids: Sequence[int] | None = None,
        document_types: Sequence | None = None,
        min_confidence: float | None = None,
    ) -> list[FieldValueSearchHit]:
        if not (query_text and query_text.strip()):
            logger.warning("fts_search: empty query_text, returning empty list")
            return []

        logger.info("fts_search: searching field values with query_text='%s', limit=%d", query_text, limit)
        
        try:
            ts_query = func.to_tsquery(
                "simple",
                func.replace(func.unaccent(query_text), " ", " | ")
            )
            text_rank = func.ts_rank_cd(DocumentFieldValue.value_tsv, ts_query)

            stmt: Select = (
                select(
                    DocumentFieldValue,
                    text_rank.label("text_rank"),
                    literal(0.0).label("vec_score"),
                    text_rank.label("hybrid_score"),
                )
                .where(DocumentFieldValue.value_tsv.op("@@")(ts_query))
            )
            stmt = self._apply_filters(
                stmt,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                field_ids=field_ids,
                exclude_field_ids=exclude_field_ids,
                document_types=document_types,
                min_confidence=min_confidence,
            ).order_by(text_rank.desc()).limit(limit)

            res = await self.session.execute(stmt)
            rows = res.all()
            hits = [
                FieldValueSearchHit(
                    field_value=r[0],
                    text_rank=float(r[1] or 0),
                    vec_score=float(r[2] or 0),
                    hybrid_score=float(r[3] or 0),
                )
                for r in rows
            ]
            logger.info("fts_search: found %d field value hits", len(hits))
            return hits
        except Exception as e:
            logger.error("fts_search: error executing search: %s", str(e), exc_info=True)
            return []

    async def vector_search(
        self,
        *,
        query_embedding: Sequence[float],
        limit: int = 20,
        document_ids: Sequence[int] | None = None,
        exclude_document_ids: Sequence[int] | None = None,
        field_ids: Sequence[int] | None = None,
        exclude_field_ids: Sequence[int] | None = None,
        document_types: Sequence | None = None,
        min_confidence: float | None = None,
    ) -> list[FieldValueSearchHit]:
        if not query_embedding:
            logger.warning("vector_search: empty query_embedding, returning empty list")
            return []

        logger.info("vector_search: searching field values with embedding length=%d, limit=%d", len(query_embedding), limit)
        
        try:
            dist = cos_dist(DocumentFieldValue.embedding, query_embedding)
            vec_score = literal(1.0) - dist

            stmt: Select = select(
                DocumentFieldValue,
                literal(0.0).label("text_rank"),
                vec_score.label("vec_score"),
                vec_score.label("hybrid_score"),
            )
            stmt = self._apply_filters(
                stmt,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                field_ids=field_ids,
                exclude_field_ids=exclude_field_ids,
                document_types=document_types,
                min_confidence=min_confidence,
            ).order_by(vec_score.desc()).limit(limit)

            res = await self.session.execute(stmt)
            rows = res.all()
            hits = [
                FieldValueSearchHit(
                    field_value=r[0],
                    text_rank=float(r[1] or 0),
                    vec_score=float(r[2] or 0),
                    hybrid_score=float(r[3] or 0),
                )
                for r in rows
            ]
            logger.info("vector_search: found %d field value hits", len(hits))
            return hits
        except Exception as e:
            logger.error("vector_search: error executing search: %s", str(e), exc_info=True)
            return []

    async def independent_hybrid_search(
        self,
        *,
        query_text: str,
        query_embedding: Sequence[float],
        document_ids: Sequence[int] | None = None,
        exclude_document_ids: Sequence[int] | None = None,
        field_ids: Sequence[int] | None = None,
        exclude_field_ids: Sequence[int] | None = None,
        document_types: Sequence | None = None,
        min_confidence: float | None = None,
        limit: int = 8,
        k_text: int = 64,
        k_vector: int = 64,
        text_weight: float = 0.6,
        vector_weight: float = 0.4,
    ) -> list[FieldValueSearchHit]:
        logger.info(
            "independent_hybrid_search (field values): query='%s', embedding_len=%d, limit=%d",
            query_text[:50] if query_text else "None",
            len(query_embedding) if query_embedding else 0,
            limit
        )

        has_text = query_text and query_text.strip()
        has_vector = query_embedding and len(query_embedding) > 0

        if not has_text and not has_vector:
            logger.warning("independent_hybrid_search: no query_text or query_embedding, returning empty list")
            return []

        if has_text and not has_vector:
            return await self.fts_search(
                query_text=query_text,
                limit=limit,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                field_ids=field_ids,
                exclude_field_ids=exclude_field_ids,
                document_types=document_types,
                min_confidence=min_confidence,
            )

        if has_vector and not has_text:
            return await self.vector_search(
                query_embedding=query_embedding,
                limit=limit,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                field_ids=field_ids,
                exclude_field_ids=exclude_field_ids,
                document_types=document_types,
                min_confidence=min_confidence,
            )

        # Гибридный поиск
        try:
            ts_query = func.to_tsquery(
                "simple",
                func.replace(func.unaccent(query_text), " ", " | ")
            )
            text_rank = func.ts_rank_cd(DocumentFieldValue.value_tsv, ts_query)
            dist = cos_dist(DocumentFieldValue.embedding, query_embedding)
            vec_score = literal(1.0) - dist
            hybrid_score = (text_rank * text_weight) + (vec_score * vector_weight)

            # FTS поиск
            fts_stmt: Select = (
                select(
                    DocumentFieldValue,
                    text_rank.label("text_rank"),
                    literal(0.0).label("vec_score"),
                    text_rank.label("hybrid_score"),
                )
                .where(DocumentFieldValue.value_tsv.op("@@")(ts_query))
            )
            fts_stmt = self._apply_filters(
                fts_stmt,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                field_ids=field_ids,
                exclude_field_ids=exclude_field_ids,
                document_types=document_types,
                min_confidence=min_confidence,
            ).order_by(text_rank.desc()).limit(k_text)

            # Vector поиск
            vec_stmt: Select = select(
                DocumentFieldValue,
                literal(0.0).label("text_rank"),
                vec_score.label("vec_score"),
                vec_score.label("hybrid_score"),
            )
            vec_stmt = self._apply_filters(
                vec_stmt,
                document_ids=document_ids,
                exclude_document_ids=exclude_document_ids,
                field_ids=field_ids,
                exclude_field_ids=exclude_field_ids,
                document_types=document_types,
                min_confidence=min_confidence,
            ).order_by(vec_score.desc()).limit(k_vector)

            # Объединяем результаты
            union_stmt = union_all(fts_stmt, vec_stmt).subquery()
            
            # Пересчитываем hybrid_score и ранжируем
            final_stmt = (
                select(
                    union_stmt.c.id,
                    (func.coalesce(union_stmt.c.text_rank, 0.0) * text_weight + 
                     func.coalesce(union_stmt.c.vec_score, 0.0) * vector_weight).label("hybrid_score"),
                )
                .order_by(text("hybrid_score DESC"))
                .limit(limit)
            )

            res = await self.session.execute(final_stmt)
            top_ids = [row[0] for row in res.all()]

            if not top_ids:
                return []

            # Получаем полные объекты с правильными скорами
            # Пересчитываем text_rank и vec_score для каждого найденного значения
            full_stmt = (
                select(
                    DocumentFieldValue,
                    func.coalesce(
                        func.ts_rank_cd(DocumentFieldValue.value_tsv, ts_query),
                        literal(0.0)
                    ).label("text_rank"),
                    func.coalesce(
                        literal(1.0) - cos_dist(DocumentFieldValue.embedding, query_embedding),
                        literal(0.0)
                    ).label("vec_score"),
                )
                .where(DocumentFieldValue.id.in_(top_ids))
            )

            res = await self.session.execute(full_stmt)
            rows = res.all()

            # Создаем map для сохранения порядка
            hit_map = {}
            for r in rows:
                text_r = float(r[1] or 0)
                vec_s = float(r[2] or 0)
                hybrid_s = (text_r * text_weight) + (vec_s * vector_weight)
                hit_map[r[0].id] = FieldValueSearchHit(
                    field_value=r[0],
                    text_rank=text_r,
                    vec_score=vec_s,
                    hybrid_score=hybrid_s,
                )

            # Возвращаем в порядке hybrid_score
            hits = [hit_map[fid] for fid in top_ids if fid in hit_map]
            hits.sort(key=lambda h: h.hybrid_score, reverse=True)

            logger.info("independent_hybrid_search: found %d field value hits", len(hits))
            return hits
        except Exception as e:
            logger.error("independent_hybrid_search: error executing search: %s", str(e), exc_info=True)
            return []

 