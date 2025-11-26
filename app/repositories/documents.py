import logging
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import Select, case, func, literal, select, text, union_all
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

        # Гибридный поиск с улучшенным ранжированием
        try:
            # Нормализуем запрос для поиска по field_name
            query_normalized = query_text.lower().strip()
            # Создаем варианты для поиска по field_name (заменяем пробелы на _ и наоборот)
            query_with_underscores = query_normalized.replace(" ", "_")
            query_words = query_normalized.split()
            
            # Создаем tsquery для FTS - используем OR для гибкости
            # Также добавляем prefix matching для частичных совпадений
            ts_query_parts = []
            for word in query_words:
                if word and len(word) > 1:
                    # Добавляем prefix matching (:*) для слов длиннее 2 символов
                    ts_query_parts.append(f"{word}:*")
            
            if ts_query_parts:
                ts_query_str = " | ".join(ts_query_parts)
            else:
                ts_query_str = func.replace(func.unaccent(query_text), " ", " | ")
            
            ts_query = func.to_tsquery("simple", ts_query_str)
            text_rank = func.ts_rank_cd(DocumentFieldValue.value_tsv, ts_query)
            
            # Добавляем бонус за совпадение с custom_field_name
            # Используем ILIKE для частичного совпадения с названием поля
            # Строим список условий для case()
            case_conditions = [
                # Точное совпадение (с заменой пробелов на _)
                (func.lower(DocumentFieldValue.custom_field_name) == query_with_underscores, literal(0.5)),
                # Частичное совпадение (содержит все слова запроса)
                (func.lower(DocumentFieldValue.custom_field_name).contains(query_with_underscores), literal(0.3)),
            ]
            # Содержит хотя бы одно значимое слово
            for word in query_words:
                if len(word) > 2:
                    case_conditions.append(
                        (func.lower(DocumentFieldValue.custom_field_name).contains(word), literal(0.15))
                    )
            
            field_name_bonus = case(*case_conditions, else_=literal(0.0))
            
            dist = cos_dist(DocumentFieldValue.embedding, query_embedding)
            vec_score = literal(1.0) - dist
            
            # Улучшенная формула hybrid_score с бонусом за название поля
            # hybrid = (text_rank * text_weight + vec_score * vector_weight) * (1 + field_name_bonus)
            base_score = (text_rank * text_weight) + (vec_score * vector_weight)

            # FTS поиск по value_tsv
            fts_stmt: Select = (
                select(
                    DocumentFieldValue,
                    text_rank.label("text_rank"),
                    literal(0.0).label("vec_score"),
                    field_name_bonus.label("field_bonus"),
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
            
            # Дополнительный FTS поиск по custom_field_name
            # Если поле содержит слова из запроса, добавляем его
            field_name_conditions = []
            for word in query_words:
                if len(word) > 2:  # Пропускаем короткие слова
                    field_name_conditions.append(
                        func.lower(DocumentFieldValue.custom_field_name).contains(word.lower())
                    )
            
            if field_name_conditions:
                from sqlalchemy import or_
                field_name_stmt: Select = (
                    select(
                        DocumentFieldValue,
                        literal(0.0).label("text_rank"),
                        literal(0.0).label("vec_score"),
                        field_name_bonus.label("field_bonus"),
                    )
                    .where(or_(*field_name_conditions))
                )
                field_name_stmt = self._apply_filters(
                    field_name_stmt,
                    document_ids=document_ids,
                    exclude_document_ids=exclude_document_ids,
                    field_ids=field_ids,
                    exclude_field_ids=exclude_field_ids,
                    document_types=document_types,
                    min_confidence=min_confidence,
                ).limit(k_text)

            # Vector поиск
            vec_stmt: Select = select(
                DocumentFieldValue,
                literal(0.0).label("text_rank"),
                vec_score.label("vec_score"),
                field_name_bonus.label("field_bonus"),
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

            # Объединяем результаты (FTS + field_name search + vector)
            if field_name_conditions:
                union_stmt = union_all(fts_stmt, field_name_stmt, vec_stmt).subquery()
            else:
                union_stmt = union_all(fts_stmt, vec_stmt).subquery()
            
            # Пересчитываем hybrid_score с учетом бонуса за название поля
            # Используем RRF-подобную формулу для лучшего объединения
            # Используем func.max() для агрегации колонок при GROUP BY
            final_stmt = (
                select(
                    union_stmt.c.id,
                    (
                        (func.coalesce(func.max(union_stmt.c.text_rank), 0.0) * text_weight + 
                         func.coalesce(func.max(union_stmt.c.vec_score), 0.0) * vector_weight) *
                        (literal(1.0) + func.coalesce(func.max(union_stmt.c.field_bonus), 0.0))
                    ).label("hybrid_score"),
                )
                .group_by(union_stmt.c.id)  # Дедупликация
                .order_by(text("hybrid_score DESC"))
                .limit(limit)
            )

            res = await self.session.execute(final_stmt)
            top_ids = [row[0] for row in res.all()]

            if not top_ids:
                # Если ничего не найдено, попробуем только vector search
                logger.info("independent_hybrid_search: no FTS hits, falling back to vector-only search")
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

            # Получаем полные объекты с правильными скорами
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
                    field_name_bonus.label("field_bonus"),
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
                field_b = float(r[3] or 0)
                # Улучшенная формула: базовый score умножаем на (1 + бонус за название)
                base_s = (text_r * text_weight) + (vec_s * vector_weight)
                hybrid_s = base_s * (1.0 + field_b)
                
                logger.debug(
                    "Field %s: text_rank=%.4f, vec_score=%.4f, field_bonus=%.4f, hybrid=%.4f",
                    r[0].custom_field_name or r[0].id, text_r, vec_s, field_b, hybrid_s
                )
                
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

 