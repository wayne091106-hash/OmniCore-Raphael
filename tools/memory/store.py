"""
tools/memory/store.py — Qdrant 記憶存取層
═════════════════════════════════════════
從 memory_test/raphael_mem2/memory_store.py 直搬。

功能：
  dense  : sentence-transformers + CUDA (multilingual-e5-small, 384-dim)
  sparse : fastembed BM42（CPU），詞頻 × 注意力權重
  hybrid : dense + sparse → Prefetch → RRF fusion
  quant  : ScalarQuantization INT8 on dense
  HNSW   : m=16, ef_construct=100

API:
  store / batch_store           — 寫入
  search / filtered_search      — 查詢（hybrid）
  prefetch_rerank / recommend / discover / group_search — 進階查詢
  get_all / scroll_page / count / category_breakdown    — 列舉統計
  update_importance / delete_by_id / delete_all         — 修改刪除
"""

import logging
import atexit
import gc
import hashlib
import math
import os
import re
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, Modifier,
    PointStruct, SparseVector,
    Filter, FieldCondition, MatchValue, Range,
    PayloadSchemaType,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    HnswConfigDiff,
    Prefetch, FusionQuery, Fusion,
    RecommendQuery, RecommendInput,
    ContextPair,
    DiscoverQuery, DiscoverInput,
)

log = logging.getLogger("memory.store")

# ── 設定 ────────────────────────────────────────────────────────────── #
COLLECTION   = "raphael_memory"
DENSE_MODEL  = "intfloat/multilingual-e5-small"
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
VECTOR_SIZE  = 384

CATEGORIES = ["personal", "preference", "technical", "project", "event", "credential", "other"]

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TOKEN_RE = re.compile(r"[\w@.+-]+|[\u4e00-\u9fff]{2,}", re.UNICODE)


def _fix_ort_cuda_path():
    """讓 fastembed 的 ORT 能找到 PyTorch 內附的 cuDNN DLL。"""
    try:
        torch_lib = str(__import__("pathlib").Path(torch.__file__).parent / "lib")
        os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")
    except Exception:
        pass


_fix_ort_cuda_path()


class MemoryStore:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6333,
        *,
        mode: str = "local",
        path: str | Path | None = None,
    ):
        self._closed = False
        if mode == "remote":
            self._local_mode = False
            self.client = QdrantClient(
                host=host,
                port=port,
                prefer_grpc=False,
                check_compatibility=False,
            )
        else:
            self._local_mode = True
            db_path = Path(path) if path else Path(__file__).resolve().parent / "qdrant_data"
            db_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(db_path))
        self._dense_model  = None
        self._sparse_model = None
        self._device       = _DEVICE
        self._embedding_backend = os.environ.get("RAPHAEL_EMBEDDING_BACKEND", "hashing").strip().lower()
        self._ensure_collection()
        atexit.register(self.close)

    # ── 內部：collection 初始化 ─────────────────────────────────────── #

    def _ensure_collection(self):
        existing = {c.name: c for c in self.client.get_collections().collections}
        if COLLECTION in existing:
            cfg = self.client.get_collection(COLLECTION)
            v = cfg.config.params.vectors
            if isinstance(v, dict):
                dense_cfg = v.get("dense")
                if dense_cfg and dense_cfg.size == VECTOR_SIZE:
                    return
            log.warning("Collection '%s' 格式不符，重建中...", COLLECTION)
            self.client.delete_collection(COLLECTION)

        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "dense": VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                    quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        )
                    ),
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF)
            },
        )
        if not self._local_mode:
            for field, schema in [
                ("user_id",    PayloadSchemaType.KEYWORD),
                ("category",   PayloadSchemaType.KEYWORD),
                ("importance", PayloadSchemaType.INTEGER),
            ]:
                self.client.create_payload_index(COLLECTION, field, schema)

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        client = getattr(self, "client", None)
        try:
            if client is not None:
                client.close()
        except Exception as e:
            log.warning("Qdrant client 關閉失敗: %s", e)
        finally:
            self.client = None
            self._dense_model = None
            self._sparse_model = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

    # ── 內部：embedding ─────────────────────────────────────────────── #

    def _dense(self, text: str, is_query: bool = False) -> list[float]:
        if self._embedding_backend == "hashing":
            return self._hash_dense(text)
        if self._dense_model is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from sentence_transformers import SentenceTransformer
                self._dense_model = SentenceTransformer(DENSE_MODEL, device=self._device)
        prefix = "query: " if is_query else "passage: "
        return self._dense_model.encode(
            [prefix + text], convert_to_numpy=True, show_progress_bar=False
        )[0].tolist()

    @property
    def device(self):
        return self._device

    def _hash_dense(self, text: str) -> list[float]:
        vec = [0.0] * VECTOR_SIZE
        tokens = self._tokens(text)
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % VECTOR_SIZE
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        raw = [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]
        grams = []
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text or ""):
            grams.extend(chunk[i:i + 2] for i in range(max(0, len(chunk) - 1)))
        return raw + grams

    def _sparse(self, text: str) -> SparseVector:
        if self._sparse_model is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from fastembed import SparseTextEmbedding
                self._sparse_model = SparseTextEmbedding(SPARSE_MODEL)
        result = list(self._sparse_model.embed([text]))[0]
        return SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )

    def _dense_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if self._embedding_backend == "hashing":
            return [self._hash_dense(t) for t in texts]
        if self._dense_model is None:
            self._dense(texts[0], is_query)
        prefix = "query: " if is_query else "passage: "
        vecs = self._dense_model.encode(
            [prefix + t for t in texts],
            convert_to_numpy=True, batch_size=64, show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def _sparse_batch(self, texts: list[str]) -> list[SparseVector]:
        if self._sparse_model is None:
            self._sparse(texts[0])
        results = list(self._sparse_model.embed(texts))
        return [SparseVector(indices=r.indices.tolist(), values=r.values.tolist()) for r in results]

    def _user_filter(self, user_id: str, category: str | None = None,
                     min_importance: int | None = None) -> Filter:
        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if category:
            must.append(FieldCondition(key="category", match=MatchValue(value=category)))
        if min_importance is not None:
            must.append(FieldCondition(key="importance", range=Range(gte=min_importance)))
        return Filter(must=must)

    def _fmt(self, p) -> dict:
        return {
            "id":         str(p.id),
            "memory":     p.payload.get("memory", ""),
            "category":   p.payload.get("category", "other"),
            "importance": p.payload.get("importance", 3),
            "created_at": p.payload.get("created_at", ""),
            "updated_at": p.payload.get("updated_at", ""),
            "access_count": p.payload.get("access_count", 0),
            "score":      getattr(p, "score", None),
        }

    @staticmethod
    def _norm_text(text: str) -> str:
        return re.sub(r"\s+", "", (text or "").strip().lower())

    @staticmethod
    def _freshness(created_at: str) -> float:
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400)
            return 1.0 / (1.0 + age_days / 30.0)
        except Exception:
            return 0.0

    def _lexical_score(self, query: str, memory: str) -> float:
        q_tokens = set(self._tokens(query))
        m_tokens = set(self._tokens(memory))
        if not q_tokens or not m_tokens:
            return 0.0
        overlap = len(q_tokens & m_tokens) / max(1, len(q_tokens))
        q_norm = self._norm_text(query)
        m_norm = self._norm_text(memory)
        if q_norm and (q_norm in m_norm or m_norm in q_norm):
            overlap = max(overlap, 0.85)
        return min(1.0, overlap)

    def _rerank(self, query: str, rows: list[dict], limit: int) -> list[dict]:
        ranked = []
        for row in rows:
            vector_score = row.get("score")
            if vector_score is None:
                vector_norm = 0.0
            else:
                vector_norm = max(0.0, min(1.0, float(vector_score)))
            lexical = self._lexical_score(query, row.get("memory", ""))
            importance = max(1, min(5, int(row.get("importance", 3)))) / 5.0
            freshness = self._freshness(row.get("updated_at") or row.get("created_at", ""))
            combined = (
                vector_norm * 0.62
                + lexical * 0.24
                + importance * 0.08
                + freshness * 0.06
            )
            row = dict(row)
            row["vector_score"] = vector_score
            row["lexical_score"] = round(lexical, 4)
            row["score"] = round(combined, 6)
            ranked.append(row)
        ranked.sort(key=lambda r: r.get("score", 0), reverse=True)
        return ranked[:limit]

    # ── 寫入 ────────────────────────────────────────────────────────── #

    def store(self, text: str, user_id: str, category: str = "other",
              importance: int = 3, tags: list[str] | None = None,
              dedupe: bool = True) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("memory text is empty")
        if dedupe:
            existing = self.search(text, user_id=user_id, limit=5)
            text_norm = self._norm_text(text)
            for row in existing:
                row_norm = self._norm_text(row.get("memory", ""))
                if (
                    row_norm == text_norm
                    or (text_norm and text_norm in row_norm)
                    or (row_norm and row_norm in text_norm)
                    or row.get("score", 0) >= 0.92
                ):
                    updated_payload = {
                        "importance": max(int(row.get("importance", 3)), max(1, min(5, importance))),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "access_count": int(row.get("access_count", 0) or 0) + 1,
                    }
                    if tags:
                        updated_payload["tags"] = sorted(set(row.get("tags", []) or []) | set(tags))
                    self.client.set_payload(
                        collection_name=COLLECTION,
                        payload=updated_payload,
                        points=[row["id"]],
                    )
                    return row["id"]

        mem_id = str(uuid.uuid4())
        payload = {
            "memory":     text,
            "user_id":    user_id,
            "category":   category if category in CATEGORIES else "other",
            "importance": max(1, min(5, importance)),
            "tags":       tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": "",
            "access_count": 0,
        }
        self.client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=mem_id,
                vector=self._vectors(text),
                payload=payload,
            )],
        )
        return mem_id

    def _vectors(self, text: str) -> dict:
        vectors = {"dense": self._dense(text)}
        if self._embedding_backend != "hashing":
            vectors["sparse"] = self._sparse(text)
        return vectors

    def batch_store(self, items: list[dict], user_id: str) -> list[str]:
        texts = [it["memory"] for it in items]
        dense_vecs  = self._dense_batch(texts)
        sparse_vecs = [] if self._embedding_backend == "hashing" else self._sparse_batch(texts)
        now = datetime.now(timezone.utc).isoformat()

        ids = [str(uuid.uuid4()) for _ in items]
        points = []
        for i, it in enumerate(items):
            vector = {"dense": dense_vecs[i]}
            if self._embedding_backend != "hashing":
                vector["sparse"] = sparse_vecs[i]
            points.append(PointStruct(
                id=ids[i],
                vector=vector,
                payload={
                    "memory":     it["memory"],
                    "user_id":    user_id,
                    "category":   it.get("category", "other") if it.get("category") in CATEGORIES else "other",
                    "importance": max(1, min(5, it.get("importance", 3))),
                    "tags":       it.get("tags", []),
                    "created_at": now,
                },
            ))
        self.client.upsert(collection_name=COLLECTION, points=points)
        return ids

    # ── 查詢：Hybrid（預設）────────────────────────────────────────── #

    def search(self, query: str, user_id: str, limit: int = 5) -> list[dict]:
        f = self._user_filter(user_id)
        query_limit = max(limit, limit * 4)
        if self._embedding_backend == "hashing":
            result = self.client.query_points(
                collection_name=COLLECTION,
                query=self._dense(query, is_query=True),
                using="dense",
                query_filter=f,
                limit=query_limit,
                with_payload=True,
            )
            return self._rerank(query, [self._fmt(p) for p in result.points], limit)
        result = self.client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                Prefetch(query=self._dense(query, is_query=True), using="dense", filter=f, limit=query_limit),
                Prefetch(query=self._sparse(query), using="sparse", filter=f, limit=query_limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=query_limit,
            with_payload=True,
        )
        return self._rerank(query, [self._fmt(p) for p in result.points], limit)

    # ── 查詢：帶過濾 ───────────────────────────────────────────────── #

    def filtered_search(self, query: str, user_id: str, category: str | None = None,
                        min_importance: int | None = None, limit: int = 5,
                        use_hybrid: bool = True) -> list[dict]:
        f = self._user_filter(user_id, category, min_importance)
        query_limit = max(limit, limit * 4)
        if use_hybrid and self._embedding_backend != "hashing":
            result = self.client.query_points(
                collection_name=COLLECTION,
                prefetch=[
                    Prefetch(query=self._dense(query, is_query=True), using="dense", filter=f, limit=query_limit),
                    Prefetch(query=self._sparse(query), using="sparse", filter=f, limit=query_limit),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=query_limit,
                with_payload=True,
            )
        else:
            result = self.client.query_points(
                collection_name=COLLECTION,
                query=self._dense(query, is_query=True),
                using="dense",
                query_filter=f,
                limit=query_limit,
                with_payload=True,
            )
        return self._rerank(query, [self._fmt(p) for p in result.points], limit)

    # ── 查詢：Prefetch rerank ──────────────────────────────────────── #

    def prefetch_rerank(self, query: str, user_id: str,
                        limit: int = 5, prefetch_limit: int = 30) -> list[dict]:
        f = self._user_filter(user_id)
        q_vec = self._dense(query, is_query=True)
        result = self.client.query_points(
            collection_name=COLLECTION,
            prefetch=[Prefetch(query=q_vec, using="dense", filter=f, limit=prefetch_limit)],
            query=q_vec,
            using="dense",
            limit=limit,
            with_payload=True,
        )
        return [self._fmt(p) for p in result.points]

    # ── 查詢：Recommend ────────────────────────────────────────────── #

    def recommend(self, positive_ids: list[str], user_id: str,
                  negative_ids: list[str] | None = None, limit: int = 5) -> list[dict]:
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=RecommendQuery(recommend=RecommendInput(
                positive=positive_ids, negative=negative_ids or [],
            )),
            using="dense",
            query_filter=self._user_filter(user_id),
            limit=limit,
            with_payload=True,
        )
        return [self._fmt(p) for p in result.points]

    # ── 查詢：Discovery ────────────────────────────────────────────── #

    def discover(self, target_query: str, context_pairs: list[tuple[str, str]],
                 user_id: str, limit: int = 5) -> list[dict]:
        target_vec = self._dense(target_query, is_query=True)
        ctx = [
            ContextPair(positive=self._dense(pos), negative=self._dense(neg))
            for pos, neg in context_pairs
        ]
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=DiscoverQuery(discover=DiscoverInput(target=target_vec, context=ctx)),
            using="dense",
            query_filter=self._user_filter(user_id),
            limit=limit,
            with_payload=True,
        )
        return [self._fmt(p) for p in result.points]

    # ── 查詢：Group search ─────────────────────────────────────────── #

    def group_search(self, query: str, user_id: str,
                     group_by: str = "category",
                     group_size: int = 2, limit: int = 5) -> dict[str, list[dict]]:
        result = self.client.query_points_groups(
            collection_name=COLLECTION,
            query=self._dense(query, is_query=True),
            using="dense",
            query_filter=self._user_filter(user_id),
            group_by=group_by,
            group_size=group_size,
            limit=limit,
        )
        out = {}
        for g in result.groups:
            out[str(g.id)] = [self._fmt(p) for p in g.hits]
        return out

    # ── 列舉 / 統計 ─────────────────────────────────────────────────── #

    def get_all(self, user_id: str, limit: int = 200) -> list[dict]:
        results, _ = self.client.scroll(
            collection_name=COLLECTION,
            scroll_filter=self._user_filter(user_id),
            limit=limit,
            with_payload=True,
        )
        return [self._fmt(r) for r in results]

    def scroll_page(self, user_id: str, limit: int = 5, offset=None):
        results, next_off = self.client.scroll(
            collection_name=COLLECTION,
            scroll_filter=self._user_filter(user_id),
            limit=limit,
            offset=offset,
            with_payload=True,
        )
        return [self._fmt(r) for r in results], next_off

    def count(self, user_id: str, category: str | None = None) -> int:
        result = self.client.count(
            collection_name=COLLECTION,
            count_filter=self._user_filter(user_id, category),
            exact=True,
        )
        return result.count

    def category_breakdown(self, user_id: str) -> dict[str, int]:
        try:
            result = self.client.facet(
                collection_name=COLLECTION,
                key="category",
                facet_filter=self._user_filter(user_id),
                limit=len(CATEGORIES) + 2,
            )
            counts = {hit.value: hit.count for hit in result.hits}
            return {cat: counts.get(cat, 0) for cat in CATEGORIES}
        except Exception:
            counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
            offset = None
            while True:
                records, offset = self.client.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=self._user_filter(user_id),
                    limit=500,
                    offset=offset,
                    with_payload=True,
                )
                for r in records:
                    cat = r.payload.get("category", "other")
                    if cat in counts:
                        counts[cat] += 1
                if offset is None:
                    break
            return counts

    # ── 修改 / 刪除 ─────────────────────────────────────────────────── #

    def update_importance(self, memory_id: str, importance: int):
        self.client.set_payload(
            collection_name=COLLECTION,
            payload={"importance": max(1, min(5, importance))},
            points=[memory_id],
        )

    def delete_by_id(self, memory_id: str):
        self.client.delete(collection_name=COLLECTION, points_selector=[memory_id])

    def delete_all(self, user_id: str):
        self.client.delete(
            collection_name=COLLECTION,
            points_selector=self._user_filter(user_id),
        )

    # ── Collection 統計 ─────────────────────────────────────────────── #

    def collection_info(self) -> dict:
        info = self.client.get_collection(COLLECTION)
        v = info.config.params.vectors
        dense_cfg = v.get("dense") if isinstance(v, dict) else None
        return {
            "points_count": info.points_count,
            "status":       str(info.status),
            "vector_size":  dense_cfg.size if dense_cfg else VECTOR_SIZE,
            "quantization": str(dense_cfg.quantization_config) if dense_cfg else None,
            "dense_model":  DENSE_MODEL,
            "sparse_model": SPARSE_MODEL,
            "device":       self._device,
        }
