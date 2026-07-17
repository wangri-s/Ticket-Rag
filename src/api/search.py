"""
检索 API — 提供语义 / 关键字 / 混合三种检索接口

端点:
  POST /api/search/semantic   — 语义检索（稠密向量）
  POST /api/search/keyword    — 关键字检索（BM25 稀疏向量）
  POST /api/search/hybrid     — 混合检索（加权重排）
"""

import logging
from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from src.embedding.embedding_client import EmbeddingClient
from src.embedding.sparse_embedder import BM25SparseEmbedder
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import MedicalWorkOrderChunker
from src.retrieval.milvus_client import MilvusStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["检索"])


# ── 请求/响应模型 ────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., description="查询文本", examples=["CT扫描图像伪影"])
    top_k: int = Field(default=5, ge=1, le=50, description="返回条数")
    ticket_id: Optional[str] = Field(default=None, description="按工单号过滤")


class HybridSearchRequest(SearchRequest):
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="语义权重")
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="关键字权重")


class SearchHit(BaseModel):
    id: int
    score: float
    content: str
    ticket_id: str
    source: str
    chunk_index: int


class SearchResponse(BaseModel):
    query: str
    mode: str
    hits: list[SearchHit]
    total: int


# ── 懒加载组件 ───────────────────────────────

@lru_cache(maxsize=1)
def _get_dense_embedder() -> EmbeddingClient:
    return EmbeddingClient()


@lru_cache(maxsize=1)
def _get_sparse_embedder() -> BM25SparseEmbedder:
    """加载语料并训练 BM25（首次调用时执行，后续缓存）"""
    logger.info("正在训练 BM25 模型...")
    docs = DocumentLoader().load_directory("data/documents")
    chunks = MedicalWorkOrderChunker().split_documents(docs)
    contents = [c.page_content for c in chunks]
    embedder = BM25SparseEmbedder()
    embedder.fit(contents)
    logger.info(f"BM25 训练完成: {len(contents)} 个 chunk")
    return embedder


@lru_cache(maxsize=1)
def _get_store() -> MilvusStore:
    return MilvusStore()


# ── 端点 ─────────────────────────────────────

@router.post("/semantic", response_model=SearchResponse)
def semantic_search(req: SearchRequest):
    """语义检索：基于稠密向量相似度"""
    emb = _get_dense_embedder()
    store = _get_store()

    query_vec = emb.embed(req.query)
    expr = f'ticket_id == "{req.ticket_id}"' if req.ticket_id else None
    hits = store.search(query_vec, top_k=req.top_k, expr=expr)

    return SearchResponse(
        query=req.query,
        mode="semantic",
        hits=[SearchHit(**h) for h in hits],
        total=len(hits),
    )


@router.post("/keyword", response_model=SearchResponse)
def keyword_search(req: SearchRequest):
    """关键字检索：基于 BM25 稀疏向量"""
    sparse = _get_sparse_embedder()
    store = _get_store()

    query_sparse = sparse.encode_queries([req.query])[0]
    expr = f'ticket_id == "{req.ticket_id}"' if req.ticket_id else None
    hits = store.sparse_search(query_sparse, top_k=req.top_k, expr=expr)

    return SearchResponse(
        query=req.query,
        mode="keyword",
        hits=[SearchHit(**h) for h in hits],
        total=len(hits),
    )


@router.post("/hybrid", response_model=SearchResponse)
def hybrid_search(req: HybridSearchRequest):
    """混合检索：语义 + 关键字加权重排"""
    emb = _get_dense_embedder()
    sparse = _get_sparse_embedder()
    store = _get_store()

    query_dense = emb.embed(req.query)
    query_sparse = sparse.encode_queries([req.query])[0]
    expr = f'ticket_id == "{req.ticket_id}"' if req.ticket_id else None

    hits = store.hybrid_search(
        query_dense, query_sparse,
        top_k=req.top_k,
        dense_weight=req.dense_weight,
        sparse_weight=req.sparse_weight,
        expr=expr,
    )

    return SearchResponse(
        query=req.query,
        mode=f"hybrid (dense={req.dense_weight}, sparse={req.sparse_weight})",
        hits=[SearchHit(**h) for h in hits],
        total=len(hits),
    )
