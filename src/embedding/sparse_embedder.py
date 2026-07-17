"""
BM25 稀疏向量生成器 — 用于关键字检索

基于 pymilvus.model.sparse.BM25EmbeddingFunction，
使用 jieba 中文分词，生成 SPARSE_FLOAT_VECTOR。

用法:
  embedder = BM25SparseEmbedder()
  embedder.fit(corpus)                          # 先训练（构建词表 + IDF）
  sparse_vecs = embedder.encode_documents(texts) # → list[dict[int, float]]
  query_vecs = embedder.encode_queries(queries)  # → list[dict[int, float]]
"""

import logging
from typing import Optional

from milvus_model.sparse import BM25EmbeddingFunction
from milvus_model.sparse.bm25.tokenizers import build_default_analyzer

logger = logging.getLogger(__name__)


def _sparse_to_dict(sparse_vec) -> dict:
    """
    scipy sparse matrix → Milvus sparse dict {int(dim): float(val)}

    兼容 csr_array（encode_documents 输出）和 coo_array（encode_queries 输出）。
    """
    # csr_array: .indices / .data
    if hasattr(sparse_vec, "indices"):
        return {int(i): float(v) for i, v in zip(sparse_vec.indices, sparse_vec.data)}
    # coo_array: .col / .data
    return {int(c): float(v) for c, v in zip(sparse_vec.col, sparse_vec.data)}


class BM25SparseEmbedder:
    """
    BM25 稀疏向量生成器（中文分词）

    典型流程:
      embedder = BM25SparseEmbedder()
      embedder.fit(all_chunks)                        # 训练 IDF
      sparse_vecs = embedder.encode_documents(chunks) # 写入 Milvus

      # 检索时:
      q_vec = embedder.encode_queries(["CT伪影"])     # 查询向量化
      store.sparse_search(q_vec[0])
    """

    def __init__(self, language: str = "zh"):
        self.analyzer = build_default_analyzer(language=language)
        self._bm25 = BM25EmbeddingFunction(self.analyzer)
        self._is_fitted = False

    # ── 训练 ──────────────────────────────────

    def fit(self, corpus: list[str]) -> "BM25SparseEmbedder":
        """用全量语料训练 BM25（构建词表 + IDF）"""
        if not corpus:
            raise ValueError("训练语料不能为空")
        self._bm25.fit(corpus)
        self._is_fitted = True
        logger.info(f"BM25 训练完成: {len(corpus)} 篇文档")
        return self

    # ── 编码 ──────────────────────────────────

    def encode_documents(self, texts: list[str]) -> list[dict]:
        """文档编码 → Milvus 可用的 sparse dict 列表"""
        self._ensure_fitted()
        sparse_vecs = self._bm25.encode_documents(texts)
        return [_sparse_to_dict(v) for v in sparse_vecs]

    def encode_queries(self, queries: list[str]) -> list[dict]:
        """查询编码 → Milvus 可用的 sparse dict 列表"""
        self._ensure_fitted()
        sparse_vecs = self._bm25.encode_queries(queries)
        return [_sparse_to_dict(v) for v in sparse_vecs]

    # ── 内部 ──────────────────────────────────

    def _ensure_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("BM25 尚未训练，请先调用 fit()")
