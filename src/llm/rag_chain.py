"""
RAG 核心链 — 串联检索 → Prompt 构建 → LLM 生成

流程:
  用户问题 → 向量检索 → 构建 Prompt → 调用 LLM → 返回答案 + 引用来源

用法:
  chain = RAGChain()
  result = chain.ask("CT扫描图像伪影怎么处理？")
  print(result["answer"])    # LLM 生成的回答
  for src in result["sources"]:  # 引用的工单来源
      print(src["ticket_id"], src["content"][:50])
"""

import logging
from functools import lru_cache
from typing import Literal, Optional

from src.config import get_config
from src.embedding.embedding_client import EmbeddingClient
from src.embedding.sparse_embedder import BM25SparseEmbedder
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import MedicalWorkOrderChunker
from src.llm.llm_client import LLMClient
from src.llm.prompts import build_full_prompt, get_fallback_answer
from src.retrieval.milvus_client import MilvusStore

logger = logging.getLogger(__name__)

SearchMode = Literal["semantic", "keyword", "hybrid"]


# ── 懒加载组件（首次使用初始化，后续复用）─────

@lru_cache(maxsize=1)
def _get_dense_embedder() -> EmbeddingClient:
    return EmbeddingClient()


@lru_cache(maxsize=1)
def _get_sparse_embedder() -> BM25SparseEmbedder:
    """加载语料并训练 BM25"""
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


@lru_cache(maxsize=1)
def _get_llm() -> LLMClient:
    return LLMClient()


# ── RAG Chain ─────────────────────────────────

class RAGChain:
    """
    RAG 问答链：检索 + 生成。

    用法:
      chain = RAGChain()
      result = chain.ask("CT伪影怎么处理？")
      result = chain.ask("MRI啸叫", mode="keyword")
      result = chain.ask("设备故障", mode="hybrid", ticket_id_filter="GD-2026-03001")
    """

    def __init__(self):
        self._dense = _get_dense_embedder()
        self._sparse = _get_sparse_embedder()
        self._store = _get_store()
        self._llm = _get_llm()
        self._cfg = get_config()

    # ── 主入口 ──────────────────────────────

    def ask(
        self,
        question: str,
        mode: SearchMode = "hybrid",
        top_k: int = None,
        ticket_id_filter: str = None,
    ) -> dict:
        """
        RAG 问答主入口。

        参数:
          question:         用户问题
          mode:             检索模式 semantic | keyword | hybrid
          top_k:            检索条数（默认取 config.yml）
          ticket_id_filter: 按工单号过滤

        返回:
          {
            "question": str,           # 原始问题
            "answer":   str,           # LLM 生成回答
            "sources":  list[dict],    # 引用的 chunk 来源
            "mode":     str,           # 检索模式
            "has_answer": bool,        # 是否找到相关工单
          }
        """
        if top_k is None:
            top_k = self._cfg.llm.retrieval_top_k
        if mode is None:
            mode = self._cfg.retrieval.default_mode

        # 1. 构建过滤表达式
        expr = f'ticket_id == "{ticket_id_filter}"' if ticket_id_filter else None

        # 2. 检索
        chunks = self._retrieve(question, mode, top_k, expr)

        # 3. 按分数阈值过滤低相关结果
        score_threshold = self._cfg.retrieval.score_threshold
        chunks = [c for c in chunks if c.get("score", 0.0) >= score_threshold]

        # 4. 构建 Prompt
        system_prompt, user_message = build_full_prompt(
            question, chunks, ticket_id_filter,
        )

        # 5. 无结果 → 兜底
        if user_message is None:
            return {
                "question": question,
                "answer": get_fallback_answer(),
                "sources": [],
                "mode": mode,
                "has_answer": False,
            }

        # 5. 调用 LLM
        logger.info(f"RAG 问答: mode={mode} question='{question[:40]}' chunks={len(chunks)}")
        answer = self._llm.generate(
            user_message=user_message,
            system_prompt=system_prompt,
        )

        # 6. 返回
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "ticket_id": c.get("ticket_id", ""),
                    "content": c.get("content", ""),
                    "source": c.get("source", ""),
                    "score": c.get("score", 0.0),
                }
                for c in chunks
            ],
            "mode": mode,
            "has_answer": True,
        }

    # ── 检索分发 ────────────────────────────

    def _retrieve(
        self, question: str, mode: SearchMode, top_k: int, expr: str,
    ) -> list[dict]:
        """根据模式分发到不同检索方法"""
        if mode == "semantic":
            return self._retrieve_semantic(question, top_k, expr)
        elif mode == "keyword":
            return self._retrieve_keyword(question, top_k, expr)
        else:  # hybrid
            return self._retrieve_hybrid(question, top_k, expr)

    def _retrieve_semantic(self, question, top_k, expr) -> list[dict]:
        query_vec = self._dense.embed(question)
        return self._store.search(query_vec, top_k=top_k, expr=expr)

    def _retrieve_keyword(self, question, top_k, expr) -> list[dict]:
        query_sparse = self._sparse.encode_queries([question])[0]
        return self._store.sparse_search(query_sparse, top_k=top_k, expr=expr)

    def _retrieve_hybrid(self, question, top_k, expr) -> list[dict]:
        query_dense = self._dense.embed(question)
        query_sparse = self._sparse.encode_queries([question])[0]
        return self._store.hybrid_search(
            query_dense, query_sparse,
            top_k=top_k,
            dense_weight=self._cfg.retrieval.hybrid_dense_weight,
            sparse_weight=self._cfg.retrieval.hybrid_sparse_weight,
            expr=expr,
        )
