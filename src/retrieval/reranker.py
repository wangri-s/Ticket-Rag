"""
重排序模块 — 使用 DashScope qwen3-rerank 交叉编码器对检索结果二次打分

原理:
  检索阶段（双路召回）返回 top_k × oversample 候选 chunks，
  重排序阶段将每个 (query, chunk) 对送入交叉编码器，
  模型直接计算语义相关性分数（而非向量余弦），
  最终取 top_n 个最相关的 chunk。

优势:
  - 交叉编码器比双编码器（embedding）更精准，但更慢
  - 先召回（快、多）→ 再精排（慢、少）= 兼顾速度与精度
  - 解决 BM25 关键词匹配和稠密向量匹配的排序偏差

用法:
  reranker = Reranker()
  reranked = reranker.rerank(
      query="CT伪影怎么处理？",
      chunks=[{"content": "...", "score": 0.9, ...}, ...],
      top_n=5,
  )
"""

import logging
from typing import Optional

import dashscope
from dashscope import TextReRank
from http import HTTPStatus

from src.config import RerankConfig, get_config

logger = logging.getLogger(__name__)


class Reranker:
    """
    DashScope 交叉编码重排序器

    模型: qwen3-rerank（100+ 语言，最大 120K tokens/次，500 文档/次）
    """

    def __init__(self, config: Optional[RerankConfig] = None):
        cfg = config or get_config().rerank
        self.model = cfg.model
        self.api_key = cfg.dashscope_api_key
        self.max_retries = cfg.max_retries
        self.timeout = cfg.timeout

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """
        对检索结果重排序。

        参数:
          query:  用户查询文本
          chunks: 初检结果列表 [{"content": str, "score": float, ...}, ...]
          top_n:  最终保留条数

        返回:
          重排序后的 chunks（score 已更新为 rerank relevance_score），按分数降序
        """
        if not chunks:
            logger.info("Reranker: 输入 chunks 为空，跳过")
            return chunks

        # 提取文档文本（截断过长内容，rerank 单条上限 4000 tokens）
        documents = [c.get("content", "")[:3000] for c in chunks]

        logger.info(
            f"Reranker: query='{query[:40]}' "
            f"candidates={len(documents)} → top_n={top_n} "
            f"model={self.model}"
        )

        try:
            resp = TextReRank.call(
                model=self.model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
                return_documents=False,
                api_key=self.api_key,
            )
        except Exception as e:
            logger.warning(f"Reranker API 调用失败，回退原始排序: {e}")
            # 失败时返回原始排序（按原 score 降序取 top_n）
            chunks_sorted = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
            return chunks_sorted[:top_n]

        if resp.status_code != HTTPStatus.OK:
            logger.warning(
                f"Reranker 返回错误 code={resp.status_code} message={resp.message}，"
                f"回退原始排序"
            )
            chunks_sorted = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
            return chunks_sorted[:top_n]

        # 解析重排序结果
        results = resp.output.results if resp.output else []

        # 按 index 重建（reranker 只返回排序后的结果，不含原始字段）
        reranked = []
        for item in results:
            idx = item.index
            if idx < len(chunks):
                c = chunks[idx].copy()
                c["score"] = item.relevance_score  # 用 reranker 分数替换原分数
                c["rerank_score"] = item.relevance_score
                reranked.append(c)

        logger.info(
            f"Reranker 完成: {len(reranked)}/{len(chunks)} 条保留, "
            f"top_score={reranked[0]['score']:.4f}" if reranked else "无结果"
        )

        return reranked
