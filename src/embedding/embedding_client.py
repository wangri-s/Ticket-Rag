"""
向量化客户端 — 封装 DashScope text-embedding-v1

支持单条和批量文本向量化，内置重试、限流保护和维度校验。
"""

import time
import logging
from typing import Optional, overload

from dashscope import TextEmbedding

from src.config import EmbeddingConfig, get_config

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    DashScope 文本向量化客户端

    用法:
      client = EmbeddingClient()
      vec = client.embed("CT 扫描偶发图像伪影")         # 单条 → list[float]
      vecs = client.embed_batch(["文本1", "文本2"])      # 批量 → list[list[float]]
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        cfg = config or get_config().embedding
        self.model = cfg.model
        self.dimensions = cfg.dimensions
        self.batch_size = cfg.batch_size
        self.max_retries = cfg.max_retries
        self.timeout = cfg.timeout
        self.sleep_interval = cfg.sleep_interval
        self.api_key = cfg.dashscope_api_key

    # ── 公开接口 ──────────────────────────────

    def embed(self, text: str) -> list[float]:
        """单条文本向量化"""
        if not text or not text.strip():
            raise ValueError("输入文本为空")

        result = self._call_with_retry([text])
        vector = result[0]
        self._validate_dim(vector)
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        批量文本向量化，自动按 batch_size 分批，
        批次之间 sleep 防止触发 API 限流。
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.info(f"Embedding 批次 [{i+1}-{min(i+len(batch), total)}/{total}]")

            vectors = self._call_with_retry(batch)
            for v in vectors:
                self._validate_dim(v)
            all_vectors.extend(vectors)

            # 批次间冷却，避免触发 QPS 限制
            if i + self.batch_size < total:
                time.sleep(self.sleep_interval)

        return all_vectors

    # ── 内部 ──────────────────────────────────

    def _call_with_retry(self, texts: list[str]) -> list[list[float]]:
        """带重试的 API 调用"""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_api(texts)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embedding API 调用失败 (第 {attempt}/{self.max_retries} 次): {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.sleep_interval * attempt)  # 指数退避

        raise RuntimeError(
            f"Embedding API 调用失败，已重试 {self.max_retries} 次"
        ) from last_error

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """单次 API 调用"""
        resp = TextEmbedding.call(
            model=self.model,
            input=texts,
            api_key=self.api_key,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"DashScope API 返回错误 "
                f"code={resp.status_code} message={resp.message}"
            )

        # 按输入顺序提取向量
        embeddings = resp.output.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"返回向量数 ({len(embeddings)}) 与输入文本数 ({len(texts)}) 不匹配"
            )

        return [emb["embedding"] for emb in embeddings]

    def _validate_dim(self, vector: list[float]) -> None:
        """校验输出维度与配置一致"""
        actual = len(vector)
        if actual != self.dimensions:
            raise ValueError(
                f"向量维度不匹配: 期望 {self.dimensions}, 实际 {actual}"
            )
