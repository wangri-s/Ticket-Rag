"""
语义问答缓存 — 相同/相似问题跳过 LLM 直接返回缓存答案

原理:
  1. 用户问题 → embedding 向量（DashScope text-embedding-v1）
  2. 与 Redis 中缓存的问题向量做余弦相似度比较
  3. 相似度 > 阈值 → 命中，返回缓存答案（省 1 次 LLM 调用）
  4. 相似度 ≤ 阈值 → 未命中，走正常 RAG → 缓存新 Q&A

Redis 存储:
  rag:cache:qa → List<JSON>  每条: {q, a, embedding, ts}
  自动 LRU: 超过 max_entries 时从左侧淘汰旧条目
  TTL: 默认 1 小时

用法:
  cache = QACache()
  hit, answer = cache.lookup("CT伪影怎么处理？")
  if hit:
      return answer  # 跳过 LLM
  # 否则正常 RAG...
  cache.store(question, answer)
"""

import json
import logging
import time
from typing import Optional, Tuple

import redis

from src.config import CacheConfig, get_config

logger = logging.getLogger(__name__)

CACHE_KEY = "rag:cache:qa"


class QACache:
    """
    语义问答缓存。

    用 embedding 向量 + 余弦相似度判断两个问题是否为"相同问题"，
    相同问题的已回答结果直接返回，不再调用 LLM。
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        cfg = config or get_config().cache
        self.enabled = cfg.enabled
        self.threshold = cfg.similarity_threshold
        self.max_entries = cfg.max_entries
        self.ttl = cfg.ttl

        # 复用全局 Redis 连接
        redis_cfg = get_config().memory.redis
        try:
            self._redis = redis.Redis(
                host=redis_cfg.host,
                port=redis_cfg.port,
                db=redis_cfg.db,
                password=redis_cfg.password,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            self._redis.ping()
            logger.info(f"QA 缓存已启用: threshold={self.threshold}, max={self.max_entries}, ttl={self.ttl}s")
        except Exception as e:
            logger.warning(f"QA 缓存不可用（Redis 连接失败）: {e}")
            self._redis = None

        self._embedder = None  # 懒加载

    # ── 获取 embedding ────────────────────────

    def _get_embedder(self):
        """懒加载 EmbeddingClient"""
        if self._embedder is None:
            from src.embedding.embedding_client import EmbeddingClient
            self._embedder = EmbeddingClient()
        return self._embedder

    def _embed(self, text: str) -> list[float]:
        """对问题文本做向量化"""
        emb = self._get_embedder()
        return emb.embed(text)

    # ── 余弦相似度 ────────────────────────────

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── 缓存操作 ──────────────────────────────

    def lookup(self, question: str) -> Tuple[bool, Optional[str]]:
        """
        查找缓存中是否有相似问题。

        返回: (hit: bool, answer: str|None)
          - hit=True  → 命中，可跳过 LLM
          - hit=False → 未命中，需正常 RAG
        """
        if not self.enabled or self._redis is None:
            return False, None

        try:
            # 获取缓存中所有条目
            raw_list = self._redis.lrange(CACHE_KEY, 0, -1)
            if not raw_list:
                return False, None

            # 对新问题做 embedding
            q_emb = self._embed(question)

            # 遍历缓存，找最相似的问题
            best_score = 0.0
            best_answer = None

            for raw in raw_list:
                try:
                    entry = json.loads(raw)
                    cached_emb = entry.get("embedding", [])
                    if not cached_emb:
                        continue
                    score = self._cosine(q_emb, cached_emb)
                    if score > best_score:
                        best_score = score
                        best_answer = entry.get("a", "")
                except (json.JSONDecodeError, KeyError):
                    continue

            if best_score >= self.threshold and best_answer:
                logger.info(
                    f"QA 缓存命中: similarity={best_score:.4f} "
                    f"question='{question[:40]}...'"
                )
                return True, best_answer

            logger.debug(
                f"QA 缓存未命中: best_similarity={best_score:.4f} "
                f"(threshold={self.threshold})"
            )
            return False, None

        except Exception as e:
            logger.error(f"QA 缓存查询失败: {e}")
            return False, None

    def store(self, question: str, answer: str) -> bool:
        """
        将 Q&A 存入缓存。

        参数:
          question: 用户问题
          answer:   LLM 回答
        """
        if not self.enabled or self._redis is None:
            return False

        try:
            q_emb = self._embed(question)
            entry = json.dumps({
                "q": question,
                "a": answer,
                "embedding": q_emb,
                "ts": time.time(),
            }, ensure_ascii=False)

            # RPUSH 加到列表右侧
            self._redis.rpush(CACHE_KEY, entry)

            # LRU: 超过 max_entries 时从左侧删除旧条目
            current_len = self._redis.llen(CACHE_KEY)
            if current_len > self.max_entries:
                overflow = current_len - self.max_entries
                self._redis.ltrim(CACHE_KEY, overflow, -1)

            # 续期 TTL
            self._redis.expire(CACHE_KEY, self.ttl)

            logger.debug(f"QA 缓存已存储: '{question[:40]}...' (total={min(current_len, self.max_entries)})")
            return True
        except Exception as e:
            logger.error(f"QA 缓存存储失败: {e}")
            return False

    def clear(self) -> bool:
        """清空所有缓存"""
        if self._redis is None:
            return False
        try:
            self._redis.delete(CACHE_KEY)
            return True
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return self.enabled and self._redis is not None

    @property
    def size(self) -> int:
        """当前缓存条目数"""
        if self._redis is None:
            return 0
        try:
            return self._redis.llen(CACHE_KEY)
        except Exception:
            return 0
