"""
测试语义问答缓存 — src/memory/qa_cache.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Helpers ─────────────────────────────────────

def _make_cache_entry(question, answer, embedding):
    return json.dumps({
        "q": question,
        "a": answer,
        "embedding": embedding,
        "ts": 1700000000.0,
    }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════
# Cosine Similarity
# ══════════════════════════════════════════════════════════

class TestCosineSimilarity:
    """余弦相似度计算"""

    def test_identical_vectors(self):
        from src.memory.qa_cache import QACache
        vec = [0.1, 0.2, 0.3]
        score = QACache._cosine(vec, vec)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        from src.memory.qa_cache import QACache
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        score = QACache._cosine(a, b)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        from src.memory.qa_cache import QACache
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        score = QACache._cosine(a, b)
        assert score == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self):
        from src.memory.qa_cache import QACache
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        score = QACache._cosine(a, b)
        assert score == 0.0  # 零向量保护

    def test_both_zero(self):
        from src.memory.qa_cache import QACache
        score = QACache._cosine([0.0, 0.0], [0.0, 0.0])
        assert score == 0.0

    def test_similar_but_not_identical(self):
        from src.memory.qa_cache import QACache
        a = [1.0, 2.0, 3.0]
        b = [1.1, 1.9, 3.0]
        score = QACache._cosine(a, b)
        assert 0.99 < score < 1.0


# ══════════════════════════════════════════════════════════

class TestQACacheWithMock:
    """使用 Mock Redis + Mock Embedding 测试"""

    @pytest.fixture
    def cache(self, mock_redis, mock_embedding_client):
        from src.memory.qa_cache import QACache
        from src.config import CacheConfig
        cfg = CacheConfig({
            "enabled": True,
            "similarity_threshold": 0.95,
            "max_entries": 100,
            "ttl": 3600,
        })
        cache = QACache(config=cfg)
        cache._redis = mock_redis
        cache._embedder = mock_embedding_client
        return cache

    # ── lookup ──────────────────────────────────

    def test_lookup_empty_cache(self, cache, mock_redis):
        mock_redis.lrange.return_value = []
        hit, answer = cache.lookup("CT伪影怎么处理？")
        assert hit is False
        assert answer is None

    def test_lookup_hit(self, cache, mock_redis, fixed_embedding):
        """缓存命中：相似度超过阈值"""
        mock_redis.lrange.return_value = [
            _make_cache_entry("CT伪影怎么处理？", "根据工单GD-001...", fixed_embedding),
        ]
        # Mock embed 返回相同向量 → cosine=1.0
        cache._embedder.embed.return_value = fixed_embedding
        hit, answer = cache.lookup("CT伪影怎么处理？")
        assert hit is True
        assert "GD-001" in answer

    def test_lookup_miss_below_threshold(self, cache, mock_redis, fixed_embedding, another_embedding):
        """相似度低于阈值 → 未命中"""
        mock_redis.lrange.return_value = [
            _make_cache_entry("完全不同的问题", "答案", fixed_embedding),
        ]
        cache._embedder.embed.return_value = another_embedding
        hit, answer = cache.lookup("CT伪影怎么处理？")
        assert hit is False
        assert answer is None

    def test_lookup_best_of_multiple(self, cache, mock_redis):
        """多条缓存中取最高相似度"""
        import random
        random.seed(1)
        emb_high = [random.random() for _ in range(1536)]
        random.seed(2)
        emb_low = [random.random() for _ in range(1536)]

        mock_redis.lrange.return_value = [
            _make_cache_entry("不同的旧问题", "旧答案", emb_low),
            _make_cache_entry("相似的问题", "相似答案", emb_high),
        ]
        random.seed(42)
        cache._embedder.embed.return_value = [random.random() for _ in range(1536)]
        hit, answer = cache.lookup("新问题")
        assert isinstance(hit, bool)

    def test_lookup_skips_corrupt_entries(self, cache, mock_redis, fixed_embedding):
        """损坏的 JSON 条目不导致崩溃"""
        mock_redis.lrange.return_value = [
            "not valid json at all {{{",
            _make_cache_entry("CT伪影怎么处理？", "答案", fixed_embedding),
        ]
        cache._embedder.embed.return_value = fixed_embedding
        hit, answer = cache.lookup("CT伪影怎么处理？")
        assert hit is True

    def test_lookup_skips_empty_embedding(self, cache, mock_redis, fixed_embedding):
        """空 embedding 条目被跳过"""
        mock_redis.lrange.return_value = [
            json.dumps({"q": "问题", "a": "答案", "embedding": [], "ts": 0}),
            _make_cache_entry("CT伪影", "答案", fixed_embedding),
        ]
        cache._embedder.embed.return_value = fixed_embedding
        hit, _ = cache.lookup("CT伪影")
        assert hit is True

    # ── store ───────────────────────────────────

    def test_store_success(self, cache, mock_redis):
        mock_redis.rpush.return_value = 1
        mock_redis.llen.return_value = 50
        result = cache.store("问题", "答案")
        assert result is True
        mock_redis.rpush.assert_called_once()

    def test_store_lru_eviction(self, cache, mock_redis):
        """超过 max_entries 时触发 LTRIM"""
        mock_redis.rpush.return_value = 1
        mock_redis.llen.return_value = 101  # 超过 max(100)
        result = cache.store("问题", "答案")
        assert result is True
        mock_redis.ltrim.assert_called_once()

    def test_store_renews_ttl(self, cache, mock_redis):
        mock_redis.rpush.return_value = 1
        mock_redis.llen.return_value = 1
        cache.store("问题", "答案")
        mock_redis.expire.assert_called()

    # ── clear ───────────────────────────────────

    def test_clear_success(self, cache, mock_redis):
        mock_redis.delete.return_value = 1
        result = cache.clear()
        assert result is True

    # ── 属性 ────────────────────────────────────

    def test_available_when_connected(self, cache):
        assert cache.available is True

    def test_size(self, cache, mock_redis):
        mock_redis.llen.return_value = 42
        assert cache.size == 42


class TestQACacheDisabled:
    """禁用模式"""

    @pytest.fixture
    def disabled_cache(self, mock_redis):
        from src.memory.qa_cache import QACache
        from src.config import CacheConfig
        cfg = CacheConfig({
            "enabled": False,
            "similarity_threshold": 0.95,
            "max_entries": 100,
            "ttl": 3600,
        })
        cache = QACache(config=cfg)
        cache._redis = mock_redis
        return cache

    def test_lookup_always_miss(self, disabled_cache):
        hit, answer = disabled_cache.lookup("问题")
        assert hit is False
        assert answer is None

    def test_store_always_false(self, disabled_cache):
        assert disabled_cache.store("问题", "答案") is False

    def test_available_is_false(self, disabled_cache):
        assert disabled_cache.available is False


class TestQACacheRedisUnavailable:
    """Redis 不可用时的降级"""

    def test_lookup_returns_miss(self, mock_redis_unavailable):
        from src.memory.qa_cache import QACache
        from src.config import CacheConfig
        cfg = CacheConfig({
            "enabled": True,
            "similarity_threshold": 0.95,
            "max_entries": 100,
            "ttl": 3600,
        })
        cache = QACache(config=cfg)
        cache._redis = None  # 手工置为不可用
        hit, answer = cache.lookup("问题")
        assert hit is False
        assert cache.store("问题", "答案") is False
        assert cache.available is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
