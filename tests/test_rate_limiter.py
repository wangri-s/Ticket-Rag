"""
测试限流与配额管理 — src/utils/rate_limiter.py
"""
import sys
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.rate_limiter import (
    BudgetExceeded,
    InMemoryTokenBucket,
    RateLimitExceeded,
    RateLimiter,
    CostTracker,
    get_user_id,
    _current_user_id,
    set_current_user,
    get_current_user,
)


# ══════════════════════════════════════════════════════════
# InMemoryTokenBucket
# ══════════════════════════════════════════════════════════

class TestInMemoryTokenBucket:
    """线程安全的 Token Bucket"""

    def test_first_consume_success(self):
        bucket = InMemoryTokenBucket(capacity=10.0, refill_rate=1.0)
        allowed, remaining, retry, cap = bucket.consume("test")
        assert allowed is True
        assert remaining == 9.0
        assert retry == 0.0
        assert cap == 10.0

    def test_consume_to_exhaustion(self):
        bucket = InMemoryTokenBucket(capacity=5.0, refill_rate=1.0)
        for i in range(5):
            allowed, _, _, _ = bucket.consume("test")
            assert allowed is True
        # 第 6 次应失败
        allowed, remaining, retry, _ = bucket.consume("test")
        assert allowed is False
        assert remaining == 0.0

    def test_refill_after_wait(self):
        bucket = InMemoryTokenBucket(capacity=10.0, refill_rate=10.0)  # 10 tokens/s
        # 耗尽
        for _ in range(10):
            bucket.consume("test")
        # 等 0.3s → 补充 3 tokens
        time.sleep(0.3)
        allowed, remaining, _, _ = bucket.consume("test")
        assert allowed is True
        assert remaining >= 1.5

    def test_capacity_cap(self):
        """token 不超 capacity"""
        bucket = InMemoryTokenBucket(capacity=5.0, refill_rate=100.0)
        time.sleep(0.2)  # 理论上能补 20 tokens，但 capped 到 5
        allowed, remaining, _, cap = bucket.consume("test")
        assert allowed is True
        assert remaining <= cap

    def test_different_keys_independent(self):
        bucket = InMemoryTokenBucket(capacity=3.0, refill_rate=1.0)
        # 耗尽 key A
        for _ in range(3):
            bucket.consume("A")
        allowed_a, _, _, _ = bucket.consume("A")
        assert allowed_a is False
        # key B 不受影响
        allowed_b, remaining_b, _, _ = bucket.consume("B")
        assert allowed_b is True
        assert remaining_b == 2.0

    def test_retry_after_calculation(self):
        bucket = InMemoryTokenBucket(capacity=5.0, refill_rate=2.0)  # 2 tokens/s
        # 消耗 5 个后 retry_after
        for _ in range(5):
            bucket.consume("test")
        allowed, remaining, retry, _ = bucket.consume("test")
        assert not allowed
        # 缺 1 token, rate=2/s → retry ≈ 0.5s
        assert 0.4 <= retry <= 0.6

    def test_zero_refill_rate(self):
        bucket = InMemoryTokenBucket(capacity=3.0, refill_rate=0.0)
        for _ in range(3):
            bucket.consume("test")
        allowed, _, retry, _ = bucket.consume("test")
        assert allowed is False
        assert retry == 60.0  # 硬编码兜底


class TestInMemoryTokenBucketConcurrency:
    """并发安全"""

    def test_concurrent_no_over_consume(self):
        bucket = InMemoryTokenBucket(capacity=10.0, refill_rate=1.0)
        results = []

        def worker():
            allowed, _, _, _ = bucket.consume("shared")
            results.append(allowed)

        threads = [threading.Thread(target=worker) for _ in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 最多 10 次成功（capacity=10）
        success_count = sum(results)
        assert success_count <= 10


# ══════════════════════════════════════════════════════════
# 用户 ID 提取
# ══════════════════════════════════════════════════════════

class TestGetUserId:
    """从请求中提取用户标识"""

    def _make_request(self, headers=None, client_host="192.168.1.1"):
        req = MagicMock()
        req.headers = headers or {}
        req.client = MagicMock()
        req.client.host = client_host
        return req

    def test_api_key_priority(self):
        req = self._make_request({"X-API-Key": "sk-abc123def456"})
        uid = get_user_id(req)
        assert uid.startswith("apikey:")

    def test_user_id_fallback(self):
        req = self._make_request({"X-User-ID": "user-001"})
        uid = get_user_id(req)
        assert uid.startswith("user:")

    def test_api_key_overrides_user_id(self):
        req = self._make_request({
            "X-API-Key": "sk-abc",
            "X-User-ID": "user-001",
        })
        uid = get_user_id(req)
        assert uid.startswith("apikey:")

    def test_session_id_fallback(self):
        req = self._make_request()
        uid = get_user_id(req, session_id="sess-001")
        assert uid.startswith("session:")
        assert "sess-001" in uid

    def test_ip_fallback(self):
        req = self._make_request()
        uid = get_user_id(req)
        assert uid.startswith("ip:")

    def test_x_forwarded_for(self):
        req = self._make_request({"X-Forwarded-For": "10.0.0.1, 10.0.0.2"})
        uid = get_user_id(req)
        assert "10.0.0.1" in uid

    def test_no_client(self):
        req = MagicMock()
        req.headers = {}
        req.client = None
        uid = get_user_id(req)
        assert uid == "ip:unknown"


# ══════════════════════════════════════════════════════════
# Context Variables
# ══════════════════════════════════════════════════════════

class TestContextVars:
    """contextvars 跨调用链传播"""

    def test_set_and_get(self):
        set_current_user("user-001")
        assert get_current_user() == "user-001"

    def test_default_empty(self):
        # 重置 contextvar（新 token）
        pass  # 默认值为 ""

    def test_multiple_set(self):
        set_current_user("user-A")
        assert get_current_user() == "user-A"
        set_current_user("user-B")
        assert get_current_user() == "user-B"


# ══════════════════════════════════════════════════════════
# RateLimiter
# ══════════════════════════════════════════════════════════

class TestRateLimiterDisabled:
    """限流禁用模式"""

    @patch("src.utils.rate_limiter.get_config")
    def test_disabled_passes(self, mock_cfg):
        from src.config import RateLimitConfig
        mock_cfg.return_value.rate_limit = RateLimitConfig({"enabled": False})
        # 需要重新实例化（绕过单例）
        limiter = RateLimiter.__new__(RateLimiter)
        limiter._cfg = RateLimitConfig({"enabled": False})
        # 不应抛异常
        limiter.check_global()
        limiter.check_user("test-user")


class TestRateLimiterMemoryMode:
    """进程内存模式"""

    def test_global_rate_limit(self, monkeypatch):
        """全局 burst=3，第4次超限"""
        from src.config import RateLimitConfig
        cfg = RateLimitConfig({
            "enabled": True,
            "global_rpm": 60,
            "global_burst": 3,
            "user_rpm": 20,
            "user_burst": 10,
            "redis_enabled": False,
        })
        # 直接使用 InMemoryTokenBucket 测试
        bucket = InMemoryTokenBucket(capacity=3.0, refill_rate=60.0 / 60.0)
        for _ in range(3):
            allowed, _, _, _ = bucket.consume("test")
            assert allowed
        allowed, _, retry, _ = bucket.consume("test")
        assert not allowed
        assert retry > 0

    def test_user_rate_limit(self):
        bucket = InMemoryTokenBucket(capacity=5.0, refill_rate=20.0 / 60.0)
        for _ in range(5):
            allowed, _, _, _ = bucket.consume("user:test")
            assert allowed
        allowed, _, retry, _ = bucket.consume("user:test")
        assert not allowed


# ══════════════════════════════════════════════════════════
# CostTracker
# ══════════════════════════════════════════════════════════

class TestCostTracker:
    """LLM 成本追踪"""

    @patch("redis.Redis")
    def test_unavailable_does_not_block(self, mock_redis_cls):
        """Redis 不可用时成本检查放行"""
        mock_redis_cls.return_value.ping.side_effect = ConnectionError("no redis")

        from src.config import RateLimitConfig
        cfg = RateLimitConfig({
            "enabled": True,
            "global_rpm": 60,
            "global_burst": 15,
            "user_rpm": 20,
            "user_burst": 5,
            "llm_max_tokens_per_hour": 100000,
            "llm_max_requests_per_hour": 50,
            "redis_enabled": True,
            "cost_enabled": True,
        })
        tracker = CostTracker(config=cfg)
        # 不可用时不抛异常
        tracker.check_budget("user-001")  # 不应抛异常
        # record_usage 也不应抛
        tracker.record_usage("user-001", 500, 300)  # 不应抛异常


# ══════════════════════════════════════════════════════════
# RateLimitExceeded
# ══════════════════════════════════════════════════════════

class TestRateLimitExceeded:
    """超限异常"""

    def test_exception_attributes(self):
        exc = RateLimitExceeded(retry_after=5.0, limit=60, remaining=0)
        assert exc.retry_after == 5.0
        assert exc.limit == 60
        assert exc.remaining == 0
        assert "5.0" in str(exc) or "5" in str(exc)


class TestBudgetExceeded:
    """预算超限异常"""

    def test_exception_attributes(self):
        exc = BudgetExceeded("token 用尽", retry_after=3600)
        assert exc.retry_after == 3600
        assert "token" in str(exc)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
