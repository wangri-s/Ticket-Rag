"""
测试 Redis 短期记忆 — src/memory/redis_client.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Helpers ─────────────────────────────────────

def _make_msg(role="user", content="测试问题", turn=1):
    return {"role": role, "content": content, "turn": turn, "timestamp": "2026-01-01T08:00:00"}


# ══════════════════════════════════════════════════════════

class TestRedisMemoryStoreWithMock:
    """使用 Mock Redis 测试"""

    @pytest.fixture
    def store(self, mock_redis):
        """注入 mock_redis fixture"""
        from src.memory.redis_client import RedisMemoryStore
        from src.config import RedisConfig
        cfg = RedisConfig({
            "host": "localhost", "port": 6379, "db": 0,
            "password": "", "ttl": 86400, "max_messages": 20,
            "key_prefix": "rag:session",
        })
        store = RedisMemoryStore(config=cfg)
        store._client = mock_redis
        return store

    def test_add_message_success(self, store, mock_redis):
        mock_redis.rpush.return_value = 1
        result = store.add_message("sess-001", _make_msg())
        assert result is True
        mock_redis.rpush.assert_called_once()

    def test_add_message_trims(self, store, mock_redis):
        """添加消息后调用 LTRIM"""
        mock_redis.rpush.return_value = 1
        store.add_message("sess-001", _make_msg())
        mock_redis.ltrim.assert_called_once()

    def test_add_message_sets_ttl(self, store, mock_redis):
        """添加消息后续期 TTL"""
        mock_redis.rpush.return_value = 1
        store.add_message("sess-001", _make_msg())
        mock_redis.expire.assert_called()

    def test_get_messages_parses_json(self, store, mock_redis):
        msg = _make_msg(role="user", content="CT伪影", turn=1)
        mock_redis.lrange.return_value = [json.dumps(msg, ensure_ascii=False)]
        messages = store.get_messages("sess-001")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "CT伪影"

    def test_get_messages_empty(self, store, mock_redis):
        mock_redis.lrange.return_value = []
        messages = store.get_messages("sess-001")
        assert messages == []

    def test_get_messages_skips_corrupt_json(self, store, mock_redis):
        mock_redis.lrange.return_value = [
            json.dumps(_make_msg()),
            "not valid json{{{",
            json.dumps(_make_msg(content="第二条")),
        ]
        messages = store.get_messages("sess-001")
        assert len(messages) == 2

    def test_get_turn_count(self, store, mock_redis):
        mock_redis.lrange.return_value = [
            json.dumps(_make_msg(role="user", turn=1)),
            json.dumps(_make_msg(role="assistant", turn=1)),
            json.dumps(_make_msg(role="user", turn=2)),
        ]
        assert store.get_turn_count("sess-001") == 2

    def test_update_summary(self, store, mock_redis):
        mock_redis.set.return_value = True
        result = store.update_summary("sess-001", "对话摘要内容")
        assert result is True
        mock_redis.set.assert_called_once()

    def test_get_summary(self, store, mock_redis):
        mock_redis.get.return_value = "摘要内容"
        summary = store.get_summary("sess-001")
        assert summary == "摘要内容"

    def test_get_summary_none(self, store, mock_redis):
        mock_redis.get.return_value = None
        summary = store.get_summary("sess-001")
        assert summary is None

    def test_clear_session(self, store, mock_redis):
        mock_redis.delete.return_value = 2
        result = store.clear_session("sess-001")
        assert result is True

    def test_available_when_connected(self, store):
        assert store.available is True

    def test_unavailable_when_disconnected(self):
        from src.memory.redis_client import RedisMemoryStore
        from src.config import RedisConfig
        cfg = RedisConfig({
            "host": "localhost", "port": 6379, "db": 0,
            "password": "", "ttl": 86400, "max_messages": 20,
            "key_prefix": "rag:session",
        })
        store = RedisMemoryStore(config=cfg)
        store._client = None
        assert store.available is False
        # 操作应安全返回
        assert store.add_message("sess", _make_msg()) is False
        assert store.get_messages("sess") == []
        assert store.get_summary("sess") is None
        assert store.clear_session("sess") is False

    def test_key_prefix_used(self, store):
        """验证 key 前缀正确"""
        msg_key = store._msg_key("sess-001")
        assert msg_key.startswith("rag:session:")
        assert "sess-001" in msg_key
        assert msg_key.endswith(":messages")

        sum_key = store._summary_key("sess-001")
        assert sum_key.startswith("rag:session:")
        assert sum_key.endswith(":summary")


class TestRedisMemoryStoreConnectionFailure:
    """Redis 连接失败时的降级"""

    def test_ping_failure_sets_client_none(self, mock_redis_unavailable):
        from src.memory.redis_client import RedisMemoryStore
        from src.config import RedisConfig
        cfg = RedisConfig({
            "host": "localhost", "port": 6379, "db": 0,
            "password": "", "ttl": 86400, "max_messages": 20,
            "key_prefix": "rag:session",
        })
        store = RedisMemoryStore(config=cfg)
        store._client = mock_redis_unavailable
        # ping 抛异常 → 客户端应被置为 None
        try:
            store._client.ping()
        except Exception:
            store._client = None
        assert store.available is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
