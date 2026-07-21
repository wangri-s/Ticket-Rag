"""
测试三级记忆编排器 — src/memory/memory_manager.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Helpers ─────────────────────────────────────

def _make_msg(role="user", content="测试问题", turn=1):
    return {"role": role, "content": content, "turn": turn, "timestamp": "2026-01-01T08:00:00"}


# ══════════════════════════════════════════════════════════

class TestMemoryManagerWithMocks:
    """Mock 全部依赖，测试编排逻辑"""

    @pytest.fixture
    def manager(self):
        """创建 Mock 全部子组件"""
        with patch("src.memory.memory_manager.RedisMemoryStore") as mock_redis, \
             patch("src.memory.memory_manager.MySQLMemoryStore") as mock_mysql, \
             patch("src.memory.memory_manager.KafkaProducerClient") as mock_kafka, \
             patch("src.memory.memory_manager.ConversationSummarizer") as mock_summ:

            # Redis — 可用
            redis_inst = mock_redis.return_value
            redis_inst.available = True
            redis_inst.add_message.return_value = True
            redis_inst.get_messages.return_value = []
            redis_inst.get_summary.return_value = None
            redis_inst.get_turn_count.return_value = 0
            redis_inst.update_summary.return_value = True
            redis_inst.clear_session.return_value = True

            # MySQL — 可用
            mysql_inst = mock_mysql.return_value
            mysql_inst.available = True
            mysql_inst.insert_message.return_value = True
            mysql_inst.insert_summary.return_value = True

            # Kafka — 可用
            kafka_inst = mock_kafka.return_value
            kafka_inst.available = True
            kafka_inst.send_message.return_value = True
            kafka_inst.send_summary.return_value = True

            # Summarizer — 可用
            summ_inst = mock_summ.return_value
            summ_inst.enabled = True
            summ_inst.trigger_turns = 5
            summ_inst.should_summarize.return_value = False
            summ_inst.generate.return_value = None

            from src.memory.memory_manager import MemoryManager
            mgr = MemoryManager()
            # 替换为 mock 实例
            mgr._redis = redis_inst
            mgr._mysql = mysql_inst
            mgr._kafka = kafka_inst
            mgr._summarizer = summ_inst

            yield mgr

    # ── add_message ─────────────────────────────

    def test_add_message_writes_to_redis(self, manager):
        result = manager.add_message("sess-001", "user", "CT伪影怎么排查？")
        assert "turn" in result
        manager._redis.add_message.assert_called_once()

    def test_add_message_sends_to_kafka(self, manager):
        manager.add_message("sess-001", "user", "问题")
        manager._kafka.send_message.assert_called_once()

    def test_add_message_increments_turn(self, manager):
        manager._redis.get_turn_count.return_value = 2
        result = manager.add_message("sess-001", "user", "新问题")
        assert result["turn"] == 3

    def test_assistant_turn_does_not_increment(self, manager):
        """assistant 消息不增加 turn 计数"""
        manager._redis.get_turn_count.return_value = 1
        result = manager.add_message("sess-001", "assistant", "回答")
        assert result["turn"] == 1

    # ── get_context ─────────────────────────────

    def test_get_context_returns_structure(self, manager):
        ctx = manager.get_context("sess-001")
        assert "messages" in ctx
        assert "summary" in ctx
        assert "turn_count" in ctx
        assert "has_memory" in ctx

    def test_get_context_empty(self, manager):
        manager._redis.get_messages.return_value = []
        ctx = manager.get_context("sess-001")
        assert ctx["has_memory"] is False
        assert ctx["turn_count"] == 0

    def test_get_context_with_history(self, manager):
        manager._redis.get_messages.return_value = [
            _make_msg("user", "问题1", 1),
            _make_msg("assistant", "回答1", 1),
        ]
        ctx = manager.get_context("sess-001")
        assert ctx["has_memory"] is True
        assert ctx["turn_count"] == 1

    # ── build_memory_prompt ────────────────────

    def test_build_memory_prompt_empty(self, manager):
        prompt = manager.build_memory_prompt("sess-001")
        assert prompt == ""

    def test_build_memory_prompt_with_messages(self, manager):
        manager._redis.get_messages.return_value = [
            _make_msg("user", "CT伪影怎么排查？", 1),
        ]
        manager._redis.get_summary.return_value = None
        prompt = manager.build_memory_prompt("sess-001")
        assert "最近对话" in prompt
        assert "CT伪影" in prompt

    def test_build_memory_prompt_with_summary(self, manager):
        manager._redis.get_messages.return_value = [
            _make_msg("user", "新问题", 5),
        ]
        manager._redis.get_summary.return_value = "前4轮讨论了CT伪影问题"
        prompt = manager.build_memory_prompt("sess-001")
        assert "历史对话摘要" in prompt
        assert "CT伪影" in prompt

    def test_build_memory_prompt_truncates_long_content(self, manager):
        """每条消息截断到 300 字符"""
        long_content = "X" * 500
        manager._redis.get_messages.return_value = [
            _make_msg("user", long_content, 1),
        ]
        prompt = manager.build_memory_prompt("sess-001")
        # 截断后不应含完整 500 字符
        assert long_content not in prompt

    # ── 摘要触发 ──────────────────────────────

    def test_summary_triggered_on_5th_turn(self, manager):
        """第 5 轮 assistant 消息触发摘要"""
        manager._summarizer.should_summarize.return_value = True
        manager._summarizer.generate.return_value = "摘要: 前5轮讨论内容"
        manager._redis.get_messages.return_value = [
            _make_msg("user", f"问题{i}", i) for i in range(1, 6)
        ]
        result = manager._maybe_summarize("sess-001", 5)
        assert result is not None
        assert "摘要" in result

    def test_summary_not_triggered_before_5(self, manager):
        """第 3 轮不触发"""
        manager._summarizer.should_summarize.return_value = False
        result = manager._maybe_summarize("sess-001", 3)
        assert result is None

    # ── clear_session ───────────────────────────

    def test_clear_session(self, manager):
        result = manager.clear_session("sess-001")
        assert result is True
        manager._redis.clear_session.assert_called_once_with("sess-001")

    # ── available ───────────────────────────────

    def test_available_depends_on_redis(self, manager):
        assert manager.available is True
        manager._redis.available = False
        assert manager.available is False


class TestMemoryManagerDegradation:
    """降级场景"""

    def test_kafka_unavailable_falls_back_to_mysql(self):
        """Kafka 挂 → 直写 MySQL"""
        with patch("src.memory.memory_manager.RedisMemoryStore") as mock_redis, \
             patch("src.memory.memory_manager.MySQLMemoryStore") as mock_mysql, \
             patch("src.memory.memory_manager.KafkaProducerClient") as mock_kafka, \
             patch("src.memory.memory_manager.ConversationSummarizer") as mock_summ:

            redis_inst = mock_redis.return_value
            redis_inst.available = True
            redis_inst.get_turn_count.return_value = 0
            redis_inst.add_message.return_value = True

            mysql_inst = mock_mysql.return_value
            mysql_inst.available = True

            kafka_inst = mock_kafka.return_value
            kafka_inst.available = False  # Kafka 挂了

            summ_inst = mock_summ.return_value
            summ_inst.enabled = True
            summ_inst.trigger_turns = 5
            summ_inst.should_summarize.return_value = False

            from src.memory.memory_manager import MemoryManager
            mgr = MemoryManager()
            mgr._redis = redis_inst
            mgr._mysql = mysql_inst
            mgr._kafka = kafka_inst
            mgr._summarizer = summ_inst

            mgr.add_message("sess-001", "user", "问题")
            # Kafka 不可用 → 不应该调用 send_message
            kafka_inst.send_message.assert_not_called()
            # 应该直写 MySQL
            mysql_inst.insert_message.assert_called_once()

    def test_both_kafka_and_mysql_unavailable(self):
        """Kafka + MySQL 都挂 → 只写 Redis"""
        with patch("src.memory.memory_manager.RedisMemoryStore") as mock_redis, \
             patch("src.memory.memory_manager.MySQLMemoryStore") as mock_mysql, \
             patch("src.memory.memory_manager.KafkaProducerClient") as mock_kafka, \
             patch("src.memory.memory_manager.ConversationSummarizer") as mock_summ:

            redis_inst = mock_redis.return_value
            redis_inst.available = True
            redis_inst.get_turn_count.return_value = 0
            redis_inst.add_message.return_value = True

            mysql_inst = mock_mysql.return_value
            mysql_inst.available = False

            kafka_inst = mock_kafka.return_value
            kafka_inst.available = False

            summ_inst = mock_summ.return_value
            summ_inst.enabled = True
            summ_inst.trigger_turns = 5
            summ_inst.should_summarize.return_value = False

            from src.memory.memory_manager import MemoryManager
            mgr = MemoryManager()
            mgr._redis = redis_inst
            mgr._mysql = mysql_inst
            mgr._kafka = kafka_inst
            mgr._summarizer = summ_inst

            # 不应崩溃
            result = mgr.add_message("sess-001", "user", "问题")
            assert isinstance(result, dict)
            mysql_inst.insert_message.assert_not_called()
            kafka_inst.send_message.assert_not_called()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
