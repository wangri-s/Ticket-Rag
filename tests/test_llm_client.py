"""
测试 LLM 客户端 — src/llm/llm_client.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.llm_client import LLMClient


# ══════════════════════════════════════════════════════════
# get_current_user 和 get_cost_tracker 在 _call_api() 内部惰性导入:
#   from src.utils.rate_limiter import get_cost_tracker, get_current_user
# 因此 patch 目标必须是 src.utils.rate_limiter
# DashScope API 返回: resp.output.text = str, resp.usage = dict
# ══════════════════════════════════════════════════════════

def _make_resp(text="回答文本", input_tokens=500, output_tokens=200, total_tokens=700):
    """构造 DashScope Generation.call() 返回值"""
    resp = MagicMock()
    resp.status_code = 200
    resp.output = MagicMock()
    resp.output.text = text
    resp.usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    return resp


def _make_fail_resp(status_code=500):
    resp = MagicMock()
    resp.status_code = status_code
    return resp


class TestLLMClientGenerate:
    """LLM 生成 — 正常流程"""

    def test_generate_returns_text(self):
        with patch("dashscope.Generation.call", return_value=_make_resp()), \
             patch("src.utils.rate_limiter.get_current_user", return_value="user:test"), \
             patch("src.utils.rate_limiter.get_cost_tracker") as mock_tracker:
            mock_tracker.return_value = MagicMock()
            client = LLMClient()
            result = client.generate(user_message="CT伪影怎么排查？")
            assert "回答文本" in result

    def test_generate_with_system_prompt(self):
        with patch("dashscope.Generation.call", return_value=_make_resp()), \
             patch("src.utils.rate_limiter.get_current_user", return_value=""), \
             patch("src.utils.rate_limiter.get_cost_tracker") as mock_tracker:
            mock_tracker.return_value = MagicMock()
            client = LLMClient()
            result = client.generate(
                user_message="问题",
                system_prompt="你是医疗专家",
            )
            assert result is not None

    def test_generate_includes_question(self):
        with patch("dashscope.Generation.call", return_value=_make_resp(text="根据工单GD-001，建议检查。")), \
             patch("src.utils.rate_limiter.get_current_user", return_value=""), \
             patch("src.utils.rate_limiter.get_cost_tracker") as mock_tracker:
            mock_tracker.return_value = MagicMock()
            client = LLMClient()
            result = client.generate(user_message="透析液电导率偏高")
            assert isinstance(result, str)
            assert len(result) > 0


class TestLLMClientRetry:
    """重试机制"""

    def test_retry_on_failure_then_success(self):
        """前 2 次失败，第 3 次成功"""
        with patch("dashscope.Generation.call") as mock_call, \
             patch("src.utils.rate_limiter.get_current_user", return_value=""), \
             patch("src.utils.rate_limiter.get_cost_tracker") as mock_tracker:
            mock_tracker.return_value = MagicMock()
            mock_call.side_effect = [
                _make_fail_resp(500),
                _make_fail_resp(500),
                _make_resp(text="成功回答"),
            ]

            client = LLMClient()
            client.max_retries = 3
            result = client.generate(user_message="问题")
            assert result == "成功回答"
            assert mock_call.call_count == 3

    def test_all_retries_exhausted(self):
        """全部重试失败抛异常"""
        with patch("dashscope.Generation.call", return_value=_make_fail_resp(500)), \
             patch("src.utils.rate_limiter.get_current_user", return_value=""), \
             patch("src.utils.rate_limiter.get_cost_tracker") as mock_tracker:
            mock_tracker.return_value = MagicMock()
            client = LLMClient()
            client.max_retries = 3
            with pytest.raises(RuntimeError):
                client.generate(user_message="问题")


class TestLLMClientStream:
    """流式生成"""

    def test_generate_stream_yields_chunks(self):
        """流式逐个 yield token"""
        chunk1 = MagicMock()
        chunk1.status_code = 200
        chunk1.output = MagicMock()
        chunk1.output.text = "根据"

        chunk2 = MagicMock()
        chunk2.status_code = 200
        chunk2.output = MagicMock()
        chunk2.output.text = "工单"

        with patch("dashscope.Generation.call", return_value=iter([chunk1, chunk2])), \
             patch("src.utils.rate_limiter.get_current_user", return_value=""), \
             patch("src.utils.rate_limiter.get_cost_tracker") as mock_tracker:
            mock_tracker.return_value = MagicMock()
            client = LLMClient()
            stream = client.generate_stream(user_message="问题")
            chunks = list(stream)
            assert len(chunks) == 2
            assert chunks[0] == "根据"
            assert chunks[1] == "工单"


class TestLLMClientCostReporting:
    """成本上报"""

    def test_cost_reported_after_success(self):
        """LLM 调用成功后上报 token 消耗"""
        with patch("dashscope.Generation.call", return_value=_make_resp()), \
             patch("src.utils.rate_limiter.get_current_user", return_value="user:test"), \
             patch("src.utils.rate_limiter.get_cost_tracker") as mock_tracker:

            tracker = MagicMock()
            mock_tracker.return_value = tracker

            client = LLMClient()
            client.generate(user_message="问题")
            tracker.record_usage.assert_called_once()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
