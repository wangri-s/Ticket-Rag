"""
测试 API 端点 — src/api/ask.py

使用 FastAPI TestClient + Mock RAGChain。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient


# ── Fixture: 创建 TestClient ────────────────────

@pytest.fixture
def client():
    """返回 Mock 了 RAGChain + 限流的 TestClient"""
    mock_chain = MagicMock()
    mock_chain.ask.return_value = {
        "question": "CT伪影怎么排查？",
        "answer": "根据工单 GD-2026-03001，建议检查探测器校准状态。",
        "sources": [
            {
                "ticket_id": "GD-2026-03001",
                "content": "CT 伪影排查方案",
                "source": "test.txt",
                "score": 0.95,
            }
        ],
        "mode": "hybrid",
        "has_answer": True,
        "output_format": "text",
        "session_id": None,
        "from_cache": False,
    }

    with patch("src.api.ask.get_rate_limiter") as mock_limiter, \
         patch("src.api.ask.get_cost_tracker") as mock_tracker:
        # Mock RateLimiter — 全部放行
        limiter_inst = mock_limiter.return_value
        limiter_inst.check_global.return_value = None
        limiter_inst.check_user.return_value = None
        limiter_inst.rate_limit_headers = {
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "59",
        }

        # Mock CostTracker — 放行
        tracker_inst = mock_tracker.return_value
        tracker_inst.check_budget.return_value = None

        # 直接注入 Mock chain 到模块级变量
        import src.api.ask as ask_mod
        ask_mod._chain = mock_chain

        from src.main import app
        with TestClient(app) as tc:
            yield tc


# ══════════════════════════════════════════════════════════
# POST /api/ask
# ══════════════════════════════════════════════════════════

class TestAskEndpoint:
    """RAG 问答端点"""

    def test_normal_request_returns_200(self, client):
        resp = client.post("/api/ask", json={
            "question": "CT伪影怎么排查？",
            "mode": "hybrid",
            "top_k": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert data["has_answer"] is True

    def test_response_has_latency(self, client):
        resp = client.post("/api/ask", json={
            "question": "CT伪影怎么排查？",
        })
        data = resp.json()
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_response_includes_rate_limit_headers(self, client):
        resp = client.post("/api/ask", json={
            "question": "CT伪影怎么排查？",
        })
        assert "x-ratelimit-limit" in resp.headers or "X-RateLimit-Limit" in resp.headers

    def test_mode_keyword(self, client):
        resp = client.post("/api/ask", json={
            "question": "测试",
            "mode": "keyword",
        })
        assert resp.status_code == 200

    def test_mode_semantic(self, client):
        resp = client.post("/api/ask", json={
            "question": "测试",
            "mode": "semantic",
        })
        assert resp.status_code == 200

    def test_with_session_id(self, client):
        resp = client.post("/api/ask", json={
            "question": "CT伪影",
            "session_id": "sess-test-001",
        })
        assert resp.status_code == 200
        data = resp.json()
        # Mock chain 返回 session_id=None; 验证字段存在即可
        assert "session_id" in data

    def test_json_output_format(self, client):
        resp = client.post("/api/ask", json={
            "question": "CT伪影",
            "output_format": "json",
        })
        assert resp.status_code == 200

    def test_with_ticket_id_filter(self, client):
        resp = client.post("/api/ask", json={
            "question": "问题",
            "ticket_id": "GD-2026-03001",
        })
        assert resp.status_code == 200

    def test_with_device_type_filter(self, client):
        resp = client.post("/api/ask", json={
            "question": "问题",
            "device_type": "CT 机",
        })
        assert resp.status_code == 200

    def test_with_rerank(self, client):
        resp = client.post("/api/ask", json={
            "question": "问题",
            "rerank": True,
        })
        assert resp.status_code == 200

    def test_with_query_expansion(self, client):
        resp = client.post("/api/ask", json={
            "question": "问题",
            "query_expansion": True,
        })
        assert resp.status_code == 200


class TestAskValidation:
    """请求校验"""

    def test_empty_question_rejected(self, client):
        resp = client.post("/api/ask", json={
            "question": "",
        })
        assert resp.status_code == 422

    def test_missing_question_rejected(self, client):
        resp = client.post("/api/ask", json={})
        assert resp.status_code == 422

    def test_question_too_long(self, client):
        resp = client.post("/api/ask", json={
            "question": "X" * 2001,
        })
        assert resp.status_code == 422

    def test_invalid_mode_rejected(self, client):
        resp = client.post("/api/ask", json={
            "question": "test",
            "mode": "invalid_mode",
        })
        assert resp.status_code == 422

    def test_invalid_output_format_rejected(self, client):
        resp = client.post("/api/ask", json={
            "question": "test",
            "output_format": "xml",
        })
        assert resp.status_code == 422

    def test_top_k_out_of_range(self, client):
        resp = client.post("/api/ask", json={
            "question": "test",
            "top_k": 100,  # 超过 20
        })
        assert resp.status_code == 422

    def test_top_k_zero(self, client):
        resp = client.post("/api/ask", json={
            "question": "test",
            "top_k": 0,  # < 1
        })
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════
# POST /api/ask/stream
# ══════════════════════════════════════════════════════════

class TestAskStreamEndpoint:
    """流式端点"""

    def test_stream_returns_200(self, client):
        resp = client.post("/api/ask/stream", json={
            "question": "CT伪影怎么排查？",
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_has_sse_headers(self, client):
        resp = client.post("/api/ask/stream", json={
            "question": "测试",
        })
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"

    def test_stream_contains_sources_event(self, client):
        resp = client.post("/api/ask/stream", json={
            "question": "CT伪影",
        })
        body = resp.text
        assert "event: sources" in body

    def test_stream_contains_delta_event(self, client):
        resp = client.post("/api/ask/stream", json={
            "question": "CT伪影",
        })
        body = resp.text
        assert "event: delta" in body

    def test_stream_contains_done_event(self, client):
        resp = client.post("/api/ask/stream", json={
            "question": "CT伪影",
        })
        body = resp.text
        assert "event: done" in body

    def test_stream_validation_rejects_invalid(self, client):
        resp = client.post("/api/ask/stream", json={
            "question": "",
        })
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════
# GET /api/ask/health
# ══════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """健康检查"""

    def test_health_ready_state(self, client):
        """chain 已注入 mock → ready"""
        resp = client.get("/api/ask/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"


# ══════════════════════════════════════════════════════════
# Rate Limiting 集成
# ══════════════════════════════════════════════════════════

class TestRateLimitIntegration:
    """限流集成 — 429 响应"""

    def test_rate_limit_returns_429(self):
        """超限时返回 429"""
        from src.utils.rate_limiter import RateLimitExceeded

        mock_chain = MagicMock()
        import src.api.ask as ask_mod
        ask_mod._chain = mock_chain

        with patch("src.api.ask.get_rate_limiter") as mock_limiter, \
             patch("src.api.ask.get_cost_tracker") as mock_tracker:

            limiter_inst = mock_limiter.return_value
            limiter_inst.check_global.side_effect = RateLimitExceeded(
                retry_after=5.0, limit=60, remaining=0,
            )

            tracker_inst = mock_tracker.return_value
            tracker_inst.check_budget.return_value = None

            from src.main import app
            with TestClient(app) as tc:
                resp = tc.post("/api/ask", json={"question": "test"})
                assert resp.status_code == 429

        ask_mod._chain = None

    def test_budget_exceeded_returns_429(self):
        """预算超限返回 429"""
        from src.utils.rate_limiter import BudgetExceeded

        mock_chain = MagicMock()
        import src.api.ask as ask_mod
        ask_mod._chain = mock_chain

        with patch("src.api.ask.get_rate_limiter"), \
             patch("src.api.ask.get_cost_tracker") as mock_tracker:

            tracker_inst = mock_tracker.return_value
            tracker_inst.check_budget.side_effect = BudgetExceeded(
                "token 已用尽", retry_after=3600,
            )

            from src.main import app
            with TestClient(app) as tc:
                resp = tc.post("/api/ask", json={"question": "test"})
                assert resp.status_code == 429

        ask_mod._chain = None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
