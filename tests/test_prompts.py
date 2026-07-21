"""
测试 Prompt 构建 — src/llm/prompts.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.prompts import (
    build_full_prompt,
    build_rag_prompt,
    build_json_prompt,
    format_chunks,
    get_fallback_answer,
    get_json_fallback,
    get_system_prompt,
)


class TestFormatChunks:
    """检索结果格式化"""

    def test_empty_list(self):
        result = format_chunks([])
        assert "无参考工单" in result

    def test_single_chunk(self):
        chunks = [{
            "content": "CT 伪影排查：探测器校准。",
            "ticket_id": "GD-2026-03001",
            "source": "test.txt",
            "chunk_index": 0,
        }]
        result = format_chunks(chunks)
        assert "工单 1" in result
        assert "GD-2026-03001" in result
        assert "test.txt" in result
        assert "CT 伪影" in result

    def test_multiple_chunks(self):
        chunks = [
            {"content": "内容A。", "ticket_id": "GD-001", "source": "a.txt", "chunk_index": 0},
            {"content": "内容B。", "ticket_id": "GD-002", "source": "b.txt", "chunk_index": 0},
        ]
        result = format_chunks(chunks)
        assert "工单 1" in result
        assert "工单 2" in result
        assert "GD-001" in result
        assert "GD-002" in result

    def test_separator_chars_stripped_from_content(self):
        """format_chunks 从 chunk content 中清除 U+2550/U+2500
        注意: 格式化头 "─── 工单 N ───" 自身也含 '─'，所以只查 '═' 不在结果中"""
        DH = chr(0x2550)  # BOX DRAWINGS DOUBLE HORIZONTAL
        LH = chr(0x2500)  # BOX DRAWINGS LIGHT HORIZONTAL
        chunks = [{
            "content": f"{DH * 5} 标题内容 {LH * 5}\n正文内容。",
            "ticket_id": "GD-001",
            "source": "test.txt",
            "chunk_index": 0,
        }]
        result = format_chunks(chunks)
        # U+2550 在格式化头中不使用 → 应被完全清除
        assert chr(0x2550) not in result
        # 正文内容保留
        assert "正文内容" in result

    def test_missing_ticket_id_defaults_to_question_mark(self):
        chunks = [{
            "content": "无编号内容。",
            "ticket_id": "",
            "source": "test.txt",
            "chunk_index": 0,
        }]
        result = format_chunks(chunks)
        assert "编号:" in result


class TestBuildRagPrompt:
    """RAG text Prompt"""

    def test_normal_prompt(self):
        chunks = [{
            "content": "CT 伪影排查。",
            "ticket_id": "GD-001",
            "source": "test.txt",
            "chunk_index": 0,
        }]
        prompt = build_rag_prompt("CT 伪影怎么办？", chunks)
        assert prompt is not None
        assert "CT 伪影怎么办？" in prompt
        assert "GD-001" in prompt

    def test_empty_chunks_returns_none(self):
        prompt = build_rag_prompt("问题", [])
        assert prompt is None

    def test_ticket_id_filter_injected(self):
        chunks = [{
            "content": "内容。",
            "ticket_id": "GD-001",
            "source": "test.txt",
            "chunk_index": 0,
        }]
        prompt = build_rag_prompt("问题", chunks, ticket_id_filter="GD-001")
        assert "GD-001" in prompt


class TestBuildJsonPrompt:
    """RAG JSON Prompt"""

    def test_normal_json_prompt(self):
        chunks = [{
            "content": "CT 伪影排查。",
            "ticket_id": "GD-001",
            "source": "test.txt",
            "chunk_index": 0,
        }]
        prompt = build_json_prompt("CT 伪影怎么办？", chunks)
        assert prompt is not None
        assert "JSON" in prompt

    def test_empty_chunks_returns_none(self):
        prompt = build_json_prompt("问题", [])
        assert prompt is None


class TestBuildFullPrompt:
    """build_full_prompt 快捷函数"""

    def test_text_format(self):
        chunks = [{
            "content": "CT 伪影是常见故障。",
            "ticket_id": "GD-001",
            "source": "test.txt",
            "chunk_index": 0,
        }]
        sys_prompt, user_msg = build_full_prompt("CT 问题", chunks, output_format="text")
        assert sys_prompt is not None
        assert user_msg is not None
        assert len(sys_prompt) > 100

    def test_json_format(self):
        chunks = [{
            "content": "CT 伪影是常见故障。",
            "ticket_id": "GD-001",
            "source": "test.txt",
            "chunk_index": 0,
        }]
        sys_prompt, user_msg = build_full_prompt("CT 问题", chunks, output_format="json")
        assert sys_prompt is not None
        assert user_msg is not None

    def test_empty_chunks_text(self):
        sys_prompt, user_msg = build_full_prompt("问题", [])
        assert sys_prompt is None
        assert user_msg is None

    def test_empty_chunks_json(self):
        sys_prompt, user_msg = build_full_prompt("问题", [], output_format="json")
        assert sys_prompt is None
        assert user_msg is None


class TestFallback:
    """兜底答案"""

    def test_text_fallback_not_empty(self):
        fb = get_fallback_answer()
        assert len(fb) > 0
        assert "未找到" in fb

    def test_json_fallback_is_valid_json(self):
        fb = get_json_fallback()
        data = json.loads(fb)
        assert isinstance(data, dict)
        assert "has_reference" in data
        assert data["has_reference"] is False
        assert "analysis" in data
        assert "references" in data
        assert "recommendations" in data

    def test_json_fallback_recommendations(self):
        fb = get_json_fallback()
        data = json.loads(fb)
        steps = data["recommendations"]["steps"]
        assert len(steps) > 0
        assert isinstance(steps, list)


class TestSystemPrompt:
    """System Prompt"""

    def test_not_empty(self):
        sp = get_system_prompt()
        assert len(sp) > 100

    def test_contains_medical_context(self):
        sp = get_system_prompt()
        assert "医疗" in sp or "设备" in sp


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
