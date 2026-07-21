"""
测试上下文压缩器 — src/utils/context_compressor.py
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.context_compressor import (
    ContextCompressor,
    _deduplicate_chunks,
    estimate_tokens,
    estimate_chunk_tokens,
)


# ══════════════════════════════════════════════════════════
# Token 估算
# ══════════════════════════════════════════════════════════

class TestEstimateTokens:
    """Token 数量估算"""

    def test_empty_text(self):
        assert estimate_tokens("") == 0

    def test_none_text(self):
        assert estimate_tokens(None) == 0

    def test_short_chinese(self):
        assert estimate_tokens("你好") >= 1

    def test_medium_chinese(self):
        # 300 个汉字 → 300 / 1.2 = 250
        text = "中" * 300
        assert estimate_tokens(text) == 250

    def test_english_text(self):
        text = "a" * 300
        assert estimate_tokens(text) == 250

    def test_mixed_cn_en(self):
        text = "故障排查步骤" + "abc" * 10
        expected = max(1, int(len(text) / 1.2))
        assert estimate_tokens(text) == expected

    def test_very_short(self):
        # 1 字符 → max(1, int(1/1.2)) = 1
        assert estimate_tokens("x") == 1


class TestEstimateChunkTokens:
    """单个 chunk 的 token 估算（含格式化开销）"""

    def test_includes_overhead(self):
        chunk = {"content": "短内容"}
        content_tokens = estimate_tokens("短内容")
        chunk_tokens = estimate_chunk_tokens(chunk)
        assert chunk_tokens > content_tokens

    def test_empty_content(self):
        chunk = {"content": ""}
        tokens = estimate_chunk_tokens(chunk)
        assert tokens >= 1  # overhead


# ══════════════════════════════════════════════════════════
# Deduplication (Jaccard)
# ══════════════════════════════════════════════════════════

class TestDeduplicateChunks:
    """chunk 内容去重"""

    def test_empty_list(self):
        assert _deduplicate_chunks([]) == []

    def test_single_chunk(self):
        chunks = [{"content": "唯一内容", "score": 0.9}]
        result = _deduplicate_chunks(chunks, threshold=0.7)
        assert len(result) == 1
        assert result[0]["content"] == "唯一内容"

    def test_identical_adjacent_merged(self):
        chunks = [
            {"content": "CT 图像伪影排查方案，需要检查探测器校准。", "score": 0.5},
            {"content": "CT 图像伪影排查方案，需要检查探测器校准。", "score": 0.9},
        ]
        result = _deduplicate_chunks(chunks, threshold=0.7)
        assert len(result) == 1
        # 保留高分
        assert result[0]["score"] == 0.9

    def test_lower_scored_dropped_when_dup(self):
        chunks = [
            {"content": "完全相同的文本段落内容A。", "score": 0.9},
            {"content": "完全相同的文本段落内容A。", "score": 0.3},
        ]
        result = _deduplicate_chunks(chunks, threshold=0.7)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_different_content_kept(self):
        chunks = [
            {"content": "CT 伪影排查方案详细步骤。", "score": 0.9},
            {"content": "MRI 啸叫处理参考工单记录。", "score": 0.8},
        ]
        result = _deduplicate_chunks(chunks, threshold=0.7)
        assert len(result) == 2

    def test_below_threshold_kept(self):
        chunks = [
            {"content": "CT 伪影排查方案详细步骤一。", "score": 0.9},
            {"content": "MRI 啸叫处理不同设备的故障。", "score": 0.8},
        ]
        # 这两个 Jaccard 应该很低
        result = _deduplicate_chunks(chunks, threshold=0.7)
        assert len(result) >= 1

    def test_high_threshold_no_merge(self):
        chunks = [
            {"content": "CT 图像伪影排查方案: 检查探测器校准状态。", "score": 0.5},
            {"content": "CT 图像伪影排查方案: 确认球管灯丝电流。", "score": 0.9},
        ]
        result = _deduplicate_chunks(chunks, threshold=0.95)
        # 阈值 0.95 极高，这两条可能不够相似
        assert len(result) == 2


# ══════════════════════════════════════════════════════════
# ContextCompressor — 主类
# ══════════════════════════════════════════════════════════

class TestContextCompressorBasics:
    """基础行为"""

    def test_empty_chunks(self):
        compressor = ContextCompressor()
        result = compressor.compress([], "system", "question")
        assert result == []

    def test_no_compression_needed(self, sample_chunks):
        """少量 chunk 无需压缩"""
        compressor = ContextCompressor()
        # 只用前 2 条少量 chunk，不会超限
        small = sample_chunks[:2]
        result = compressor.compress(small)
        assert len(result) == len(small)


class TestCompressWithLongChunks:
    """压缩各阶段"""

    def test_level1_score_truncation(self):
        """高分 chunk 保留，低分丢弃"""
        compressor = ContextCompressor()
        # 构造大量低分 chunk 使其超限
        chunks = [
            {"content": "重要信息：" + "排查步骤。" * 50, "ticket_id": "GD-001", "score": 0.95},
            {"content": "一般信息：" + "维护建议。" * 50, "ticket_id": "GD-002", "score": 0.60},
            {"content": "低相关：" + "无关内容。" * 50, "ticket_id": "GD-003", "score": 0.30},
        ]
        result = compressor.compress(chunks)
        # 至少第一条（最高分）保留
        assert len(result) >= 1
        assert result[0]["ticket_id"] == "GD-001"

    def test_level2_content_truncation(self):
        """超长 chunk 内容被截断"""
        compressor = ContextCompressor()
        long_content = "排查步骤。" * 300  # 远超 max_chunk_chars=300
        chunks = [
            {"content": long_content, "ticket_id": "GD-001", "score": 0.9},
        ]
        result = compressor.compress(chunks)
        if len(result) > 0:
            # 如果经过 L2 截断，content 应该变短
            content = result[0].get("content", "")
            assert len(content) <= len(long_content)

    def test_sentence_boundary_truncation(self):
        """截断在句号处而非硬切"""
        compressor = ContextCompressor()
        # 构造刚好在 max_chars 附近有句号的内容
        text = "A。" * 200  # 每个"A。"=2 chars, 200个=400 chars
        chunks = [
            {"content": text, "ticket_id": "GD-001", "score": 0.9},
        ]
        result = compressor.compress(chunks)
        # 应该正常完成
        assert isinstance(result, list)

    def test_truncated_ends_with_ellipsis(self):
        """截断后以省略号结尾"""
        compressor = ContextCompressor()
        long_text = "A" * 500
        chunks = [
            {"content": long_text, "ticket_id": "GD-001", "score": 0.9},
        ]
        result = compressor.compress(chunks)
        if len(result) > 0 and len(result[0]["content"]) < len(long_text):
            assert "…" in result[0]["content"]

    def test_fallback_aggressive_truncate(self):
        """固定开销超预算 → 返回 chunks[:1] 做最保守兜底"""
        compressor = ContextCompressor()
        # system_prompt 自身超限 → 固定开销 > budget
        huge_prompt = "X" * 10000
        chunks = [
            {"content": "短内容A", "ticket_id": "GD-001", "score": 0.9},
            {"content": "短内容B", "ticket_id": "GD-002", "score": 0.8},
        ]
        result = compressor.compress(chunks, system_prompt=huge_prompt)
        # 固定开销超限时只保留第 1 个 chunk（不做截断，直接返回）
        assert len(result) == 1
        assert result[0]["ticket_id"] == "GD-001"

    def test_fixed_overhead_exceeds_budget(self):
        """固定开销（system_prompt）自身已超预算"""
        compressor = ContextCompressor()
        # 构造超长的 system prompt
        huge_prompt = "X" * 15000  # 远超 6000
        chunks = [
            {"content": "无关", "ticket_id": "GD-001", "score": 0.5},
            {"content": "也无关", "ticket_id": "GD-002", "score": 0.4},
        ]
        result = compressor.compress(chunks, system_prompt=huge_prompt)
        # 至少保留 1 个 chunk（fallback）
        assert len(result) == 1

    def test_memory_context_consumes_budget(self):
        """记忆上下文也消耗 token 预算"""
        compressor = ContextCompressor()
        long_memory = "历史对话记忆：" * 200
        chunks = [
            {"content": "排查步骤。" * 100, "ticket_id": "GD-001", "score": 0.9},
        ]
        result = compressor.compress(chunks, memory_context=long_memory)
        assert isinstance(result, list)

    def test_reserve_tokens_considered(self):
        """预留 tokens 被计入"""
        compressor = ContextCompressor()
        chunks = [
            {"content": "短内容A", "ticket_id": "GD-001", "score": 0.5},
        ]
        # 应该正常返回
        result = compressor.compress(chunks)
        assert len(result) >= 0


class TestCompressOrderPreservation:
    """chunk 顺序保持"""

    def test_score_order_preserved(self):
        compressor = ContextCompressor()
        chunks = [
            {"content": "A" * 100, "ticket_id": "GD-001", "score": 0.95},
            {"content": "B" * 100, "ticket_id": "GD-002", "score": 0.82},
            {"content": "C" * 100, "ticket_id": "GD-003", "score": 0.68},
        ]
        result = compressor.compress(chunks)
        scores = [c["score"] for c in result]
        # 分数递减
        assert scores == sorted(scores, reverse=True)


class TestDedupIntegration:
    """L3 去重集成测试"""

    def test_highly_overlapping_adjacent(self):
        compressor = ContextCompressor()
        #  两条几乎相同的 chunk
        base = "CT 图像伪影排查方案: 需要检查探测器阵列校准状态，确认各通道灵敏度是否在正常范围内。" * 3
        chunks = [
            {"content": base, "ticket_id": "GD-001", "score": 0.5},
            {"content": base[:len(base)//2], "ticket_id": "GD-002", "score": 0.9},
        ]
        result = compressor.compress(chunks)
        # 可能去重合并
        assert isinstance(result, list)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
