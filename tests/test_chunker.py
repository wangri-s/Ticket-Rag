"""
测试文本分割器 src/ingestion/chunker.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from src.ingestion.chunker import (
    MedicalWorkOrderChunker,
    extract_ticket_id,
)


class TestTicketIdExtraction:
    """工单编号提取"""

    def test_valid_ticket_id(self):
        text = "【工单编号】GD-2026-03001\n【故障现象】CT 伪影"
        assert extract_ticket_id(text) == "GD-2026-03001"

    def test_no_ticket_id(self):
        text = "这是一段没有编号的文字"
        assert extract_ticket_id(text) is None

    def test_multiple_same_type(self):
        text = "GD-2026-03005 和 GD-2026-03006 都出现了"
        assert extract_ticket_id(text) == "GD-2026-03005"  # 返回第一个

    def test_partial_match(self):
        assert extract_ticket_id("GD-2026-03") is None


class TestChunker:
    """分块器核心功能"""

    @property
    def chunker(self):
        return MedicalWorkOrderChunker()

    def test_empty_text(self):
        assert self.chunker.split_text("") == []
        assert self.chunker.split_text("   ") == []

    def test_short_text_unchanged(self):
        # 文本需超过 min_chunk_length (20) 才不会被过滤
        text = "CT 扫描偶发伪影表现为条状低密度阴影需排查探测器阵列问题"
        result = self.chunker.split_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_is_split(self):
        text = "故障排查步骤。" * 200  # 远超 500 字符
        result = self.chunker.split_text(text)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 500 + 50  # chunk_size + overlap 缓冲

    def test_min_chunk_length_filtered(self):
        # 极短文本应被过滤
        text = "AB"  # < 20
        result = self.chunker.split_text(text)
        assert result == []

    def test_separator_priority(self):
        """验证工单分隔符优先于短分隔符：文本超 chunk_size 时才触发分割"""
        # 复制多次使文本超过 500 字符
        unit = (
            "【工单编号】GD-2026-03001\n【故障现象】CT 伪影排查方案。" * 3
            + "\n══════════════════════\n"
            + "【工单编号】GD-2026-03002\n【故障现象】MRI 啸叫处理步骤。" * 3
        )
        text = unit * 4  # 远超过 500 字符
        result = self.chunker.split_text(text)
        # 超长文本被分割，且含工单分隔符，应产生多块
        assert len(result) >= 2

    def test_split_documents_metadata(self):
        docs = [
            Document(
                page_content="【工单编号】GD-2026-03001\n【故障现象】CT 伪影。" * 5,
                metadata={"source": "test.txt", "page": 1},
            )
        ]
        result = self.chunker.split_documents(docs)
        assert len(result) >= 1
        for doc in result:
            assert "chunk_index" in doc.metadata
            assert doc.metadata["source"] == "test.txt"

    def test_ticket_id_injected_to_metadata(self):
        docs = [
            Document(
                page_content="【工单编号】GD-2026-03001\n【故障现象】CT 图像伪影。" * 5,
                metadata={"source": "test.txt"},
            )
        ]
        result = self.chunker.split_documents(docs)
        ids = [d.metadata.get("ticket_id") for d in result]
        assert "GD-2026-03001" in ids


class TestSplitWithStats:
    """分块统计"""

    @property
    def chunker(self):
        return MedicalWorkOrderChunker()

    def test_stats_returns_all_fields(self):
        text = "【工单编号】GD-2026-03001\n【故障现象】CT 伪影的排查与处理方案。" * 15
        stats = self.chunker.split_with_stats(text)
        assert "chunks" in stats
        assert "count" in stats
        assert "avg_size" in stats
        assert "min_size" in stats
        assert "max_size" in stats
        assert "ticket_ids" in stats

    def test_stats_max_under_limit(self):
        text = "【工单编号】GD-2026-03001\n" + ("故障现象描述内容。" * 200)
        stats = self.chunker.split_with_stats(text)
        assert stats["max_size"] <= 500 + 50


class TestGroupByTicket:
    """按工单分组"""

    def test_grouping(self):
        chunker = MedicalWorkOrderChunker()
        docs = [
            Document(
                page_content="【工单编号】GD-2026-03001\n【故障现象】CT 伪影。" * 5,
                metadata={"ticket_id": "GD-2026-03001"},
            ),
            Document(
                page_content="【工单编号】GD-2026-03002\n【故障现象】MRI 啸叫。" * 5,
                metadata={"ticket_id": "GD-2026-03002"},
            ),
            Document(
                page_content="通用设备维护说明。" * 10,
                metadata={},
            ),
        ]
        groups = chunker.group_by_ticket(docs)
        assert "GD-2026-03001" in groups
        assert "GD-2026-03002" in groups
        assert "__unmatched__" in groups
        assert len(groups["__unmatched__"]) == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

