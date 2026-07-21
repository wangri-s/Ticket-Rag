"""
测试标量过滤表达式 — src/retrieval/metadata_filter.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.metadata_filter import build_filter_expr, KNOWN_DEVICE_TYPES


class TestBuildFilterExpr:
    """标量过滤表达式构建"""

    # ── 单条件 ──────────────────────────────────

    def test_single_ticket_id(self):
        expr = build_filter_expr(ticket_id="GD-2026-03001")
        assert expr == 'ticket_id == "GD-2026-03001"'

    def test_single_device_type(self):
        expr = build_filter_expr(device_type="CT 机")
        assert expr == 'device_type == "CT 机"'

    # ── 多条件 AND ──────────────────────────────

    def test_both_conditions_and(self):
        expr = build_filter_expr(
            ticket_id="GD-2026-03001",
            device_type="CT 机",
        )
        assert "ticket_id == " in expr
        assert "device_type == " in expr
        assert " and " in expr

    # ── 无参数 / 空参数 ─────────────────────────

    def test_no_params_returns_none(self):
        assert build_filter_expr() is None

    def test_none_params_returns_none(self):
        assert build_filter_expr(ticket_id=None, device_type=None) is None

    def test_empty_string_returns_none(self):
        # 空字符串在 Python 中是 falsy，不加入条件
        expr = build_filter_expr(ticket_id="")
        assert expr is None

    def test_empty_string_both_returns_none(self):
        assert build_filter_expr(ticket_id="", device_type="") is None

    # ── 部分参数 ───────────────────────────────

    def test_only_device_type_when_ticket_id_none(self):
        expr = build_filter_expr(ticket_id=None, device_type="MRI 核磁共振")
        assert expr == 'device_type == "MRI 核磁共振"'

    def test_only_ticket_id_when_device_type_none(self):
        expr = build_filter_expr(ticket_id="GD-2026-03005", device_type=None)
        assert expr == 'ticket_id == "GD-2026-03005"'

    # ── 特殊字符 ────────────────────────────────

    def test_ticket_id_with_hyphen(self):
        expr = build_filter_expr(ticket_id="GD-2026-03001")
        assert '-2026-' in expr

    def test_device_type_with_spaces(self):
        expr = build_filter_expr(device_type="MRI 核磁共振")
        assert '"MRI 核磁共振"' in expr


class TestKnownDeviceTypes:
    """已知设备类型列表"""

    def test_has_minimum_entries(self):
        assert len(KNOWN_DEVICE_TYPES) >= 10

    def test_contains_ct(self):
        assert "CT 机" in KNOWN_DEVICE_TYPES

    def test_contains_mri(self):
        assert "MRI 核磁共振" in KNOWN_DEVICE_TYPES

    def test_contains_hemodialysis(self):
        assert "血液透析机" in KNOWN_DEVICE_TYPES


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
