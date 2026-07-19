"""
标量过滤表达式构建器 — 将元数据过滤条件转为 Milvus filter 表达式

支持的过滤字段（与 Milvus schema 中 VARCHAR 字段对应）:
  - ticket_id:   工单编号, 如 "GD-2026-03001"
  - device_type: 设备类型, 如 "CT 机"、"MRI 核磁共振"、"血液透析机"

用法:
  from src.retrieval.metadata_filter import build_filter_expr

  # 单条件
  expr = build_filter_expr(ticket_id="GD-2026-03001")
  # → 'ticket_id == "GD-2026-03001"'

  # 多条件（AND）
  expr = build_filter_expr(device_type="CT 机", ticket_id="GD-2026-03001")
  # → 'device_type == "CT 机" and ticket_id == "GD-2026-03001"'

  # 无过滤
  expr = build_filter_expr()           # → None
  expr = build_filter_expr(ticket_id=None)  # → None
"""

from typing import Optional

# 支持的过滤字段 → Milvus schema 字段名映射
_FIELD_MAP = {
    "ticket_id": "ticket_id",
    "device_type": "device_type",
}


def build_filter_expr(
    ticket_id: Optional[str] = None,
    device_type: Optional[str] = None,
) -> Optional[str]:
    """
    将过滤条件组合为 Milvus 标量过滤表达式。

    参数:
      ticket_id:   工单编号
      device_type: 设备类型

    返回:
      Milvus filter 表达式字符串，无条件时返回 None
    """
    conditions = []

    if ticket_id:
        conditions.append(f'ticket_id == "{ticket_id}"')
    if device_type:
        conditions.append(f'device_type == "{device_type}"')

    if not conditions:
        return None

    return " and ".join(conditions)


# ── 已知设备类型列表（用于 UI 下拉）──────────

KNOWN_DEVICE_TYPES = [
    "CT 机",
    "MRI 核磁共振",
    "全自动生化分析仪",
    "彩色多普勒超声诊断仪",
    "监护仪中央站",
    "血液透析机",
    "DR 数字化X射线摄影系统",
    "麻醉机",
    "高压灭菌器",
    "HIS 医院信息系统",
]
