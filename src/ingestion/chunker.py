"""
文本分割器 — 递归字符分割

基于 config.yml 中 chunking 配置，将长文档切分为语义完整的文本块。
分隔符按优先级逐级尝试：工单分隔线 → 字段标签 → 段落 → 句子 → 兜底空格。

每个切出的块自动提取所属工单编号，存入 metadata 用于检索溯源。
"""

import re
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import ChunkingConfig, get_config

# 工单编号匹配: GD-YYYY-MMDDD
RE_TICKET_ID = re.compile(r"GD-\d{4}-\d{5}")

# 设备类型匹配: 【设备类型】xxx
RE_DEVICE_TYPE = re.compile(r"【设备类型】(.+)")


def extract_ticket_id(text: str) -> str | None:
    """从文本中提取工单编号，找不到返回 None"""
    m = RE_TICKET_ID.search(text)
    return m.group(0) if m else None


def extract_device_type(text: str) -> str | None:
    """从文本中提取设备类型，找不到返回 None"""
    m = RE_DEVICE_TYPE.search(text)
    return m.group(1).strip() if m else None


class MedicalWorkOrderChunker:
    """医疗工单专用文本分割器

    核心职责:
      1. 递归分块 — 7 级分隔符逐级降级，保证每块语义完整
      2. 元数据继承 — source / page 等信息从父文档传递到子块
      3. 工单溯源 — 自动提取每块所属工单编号，检索结果可追溯到原文
      4. 碎片过滤 — 丢弃短于 min_chunk_length 的无意义块
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        cfg = config or get_config().chunking
        self.chunk_size = cfg.chunk_size
        self.chunk_overlap = cfg.chunk_overlap
        self.min_chunk_length = cfg.min_chunk_length
        self.separators = cfg.separators

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,               # 字符计数，不依赖 token 编码器
            separators=self.separators,
            keep_separator=True,               # 保留分隔符，检索上下文更完整
        )

    # ──────────────────────────────────────────────
    #  核心分块方法
    # ──────────────────────────────────────────────

    def split_text(self, text: str) -> list[str]:
        """纯文本分块，返回字符串列表"""
        if not text or not text.strip():
            return []

        chunks = self._splitter.split_text(text)
        return [c for c in chunks if len(c) >= self.min_chunk_length]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        文档对象分块，保留并增强元数据。

        每个子块继承父文档的 metadata，并自动追加:
          - chunk_index:  块序号
          - ticket_id:    所属工单编号
            · 先从当前块文本提取
            · 提取不到则继承上一个块的 ticket_id（同一工单的后续分块）
        """
        if not documents:
            return []

        split_docs = self._splitter.split_documents(documents)

        enriched: list[Document] = []
        last_ticket_id: str | None = None
        last_device_type: str | None = None

        for i, doc in enumerate(split_docs):
            if len(doc.page_content) < self.min_chunk_length:
                continue

            doc.metadata["chunk_index"] = i

            # 提取工单编号
            tid = extract_ticket_id(doc.page_content)
            if tid:
                doc.metadata["ticket_id"] = tid
                last_ticket_id = tid
            elif last_ticket_id:
                doc.metadata["ticket_id"] = last_ticket_id

            # 提取设备类型
            dtype = extract_device_type(doc.page_content)
            if dtype:
                doc.metadata["device_type"] = dtype
                last_device_type = dtype
            elif last_device_type:
                doc.metadata["device_type"] = last_device_type

            enriched.append(doc)

        return enriched

    # ──────────────────────────────────────────────
    #  工具方法
    # ──────────────────────────────────────────────

    def split_with_stats(self, text: str) -> dict:
        """
        分块并返回统计信息，用于调试 & 预览分块质量。

        返回:
          { "chunks": [...], "count": int, "avg_size": float,
            "min_size": int, "max_size": int, "ticket_ids": [...] }
        """
        chunks = self.split_text(text)
        sizes = [len(c) for c in chunks]
        return {
            "chunks": chunks,
            "count": len(chunks),
            "avg_size": round(sum(sizes) / len(chunks), 1) if chunks else 0,
            "min_size": min(sizes) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "ticket_ids": [tid for c in chunks if (tid := extract_ticket_id(c))],
        }

    def group_by_ticket(
        self, documents: list[Document]
    ) -> dict[str, list[Document]]:
        """
        将分块后的文档按工单编号分组。

        返回:
          { "GD-2026-03001": [Doc1, Doc2, ...], "GD-2026-03002": [...] }
        未识别工单编号的块放入 "__unmatched__" 键。
        """
        groups: dict[str, list[Document]] = {}
        for doc in documents:
            tid = doc.metadata.get("ticket_id") or extract_ticket_id(doc.page_content)
            key = tid or "__unmatched__"
            groups.setdefault(key, []).append(doc)
        return groups
