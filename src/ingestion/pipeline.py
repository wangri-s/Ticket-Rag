"""
摄入流水线 — 编排加载 → 分块 → 统计，产生可直接向量化的文档列表

调用链:
  DocumentLoader → MedicalWorkOrderChunker → 增强 metadata → 统计输出
"""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from src.config import get_config
from src.ingestion.chunker import MedicalWorkOrderChunker
from src.ingestion.loader import DocumentLoader


class IngestionPipeline:
    """
    文档摄入流水线

    用法:
      pipeline = IngestionPipeline()
      docs = pipeline.run("data/documents")

      # 按工单查看
      groups = pipeline.group_by_ticket(docs)
      for tid, chunks in groups.items():
          print(f"{tid}: {len(chunks)} 块")
    """

    def __init__(self):
        cfg = get_config()
        self.loader = DocumentLoader()
        self.chunker = MedicalWorkOrderChunker(cfg.chunking)

    # ── 主入口 ──────────────────────────────────

    def run(
        self, source: str, recursive: bool = True
    ) -> list[Document]:
        """
        执行完整摄入流程：加载 → 分块 → 返回

        参数:
          source:   文件路径或目录路径
          recursive: 若为目录，是否递归扫描子目录
        """
        source_path = Path(source)

        # Step 1: 加载
        if source_path.is_dir():
            docs = self.loader.load_directory(str(source_path), recursive=recursive)
        else:
            docs = self.loader.load(str(source_path))

        if not docs:
            print(f"[摄入] 未找到可处理的文档: {source}")
            return []

        total_chars = sum(len(d.page_content) for d in docs)
        print(f"[摄入] 加载完成: {len(docs)} 个文档, 共 {total_chars} 字符")

        # Step 2: 分块
        chunks = self.chunker.split_documents(docs)

        # Step 3: 统计
        self._print_stats(chunks)

        return chunks

    # ── 工具方法 ──────────────────────────────────

    def run_with_stats(self, source: str) -> dict:
        """摄入 + 返回详细统计数据"""
        chunks = self.run(source)
        group_map = self.group_by_ticket(chunks)

        sizes = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "total_tickets": len(group_map) - (1 if "__unmatched__" in group_map else 0),
            "avg_chunk_size": round(sum(sizes) / len(chunks), 1) if chunks else 0,
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
            "ticket_ids": sorted(
                [k for k in group_map if k != "__unmatched__"]
            ),
            "chunks": chunks,
        }

    @staticmethod
    def group_by_ticket(chunks: list[Document]) -> dict[str, list[Document]]:
        """将分块结果按工单编号分组"""
        from src.ingestion.chunker import extract_ticket_id

        groups: dict[str, list[Document]] = {}
        for doc in chunks:
            tid = doc.metadata.get("ticket_id") or extract_ticket_id(doc.page_content)
            key = tid or "__unmatched__"
            groups.setdefault(key, []).append(doc)
        return groups

    # ── 内部 ─────────────────────────────────────

    def _print_stats(self, chunks: list[Document]) -> None:
        if not chunks:
            print("[摄入] 分块结果为空")
            return

        sizes = [len(c.page_content) for c in chunks]
        group_map = self.group_by_ticket(chunks)
        ticket_count = len(group_map) - (1 if "__unmatched__" in group_map else 0)

        print(
            f"[摄入] 分块完成: {len(chunks)} 块, "
            f"平均 {round(sum(sizes) / len(chunks))} 字符/块, "
            f"范围 [{min(sizes)}, {max(sizes)}], "
            f"涵盖 {ticket_count} 张工单"
        )
