"""
构建 Milvus 知识库 — 完整 ETL 流水线脚本

流程:  文档加载 → 文本分块 → 向量化 → Milvus 存储
用法:
  python build_milvus.py                          # 增量写入（跳过已有数据）
  python build_milvus.py --rebuild                # 删库重建
  python build_milvus.py --source data/documents/ # 指定文档目录
  python build_milvus.py --dry-run                # 只统计不写入，预览分块效果
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import get_config
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import MedicalWorkOrderChunker
from src.embedding.embedding_client import EmbeddingClient
from src.embedding.sparse_embedder import BM25SparseEmbedder
from src.retrieval.milvus_client import MilvusStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_milvus")


# ── 主流程 ────────────────────────────────────

def build_knowledge_base(
    source_dir: str = "data/documents",
    rebuild: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    执行完整 ETL 流水线。

    参数:
      source_dir: 文档目录路径
      rebuild:    True=删库重建, False=增量追加
      dry_run:    True=只分块不写库，预览效果

    返回:
      {
        "documents": int,      # 原始文档数
        "chunks": int,         # 分块数
        "vectors_written": int, # 写入量（dry_run 时为 0）
        "ticket_ids": list,    # 覆盖的工单编号
        "elapsed_seconds": float,
      }
    """
    t0 = time.time()
    cfg = get_config()

    # ── 1. 加载文档 ──────────────────────────
    print("\n" + "=" * 56)
    print("  [1/5] 加载文档")
    print("=" * 56)

    loader = DocumentLoader()
    docs = loader.load_directory(source_dir)

    if not docs:
        print("  [WARN] 没有找到任何文档，退出")
        return {"documents": 0, "chunks": 0, "vectors_written": 0,
                "ticket_ids": [], "elapsed_seconds": 0.0}

    print(f"  [OK] 加载完成: {len(docs)} 个文档")
    for d in docs:
        fname = d.metadata.get("source", "?")
        print(f"       - {fname}  ({len(d.page_content)} 字符)")

    # ── 2. 分块 ──────────────────────────────
    print("\n" + "=" * 56)
    print("  [2/5] 文本分块")
    print("=" * 56)

    chunker = MedicalWorkOrderChunker()
    chunks = chunker.split_documents(docs)

    # 统计
    sizes = [len(c.page_content) for c in chunks]
    ticket_ids = sorted(set(
        c.metadata.get("ticket_id", "__unmatched__") for c in chunks
    ))

    print(f"  [OK] 分块完成: {len(chunks)} 块")
    print(f"       平均尺寸: {sum(sizes)/len(sizes):.0f} 字符  "
          f"最小: {min(sizes)}  最大: {max(sizes)}")
    print(f"       覆盖工单: {len(ticket_ids)} 张")

    if dry_run:
        # 预览每块内容
        print("\n  ── 分块预览 (前 5 块) ──")
        for i, c in enumerate(chunks[:5]):
            preview = c.page_content[:80].replace("\n", "\\n")
            tid = c.metadata.get("ticket_id", "?")
            print(f"  [{i}] ticket={tid}  len={len(c.page_content)}  "
                  f"\"{preview}...\"")
        elapsed = time.time() - t0
        print(f"\n  [DRY-RUN] 跳过写入，耗时 {elapsed:.1f}s")
        return {
            "documents": len(docs), "chunks": len(chunks),
            "vectors_written": 0, "ticket_ids": ticket_ids,
            "elapsed_seconds": elapsed,
        }

    contents = [c.page_content for c in chunks]

    # ── 3. BM25 稀疏向量 ─────────────────────
    print("\n" + "=" * 56)
    print("  [3/5] BM25 稀疏向量（关键字检索）")
    print("=" * 56)

    sparse_embedder = BM25SparseEmbedder()
    sparse_embedder.fit(contents)
    sparse_vectors = sparse_embedder.encode_documents(contents)
    print(f"  [OK] BM25 训练+编码完成: {len(sparse_vectors)} 个稀疏向量")

    # ── 4. 稠密向量化 ────────────────────────
    print("\n" + "=" * 56)
    print("  [4/5] 稠密向量化 (DashScope text-embedding-v1)")
    print("=" * 56)

    emb = EmbeddingClient()
    # contents already extracted above
    batch_size = cfg.embedding.batch_size

    all_vectors = emb.embed_batch(contents)  # 内部自动分批 + 限流

    print(f"  [OK] 向量化完成: {len(all_vectors)} 条, 维度={len(all_vectors[0]) if all_vectors else 'N/A'}")

    # ── 5. 写入 Milvus ───────────────────────
    print("\n" + "=" * 56)
    print("  [5/5] 写入 Milvus")
    print("=" * 56)

    store = MilvusStore()

    # 建库（rebuild 时删旧库重建）
    if rebuild:
        store.create_collection(drop_if_exists=True)
    elif not store.collection_exists():
        store.create_collection()
    else:
        before_count = store.num_entities
        print(f"  Collection 已存在 ({before_count} 条)，增量追加")

    # 准备插入数据
    ticket_ids_list = [c.metadata.get("ticket_id", "__unknown__") for c in chunks]
    device_types_list = [c.metadata.get("device_type", "") for c in chunks]
    sources_list = [c.metadata.get("source", "unknown") for c in chunks]
    chunk_indices_list = [c.metadata.get("chunk_index", i) for i, c in enumerate(chunks)]

    # 写入
    written = store.insert(
        contents=contents,
        vectors=all_vectors,
        ticket_ids=ticket_ids_list,
        sources=sources_list,
        chunk_indices=chunk_indices_list,
        sparse_vectors=sparse_vectors,
        device_types=device_types_list,
    )

    elapsed = time.time() - t0

    # ── 汇总 ─────────────────────────────────
    print("\n" + "=" * 56)
    print(f"  [DONE] 知识库构建完成")
    print("=" * 56)
    print(f"  文档数:       {len(docs)}")
    print(f"  分块数:       {len(chunks)}")
    print(f"  写入向量:     {written} 条")
    print(f"  当前总量:     {store.num_entities} 条")
    print(f"  覆盖工单:     {len(ticket_ids)} 张")
    print(f"  总耗时:       {elapsed:.1f} 秒")
    print()

    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "vectors_written": written,
        "ticket_ids": ticket_ids,
        "elapsed_seconds": elapsed,
    }


# ── CLI 入口 ────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="构建 Milvus 知识库 — 完整 ETL 流水线",
    )
    parser.add_argument(
        "-s", "--source", default="data/documents",
        help="文档目录路径 (默认: data/documents)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="删库重建（清空旧数据后重新写入）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只分块不写入，预览分块效果",
    )
    args = parser.parse_args()

    build_knowledge_base(
        source_dir=args.source,
        rebuild=args.rebuild,
        dry_run=args.dry_run,
    )
