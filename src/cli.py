"""
命令行工具 — 无需写代码即可执行摄入、预览、统计等操作

用法:
  python -m src.cli ingest -s data/documents       # 摄入文档
  python -m src.cli ingest -s data/documents --json # JSON 输出
  python -m src.cli stats                           # 查看配置摘要
  python -m src.cli chunk-preview -f data/documents/工单知识库.txt  # 预览分块效果
"""

import argparse
import json
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cmd_ingest(args):
    """摄入文档：加载 → 分块 → 统计"""
    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    result = pipeline.run_with_stats(args.source)

    if args.json:
        # JSON 模式：只输出数据（供脚本调用）
        output = {
            "total_chunks": result["total_chunks"],
            "total_tickets": result["total_tickets"],
            "avg_chunk_size": result["avg_chunk_size"],
            "min_chunk_size": result["min_chunk_size"],
            "max_chunk_size": result["max_chunk_size"],
            "ticket_ids": result["ticket_ids"],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        # 人类可读模式
        print()
        print("=" * 55)
        print("  摄入结果")
        print("=" * 55)
        print(f"  总块数:     {result['total_chunks']}")
        print(f"  工单数:     {result['total_tickets']}")
        print(f"  平均大小:   {result['avg_chunk_size']} 字符")
        print(f"  最小/最大:  {result['min_chunk_size']} / {result['max_chunk_size']} 字符")
        print(f"  工单编号:   {', '.join(result['ticket_ids'])}")

        # 打印每张工单的块数
        print()
        print("  每工单块数:")
        groups = pipeline.group_by_ticket(result["chunks"])
        for tid in sorted(groups.keys()):
            print(f"    {tid}: {len(groups[tid])} 块")
        print()


def cmd_chunk_preview(args):
    """预览分块效果：显示前 N 块的内容"""
    from src.ingestion.loader import DocumentLoader
    from src.ingestion.chunker import MedicalWorkOrderChunker

    loader = DocumentLoader()
    chunker = MedicalWorkOrderChunker()

    docs = loader.load(args.file)
    chunks = chunker.split_documents(docs)

    print()
    print(f"文件: {args.file}")
    print(f"共 {len(chunks)} 块\n")

    n = min(args.top, len(chunks))
    for i in range(n):
        c = chunks[i]
        tid = c.metadata.get("ticket_id", "?")
        print(f"─── 块 {i}  [{tid}]  ({len(c.page_content)} 字符) ───")
        print(c.page_content[:300])
        if len(c.page_content) > 300:
            print("...")
        print()


def cmd_stats(args):
    """显示当前配置摘要"""
    from src.config import get_config

    cfg = get_config()

    print()
    print("=" * 45)
    print("  系统配置")
    print("=" * 45)
    print(f"  Milvus:    {cfg.milvus.uri}")
    print(f"  Collection:{cfg.milvus.collection_name} (dim={cfg.milvus.vector_dim})")
    print(f"  Embedding: {cfg.embedding.model} @ {cfg.embedding.provider}")
    print(f"  LLM:       {cfg.llm.model} (T={cfg.llm.temperature})")
    print(f"  Retrieval: top_k={cfg.retrieval.top_k}")
    print(f"  Chunking:  {cfg.chunking.strategy}, "
          f"size={cfg.chunking.chunk_size}, "
          f"overlap={cfg.chunking.chunk_overlap}")

    missing = cfg.validate()
    if missing:
        print(f"\n  [WARN] 缺失配置: {', '.join(missing)}")
    else:
        print(f"\n  [OK] 配置完整")
    print()


# ── CLI 入口 ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="智能工单 RAG 系统 — 命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.cli ingest -s data/documents         # 摄入知识库
  python -m src.cli ingest -s data/documents --json  # JSON 输出
  python -m src.cli chunk-preview -f data/documents/工单知识库.txt -n 5
  python -m src.cli stats                            # 查看配置
        """,
    )

    sub = parser.add_subparsers(dest="command", help="子命令")

    # ingest
    p_ingest = sub.add_parser("ingest", help="摄入文档（加载 + 分块）")
    p_ingest.add_argument("-s", "--source", required=True,
                          help="文件或目录路径")
    p_ingest.add_argument("--json", action="store_true",
                          help="JSON 格式输出")

    # chunk-preview
    p_preview = sub.add_parser("chunk-preview", help="预览分块效果")
    p_preview.add_argument("-f", "--file", required=True,
                           help="文件路径")
    p_preview.add_argument("-n", "--top", type=int, default=5,
                           help="显示前 N 块 (默认 5)")

    # stats
    sub.add_parser("stats", help="显示系统配置摘要")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "chunk-preview":
        cmd_chunk_preview(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
