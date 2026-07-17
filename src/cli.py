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


# ── RAG 问答 ──────────────────────────────────

HELP_ASK = """
  命令:
    <问题>      直接输入问题进行问答
    mode <m>    切换检索模式: semantic / keyword / hybrid
    topk <n>    设置检索条数 (1-20)
    filter <id> 按工单号过滤
    clear       清除过滤条件
    help        显示此帮助
    stats       显示当前设置
    quit/exit   退出交互模式
"""


def cmd_ask(args):
    """RAG 问答：检索 + LLM 生成"""
    from src.llm.rag_chain import RAGChain

    # 命令行模式（-q 指定问题）
    if args.question:
        chain = RAGChain()
        result = chain.ask(
            question=args.question,
            mode=args.mode,
            top_k=args.top_k,
            ticket_id_filter=args.ticket_id,
        )
        _print_result(result, args.mode)
        return

    # ── 交互模式 ──
    print()
    print("=" * 56)
    print("  RAG 智能工单问答系统  v1.0")
    print("  Powered by Milvus + Qwen-Max")
    print("=" * 56)

    print()
    print("  正在初始化组件...")
    from src.config import get_config
    chain = RAGChain()
    cfg = get_config()
    print(f"  [OK] Embedding: {cfg.embedding.model} | "
          f"Milvus: {cfg.milvus.uri} | LLM: {cfg.llm.model}")
    print(f"  [OK] 全部组件就绪")

    print(HELP_ASK)
    print(f"  当前模式: {args.mode}")

    # 交互状态（可变，支持运行时切换）
    mode = args.mode
    top_k = args.top_k
    ticket_id = args.ticket_id

    while True:
        try:
            mode_label = f"[{mode}]" if not ticket_id else f"[{mode} @{ticket_id}]"
            question = input(f"\n  {mode_label}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not question:
            continue

        # 内置命令
        cmd = question.lower()
        if cmd in ("quit", "exit", "q"):
            print("  再见！")
            break
        if cmd == "help":
            print(HELP_ASK)
            continue
        if cmd == "stats":
            _print_stats(mode, top_k, ticket_id)
            continue
        if cmd == "clear":
            ticket_id = None
            print("  已清除过滤条件")
            continue
        if cmd.startswith("mode "):
            new = cmd.split()[1]
            if new in ("semantic", "keyword", "hybrid"):
                mode = new
                print(f"  检索模式 → {mode}")
            else:
                print(f"  无效模式: {new}")
            continue
        if cmd.startswith("topk "):
            try:
                n = int(cmd.split()[1])
                if 1 <= n <= 20:
                    top_k = n
                    print(f"  检索条数 → {top_k}")
                else:
                    print("  请输入 1-20")
            except (IndexError, ValueError):
                print("  用法: topk <数字>")
            continue
        if cmd.startswith("filter "):
            ticket_id = cmd.split(None, 1)[1].strip()
            print(f"  工单过滤 → {ticket_id}")
            continue

        # RAG 问答
        try:
            result = chain.ask(
                question=question, mode=mode,
                top_k=top_k, ticket_id_filter=ticket_id,
            )
            _print_result(result, mode)
        except Exception as e:
            print(f"\n  [ERROR] {e}\n  请重试或输入 quit 退出")


def _print_result(result: dict, mode: str):
    """打印回答 + 引用来源"""
    print()
    print("─" * 56)
    print("  【回答】")
    print("─" * 56)
    print(result["answer"])
    print()

    sources = result.get("sources", [])
    if sources:
        print("─" * 56)
        print(f"  【引用来源】共 {len(sources)} 条  |  mode={mode}")
        print("─" * 56)
        for i, s in enumerate(sources, 1):
            tid = s.get("ticket_id", "?")
            score = s.get("score", 0.0)
            content = s.get("content", "")[:100].replace("\n", " ")
            print(f"  [{i}] {tid}  (score={score:.4f})")
            print(f"      {content}...")
            print()
    else:
        print("  [注意] 本次回答未引用知识库工单")


def _print_stats(mode: str, top_k: int, ticket_id: str):
    """显示当前交互状态"""
    from src.config import get_config

    cfg = get_config()
    print(f"""
  当前设置:
    检索模式:   {mode}
    检索条数:   {top_k or cfg.retrieval.top_k}
    工单过滤:   {ticket_id or '无'}
    LLM 模型:   {cfg.llm.model}
    LLM 温度:   {cfg.llm.temperature}
""")


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
  python -m src.cli ask -q "CT伪影怎么处理？"          # RAG 问答
  python -m src.cli ask                               # 交互式问答
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

    # ask
    p_ask = sub.add_parser("ask", help="RAG 智能问答（检索 + LLM 生成）")
    p_ask.add_argument("-q", "--question", default=None,
                        help="用户问题（不填则进入交互模式）")
    p_ask.add_argument("-m", "--mode", default="hybrid",
                        choices=["semantic", "keyword", "hybrid"],
                        help="检索模式 (默认: hybrid)")
    p_ask.add_argument("-k", "--top-k", type=int, default=None,
                        help="检索条数 (默认取 config.yml)")
    p_ask.add_argument("--ticket-id", default=None,
                        help="按工单号过滤")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "chunk-preview":
        cmd_chunk_preview(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "ask":
        cmd_ask(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
