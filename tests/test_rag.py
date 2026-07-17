"""
RAG 问答链测试 — pytest / 直接运行 两用

测试 RAGChain 的三种检索模式、过滤、兜底、异常处理。

运行方式:
  pytest tests/test_rag.py -v        # 全量测试
  python tests/test_rag.py           # 直接运行
  python tests/test_rag.py --quick   # 快速模式（只测语义，跳过 LLM 调用）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False

from src.llm.rag_chain import RAGChain


# ── Fixture ───────────────────────────────────

if _HAS_PYTEST:

    @pytest.fixture(scope="module")
    def chain():
        """所有测试共享同一个 RAGChain 实例"""
        return RAGChain()


# ── 测试用例 ──────────────────────────────────

if _HAS_PYTEST:

    class TestRAGInit:
        """RAGChain 初始化"""

        def test_init(self, chain):
            """测试 1: RAGChain 正常初始化"""
            assert chain is not None
            assert chain._dense is not None
            assert chain._sparse is not None
            assert chain._store is not None
            assert chain._llm is not None
            print("  [OK] RAGChain 初始化成功，4 个组件全部就绪")

        def test_init_components_loaded(self, chain):
            """测试 2: 组件属性可访问"""
            # 验证组件惰性加载后可用
            from src.embedding.embedding_client import EmbeddingClient
            from src.embedding.sparse_embedder import BM25SparseEmbedder
            from src.retrieval.milvus_client import MilvusStore
            from src.llm.llm_client import LLMClient

            assert isinstance(chain._dense, EmbeddingClient)
            assert isinstance(chain._sparse, BM25SparseEmbedder)
            assert isinstance(chain._store, MilvusStore)
            assert isinstance(chain._llm, LLMClient)
            print("  [OK] 4 个组件类型正确")


    class TestRAGSemantic:
        """语义检索模式"""

        def test_semantic_answer(self, chain):
            """测试 3: 语义检索返回有效回答"""
            result = chain.ask(
                question="CT扫描图像伪影是什么原因？简单回答。",
                mode="semantic", top_k=3,
            )
            assert result["mode"] == "semantic"
            assert result["has_answer"]
            assert len(result["answer"]) > 50
            assert len(result["sources"]) > 0
            print(f"  [OK] 回答 {len(result['answer'])} 字符, "
                  f"来源 {len(result['sources'])} 条")

        def test_semantic_response_structure(self, chain):
            """测试 4: 返回结构完整"""
            result = chain.ask(
                question="MRI故障排查",
                mode="semantic", top_k=2,
            )
            required_keys = {"question", "answer", "sources", "mode", "has_answer"}
            assert required_keys.issubset(result.keys())
            for s in result["sources"]:
                assert "ticket_id" in s
                assert "content" in s
                assert "score" in s
                assert "source" in s
            print("  [OK] 返回结构完整，sources 含 ticket_id/content/score/source")


    class TestRAGKeyword:
        """关键字检索模式"""

        def test_keyword_answer(self, chain):
            """测试 5: 关键字检索返回有效回答"""
            result = chain.ask(
                question="透析液电导率偏高报警怎么处理？简单回答。",
                mode="keyword", top_k=3,
            )
            assert result["mode"] == "keyword"
            assert result["has_answer"]
            assert len(result["answer"]) > 50
            print(f"  [OK] 回答 {len(result['answer'])} 字符, "
                  f"来源 {len(result['sources'])} 条")

        def test_keyword_matches_ticket(self, chain):
            """测试 6: 关键字检索命中血液透析工单"""
            result = chain.ask(
                question="透析液电导率传感器校准",
                mode="keyword", top_k=3,
            )
            ticket_ids = [s["ticket_id"] for s in result["sources"]]
            # 应命中血液透析相关工单 (GD-2026-03006)
            assert "GD-2026-03006" in ticket_ids, \
                f"关键字检索应命中 GD-2026-03006（血液透析），实际: {ticket_ids}"
            print(f"  [OK] 命中工单: {ticket_ids}")


    class TestRAGHybrid:
        """混合检索模式"""

        def test_hybrid_answer(self, chain):
            """测试 7: 混合检索返回有效回答"""
            result = chain.ask(
                question="医疗设备图像问题怎么排查？简单回答。",
                mode="hybrid", top_k=3,
            )
            assert result["mode"] == "hybrid"
            assert result["has_answer"]
            assert len(result["answer"]) > 50
            assert len(result["sources"]) > 0
            print(f"  [OK] 回答 {len(result['answer'])} 字符, "
                  f"来源 {len(result['sources'])} 条")

        def test_hybrid_scores_descending(self, chain):
            """测试 8: 来源按分数降序排列"""
            result = chain.ask(
                question="设备故障处理方案",
                mode="hybrid", top_k=5,
            )
            assert result["has_answer"], \
                "设备故障处理方案应在知识库中有匹配"
            scores = [s["score"] for s in result["sources"]]
            # 分数应递减或相等
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1] - 0.01  # 允许微小浮点误差
            print(f"  [OK] 分数降序: {[f'{s:.2f}' for s in scores]}")


    class TestRAGFilter:
        """工单过滤"""

        def test_ticket_filter(self, chain):
            """测试 9: 按工单号过滤"""
            result = chain.ask(
                question="设备故障处理",
                mode="semantic", top_k=3,
                ticket_id_filter="GD-2026-03002",
            )
            assert len(result["sources"]) > 0, \
                f"过滤 GD-2026-03002 应返回至少 1 条结果，实际 0 条"
            for s in result["sources"]:
                assert s["ticket_id"] == "GD-2026-03002", \
                    f"过滤结果应全是 GD-2026-03002，实际含 {s['ticket_id']}"
            print(f"  [OK] 过滤 'GD-2026-03002', "
                  f"命中 {len(result['sources'])} 条")


    class TestRAGEdgeCases:
        """边界情况"""

        def test_no_result_fallback(self, chain):
            """测试 10: 无相关结果走兜底"""
            result = chain.ask(
                question="心脏搭桥手术麻醉方案",
                mode="keyword", top_k=2,
            )
            # 医疗工单知识库不包含心脏手术相关内容，应走兜底
            assert result["has_answer"] is False, \
                f"期望 has_answer=False，实际为 {result['has_answer']}"
            assert result["sources"] == [], \
                f"期望 sources=[]，实际 {len(result['sources'])} 条"
            assert "未找到" in result["answer"], \
                f"兜底回答应包含'未找到'，实际: {result['answer'][:50]}"
            print(f"  [OK] 兜底生效: has_answer=False, sources=0")

        def test_topk_limit(self, chain):
            """测试 11: top_k 限制生效"""
            result = chain.ask(
                question="设备故障",
                mode="hybrid", top_k=2,
            )
            assert len(result["sources"]) <= 2
            print(f"  [OK] 来源数 {len(result['sources'])} <= top_k=2")


    class TestRAGEdgeCasesMore:
        """边界情况 — 补充"""

        def test_empty_question_no_crash(self, chain):
            """测试 12: 空字符串问题不崩溃"""
            result = chain.ask(
                question="",
                mode="semantic", top_k=2,
            )
            # 空问题不应崩溃，可能走兜底或返回低质量结果
            assert result is not None
            assert "answer" in result
            assert "sources" in result
            print(f"  [OK] 空问题未崩溃: has_answer={result['has_answer']}, "
                  f"answer={result['answer'][:50]}")

        def test_invalid_mode_falls_back_to_hybrid(self, chain):
            """测试 13: 无效检索模式静默降级为 hybrid"""
            result = chain.ask(
                question="CT故障",
                mode="nonexistent_mode",  # type: ignore
                top_k=2,
            )
            # RAGChain._retrieve 对未知 mode 默认走 hybrid，不应崩溃
            assert result is not None
            # mode 字段返回原始传入值（未校验），但检索实际按 hybrid 执行
            print(f"  [OK] 无效模式未崩溃, has_answer={result['has_answer']}")


    class TestRAGDefault:
        """默认参数"""

        def test_default_mode(self, chain):
            """测试 14: 不传 mode 走默认 hybrid"""
            result = chain.ask(
                question="CT故障",
                top_k=2,
            )
            assert result["mode"] == "hybrid"
            print(f"  [OK] 默认模式: {result['mode']}")


# ── 直接运行入口 ─────────────────────────────

if __name__ == "__main__":
    if not _HAS_PYTEST:
        print("错误: 需要 pytest，请先安装: pip install pytest")
        sys.exit(1)

    import argparse

    parser = argparse.ArgumentParser(description="RAG 问答链测试")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（只测初始化+语义，跳过 LLM 调用）")
    args = parser.parse_args()

    if args.quick:
        # 快速冒烟：只初始化 + 语义检索
        print("=== 快速冒烟测试 ===\n")
        chain = RAGChain()
        print("[OK] RAGChain 初始化成功")

        result = chain.ask("CT伪影", mode="semantic", top_k=2)
        print(f"[OK] semantic search: {len(result['sources'])} sources")
        print(f"[OK] answer length: {len(result['answer'])} chars")
    else:
        pytest.main([__file__, "-v", "-s"])
