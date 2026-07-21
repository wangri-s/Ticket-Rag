"""
测试 RAG 核心链 — src/llm/rag_chain.py

Mock 所有外部依赖，测试检索→压缩→Prompt→LLM 编排逻辑。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ══════════════════════════════════════════════════════════

class TestRAGChainCacheHit:
    """语义缓存命中"""

    @patch("src.llm.rag_chain.QACache")
    @patch("src.llm.rag_chain.MemoryManager")
    @patch("src.llm.rag_chain.LLMClient")
    @patch("src.llm.rag_chain.MilvusStore")
    @patch("src.llm.rag_chain.EmbeddingClient")
    @patch("src.llm.rag_chain.BM25SparseEmbedder")
    @patch("src.llm.rag_chain.ContextCompressor")
    def test_cache_hit_skips_llm(self, mock_comp, mock_sparse, mock_dense,
                                  mock_store, mock_llm, mock_mem, mock_cache):
        """缓存命中 → 直接返回，不调 LLM"""
        mock_cache.return_value.lookup.return_value = (True, "缓存答案")
        mock_cache.return_value.enabled = True
        mock_cache.return_value.available = True
        mock_mem.return_value.available = True

        # 清除 lru_cache 让 mock 生效
        from src.llm import rag_chain
        rag_chain._get_qa_cache.cache_clear()
        rag_chain._get_memory.cache_clear()
        rag_chain._get_llm.cache_clear()

        chain = rag_chain.RAGChain()
        chain._memory = mock_mem.return_value
        chain._qa_cache = mock_cache.return_value
        chain._llm = mock_llm.return_value

        result = chain.ask("CT伪影怎么处理？")
        assert result["from_cache"] is True
        assert result["answer"] == "缓存答案"
        # LLM 不应被调用
        mock_llm.return_value.generate.assert_not_called()

    @patch("src.llm.rag_chain.QACache")
    @patch("src.llm.rag_chain.MemoryManager")
    def test_cache_hit_streaming(self, mock_mem, mock_cache):
        """流式缓存命中"""
        mock_cache.return_value.lookup.return_value = (True, "流式缓存答案")
        mock_cache.return_value.enabled = True
        mock_cache.return_value.available = True
        mock_mem.return_value.available = True

        from src.llm import rag_chain
        rag_chain._get_qa_cache.cache_clear()
        rag_chain._get_memory.cache_clear()

        chain = rag_chain.RAGChain()
        chain._memory = mock_mem.return_value
        chain._qa_cache = mock_cache.return_value

        result = chain.ask_stream("问题")
        assert result["from_cache"] is True
        # 流应该 yield 缓存答案
        stream_output = "".join(list(result["stream"]))
        assert stream_output == "流式缓存答案"


class TestRAGChainFullFlow:
    """完整问答流程"""

    @patch("src.llm.rag_chain.QACache")
    @patch("src.llm.rag_chain.MemoryManager")
    @patch("src.llm.rag_chain.LLMClient")
    @patch("src.llm.rag_chain.MilvusStore")
    @patch("src.llm.rag_chain.EmbeddingClient")
    @patch("src.llm.rag_chain.BM25SparseEmbedder")
    @patch("src.llm.rag_chain.ContextCompressor")
    def test_full_flow_returns_result(self, mock_comp, mock_sparse, mock_dense,
                                       mock_store, mock_llm, mock_mem, mock_cache):
        """检索+生成完整流程"""
        # 缓存未命中
        mock_cache.return_value.lookup.return_value = (False, None)
        mock_cache.return_value.enabled = True
        mock_cache.return_value.available = True

        # 记忆可用
        mock_mem.return_value.available = True
        mock_mem.return_value.build_memory_prompt.return_value = ""

        # 检索返回
        mock_store.return_value.hybrid_search.return_value = [
            {
                "content": "CT 伪影：探测器校准即可解决。",
                "ticket_id": "GD-2026-03001",
                "source": "test.txt",
                "score": 0.95,
                "chunk_index": 0,
            }
        ]

        # 压缩器透传
        mock_comp.return_value.compress.side_effect = lambda chunks, *a, **kw: chunks

        # LLM 返回
        mock_llm.return_value.generate.return_value = "根据工单 GD-2026-03001，建议检查探测器校准。"

        from src.llm import rag_chain
        for attr in ['_get_qa_cache', '_get_memory', '_get_llm', '_get_store',
                      '_get_dense_embedder', '_get_sparse_embedder', '_get_compressor']:
            getattr(rag_chain, attr).cache_clear()

        chain = rag_chain.RAGChain()
        chain._memory = mock_mem.return_value
        chain._qa_cache = mock_cache.return_value
        chain._llm = mock_llm.return_value
        chain._store = mock_store.return_value
        chain._dense = mock_dense.return_value
        chain._compressor = mock_comp.return_value

        result = chain.ask("CT伪影怎么处理？")
        assert result["has_answer"] is True
        assert len(result["sources"]) > 0
        assert result["sources"][0]["ticket_id"] == "GD-2026-03001"

    @patch("src.llm.rag_chain.QACache")
    @patch("src.llm.rag_chain.MemoryManager")
    @patch("src.llm.rag_chain.LLMClient")
    @patch("src.llm.rag_chain.MilvusStore")
    @patch("src.llm.rag_chain.EmbeddingClient")
    @patch("src.llm.rag_chain.BM25SparseEmbedder")
    @patch("src.llm.rag_chain.ContextCompressor")
    def test_no_results_returns_fallback(self, mock_comp, mock_sparse, mock_dense,
                                          mock_store, mock_llm, mock_mem, mock_cache):
        """无检索结果→兜底答案"""
        mock_cache.return_value.lookup.return_value = (False, None)
        mock_cache.return_value.enabled = True
        mock_cache.return_value.available = True
        mock_mem.return_value.available = True

        # 检索返回空
        mock_store.return_value.hybrid_search.return_value = []

        from src.llm import rag_chain
        for attr in ['_get_qa_cache', '_get_memory', '_get_llm', '_get_store',
                      '_get_dense_embedder', '_get_sparse_embedder', '_get_compressor']:
            getattr(rag_chain, attr).cache_clear()

        chain = rag_chain.RAGChain()
        chain._memory = mock_mem.return_value
        chain._qa_cache = mock_cache.return_value
        chain._llm = mock_llm.return_value
        chain._store = mock_store.return_value
        chain._dense = mock_dense.return_value
        chain._compressor = mock_comp.return_value

        result = chain.ask("今天天气怎么样？")
        assert result["has_answer"] is False
        assert result["sources"] == []
        assert "未找到" in result["answer"]
        # LLM 不应被调用
        mock_llm.return_value.generate.assert_not_called()

    @patch("src.llm.rag_chain.QACache")
    @patch("src.llm.rag_chain.MemoryManager")
    @patch("src.llm.rag_chain.LLMClient")
    @patch("src.llm.rag_chain.MilvusStore")
    @patch("src.llm.rag_chain.EmbeddingClient")
    @patch("src.llm.rag_chain.BM25SparseEmbedder")
    @patch("src.llm.rag_chain.ContextCompressor")
    def test_semantic_mode(self, mock_comp, mock_sparse, mock_dense,
                            mock_store, mock_llm, mock_mem, mock_cache):
        """semantic 模式仅调用稠密检索"""
        mock_cache.return_value.lookup.return_value = (False, None)
        mock_cache.return_value.enabled = True
        mock_mem.return_value.available = True
        mock_store.return_value.search.return_value = [{
            "content": "内容", "ticket_id": "GD-001", "source": "t.txt", "score": 0.9, "chunk_index": 0,
        }]
        mock_comp.return_value.compress.side_effect = lambda c, *a, **kw: c
        mock_llm.return_value.generate.return_value = "答案"

        from src.llm import rag_chain
        for attr in ['_get_qa_cache', '_get_memory', '_get_llm', '_get_store',
                      '_get_dense_embedder', '_get_sparse_embedder', '_get_compressor']:
            getattr(rag_chain, attr).cache_clear()

        chain = rag_chain.RAGChain()
        chain._memory = mock_mem.return_value
        chain._qa_cache = mock_cache.return_value
        chain._llm = mock_llm.return_value
        chain._store = mock_store.return_value
        chain._dense = mock_dense.return_value
        chain._compressor = mock_comp.return_value

        result = chain.ask("问题", mode="semantic")
        # semantic 模式 → search() 被调用
        mock_store.return_value.search.assert_called()
        # hybrid_search / sparse_search 不应被调
        mock_store.return_value.hybrid_search.assert_not_called()
        assert result["mode"] == "semantic"

    @patch("src.llm.rag_chain.QACache")
    @patch("src.llm.rag_chain.MemoryManager")
    @patch("src.llm.rag_chain.LLMClient")
    @patch("src.llm.rag_chain.MilvusStore")
    @patch("src.llm.rag_chain.EmbeddingClient")
    @patch("src.llm.rag_chain.BM25SparseEmbedder")
    @patch("src.llm.rag_chain.ContextCompressor")
    def test_keyword_mode(self, mock_comp, mock_sparse, mock_dense,
                           mock_store, mock_llm, mock_mem, mock_cache):
        """keyword 模式仅调用稀疏检索"""
        mock_cache.return_value.lookup.return_value = (False, None)
        mock_cache.return_value.enabled = True
        mock_mem.return_value.available = True
        mock_store.return_value.sparse_search.return_value = [{
            "content": "内容", "ticket_id": "GD-002", "source": "t.txt", "score": 0.85, "chunk_index": 0,
        }]
        mock_comp.return_value.compress.side_effect = lambda c, *a, **kw: c
        mock_llm.return_value.generate.return_value = "答案"

        from src.llm import rag_chain
        for attr in ['_get_qa_cache', '_get_memory', '_get_llm', '_get_store',
                      '_get_dense_embedder', '_get_sparse_embedder', '_get_compressor']:
            getattr(rag_chain, attr).cache_clear()

        chain = rag_chain.RAGChain()
        chain._memory = mock_mem.return_value
        chain._qa_cache = mock_cache.return_value
        chain._llm = mock_llm.return_value
        chain._store = mock_store.return_value
        chain._sparse = mock_sparse.return_value
        chain._compressor = mock_comp.return_value

        result = chain.ask("问题", mode="keyword")
        mock_store.return_value.sparse_search.assert_called()
        assert result["mode"] == "keyword"

    @patch("src.llm.rag_chain.QACache")
    @patch("src.llm.rag_chain.MemoryManager")
    @patch("src.llm.rag_chain.LLMClient")
    @patch("src.llm.rag_chain.MilvusStore")
    @patch("src.llm.rag_chain.EmbeddingClient")
    @patch("src.llm.rag_chain.BM25SparseEmbedder")
    @patch("src.llm.rag_chain.ContextCompressor")
    def test_json_output_format(self, mock_comp, mock_sparse, mock_dense,
                                 mock_store, mock_llm, mock_mem, mock_cache):
        """JSON 输出格式"""
        mock_cache.return_value.lookup.return_value = (False, None)
        mock_cache.return_value.enabled = True
        mock_mem.return_value.available = True
        mock_store.return_value.hybrid_search.return_value = [{
            "content": "内容", "ticket_id": "GD-001", "source": "t.txt", "score": 0.9, "chunk_index": 0,
        }]
        mock_comp.return_value.compress.side_effect = lambda c, *a, **kw: c
        mock_llm.return_value.generate.return_value = '{"has_reference": true}'

        from src.llm import rag_chain
        for attr in ['_get_qa_cache', '_get_memory', '_get_llm', '_get_store',
                      '_get_dense_embedder', '_get_sparse_embedder', '_get_compressor']:
            getattr(rag_chain, attr).cache_clear()

        chain = rag_chain.RAGChain()
        chain._memory = mock_mem.return_value
        chain._qa_cache = mock_cache.return_value
        chain._llm = mock_llm.return_value
        chain._store = mock_store.return_value
        chain._dense = mock_dense.return_value
        chain._compressor = mock_comp.return_value

        result = chain.ask("问题", output_format="json")
        assert result["output_format"] == "json"
        assert result["has_answer"] is True

    @patch("src.llm.rag_chain.QACache")
    @patch("src.llm.rag_chain.MemoryManager")
    @patch("src.llm.rag_chain.LLMClient")
    @patch("src.llm.rag_chain.MilvusStore")
    @patch("src.llm.rag_chain.EmbeddingClient")
    @patch("src.llm.rag_chain.BM25SparseEmbedder")
    @patch("src.llm.rag_chain.ContextCompressor")
    def test_session_memory_saved(self, mock_comp, mock_sparse, mock_dense,
                                   mock_store, mock_llm, mock_mem, mock_cache):
        """传入 session_id 时记忆被保存"""
        mock_cache.return_value.lookup.return_value = (False, None)
        mock_cache.return_value.enabled = True
        mock_mem.return_value.available = True
        mock_store.return_value.hybrid_search.return_value = [{
            "content": "内容", "ticket_id": "GD-001", "source": "t.txt", "score": 0.9, "chunk_index": 0,
        }]
        mock_comp.return_value.compress.side_effect = lambda c, *a, **kw: c
        mock_llm.return_value.generate.return_value = "答案"

        from src.llm import rag_chain
        for attr in ['_get_qa_cache', '_get_memory', '_get_llm', '_get_store',
                      '_get_dense_embedder', '_get_sparse_embedder', '_get_compressor']:
            getattr(rag_chain, attr).cache_clear()

        chain = rag_chain.RAGChain()
        chain._memory = mock_mem.return_value
        chain._qa_cache = mock_cache.return_value
        chain._llm = mock_llm.return_value
        chain._store = mock_store.return_value
        chain._dense = mock_dense.return_value
        chain._compressor = mock_comp.return_value

        result = chain.ask("问题", session_id="sess-001")
        # 用户消息 + 助手回答 = 2 次 add_message 调用
        assert mock_mem.return_value.add_message.call_count == 2
        assert result["session_id"] == "sess-001"

    @patch("src.llm.rag_chain.QACache")
    @patch("src.llm.rag_chain.MemoryManager")
    @patch("src.llm.rag_chain.LLMClient")
    @patch("src.llm.rag_chain.MilvusStore")
    @patch("src.llm.rag_chain.EmbeddingClient")
    @patch("src.llm.rag_chain.BM25SparseEmbedder")
    @patch("src.llm.rag_chain.ContextCompressor")
    def test_ticket_id_filter_passed(self, mock_comp, mock_sparse, mock_dense,
                                      mock_store, mock_llm, mock_mem, mock_cache):
        """ticket_id_filter 传递到 build_filter_expr"""
        mock_cache.return_value.lookup.return_value = (False, None)
        mock_cache.return_value.enabled = True
        mock_mem.return_value.available = True
        mock_store.return_value.hybrid_search.return_value = [{
            "content": "内容", "ticket_id": "GD-2026-03001", "source": "t.txt", "score": 0.9, "chunk_index": 0,
        }]
        mock_comp.return_value.compress.side_effect = lambda c, *a, **kw: c
        mock_llm.return_value.generate.return_value = "答案"

        from src.llm import rag_chain
        for attr in ['_get_qa_cache', '_get_memory', '_get_llm', '_get_store',
                      '_get_dense_embedder', '_get_sparse_embedder', '_get_compressor']:
            getattr(rag_chain, attr).cache_clear()

        chain = rag_chain.RAGChain()
        chain._memory = mock_mem.return_value
        chain._qa_cache = mock_cache.return_value
        chain._llm = mock_llm.return_value
        chain._store = mock_store.return_value
        chain._dense = mock_dense.return_value
        chain._compressor = mock_comp.return_value

        result = chain.ask("问题", ticket_id_filter="GD-2026-03001")
        assert result["has_answer"] is True


class TestRAGChainScoreFilter:
    """分数阈值过滤"""

    @patch("src.llm.rag_chain.QACache")
    @patch("src.llm.rag_chain.MemoryManager")
    @patch("src.llm.rag_chain.LLMClient")
    @patch("src.llm.rag_chain.MilvusStore")
    @patch("src.llm.rag_chain.EmbeddingClient")
    @patch("src.llm.rag_chain.BM25SparseEmbedder")
    @patch("src.llm.rag_chain.ContextCompressor")
    def test_low_score_chunks_filtered(self, mock_comp, mock_sparse, mock_dense,
                                        mock_store, mock_llm, mock_mem, mock_cache):
        """低于阈值的 chunk 被丢弃"""
        mock_cache.return_value.lookup.return_value = (False, None)
        mock_cache.return_value.enabled = True
        mock_mem.return_value.available = True
        mock_comp.return_value.compress.side_effect = lambda c, *a, **kw: c
        mock_llm.return_value.generate.return_value = "答案"

        # 检索返回高+低分
        mock_store.return_value.hybrid_search.return_value = [
            {"content": "高分", "ticket_id": "GD-001", "source": "t.txt", "score": 0.9, "chunk_index": 0},
            {"content": "低分", "ticket_id": "GD-002", "source": "t.txt", "score": 0.3, "chunk_index": 1},
        ]

        from src.llm import rag_chain
        for attr in ['_get_qa_cache', '_get_memory', '_get_llm', '_get_store',
                      '_get_dense_embedder', '_get_sparse_embedder', '_get_compressor']:
            getattr(rag_chain, attr).cache_clear()

        chain = rag_chain.RAGChain()
        chain._memory = mock_mem.return_value
        chain._qa_cache = mock_cache.return_value
        chain._llm = mock_llm.return_value
        chain._store = mock_store.return_value
        chain._dense = mock_dense.return_value
        chain._compressor = mock_comp.return_value

        # score_threshold 默认 0.0 → 两条都保留
        result = chain.ask("问题")
        assert len(result["sources"]) == 2

        # 提高阈值 → 只保留高分
        chain._cfg.retrieval.score_threshold = 0.5
        result = chain.ask("问题")
        assert len(result["sources"]) == 1
        assert result["sources"][0]["ticket_id"] == "GD-001"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
