"""
共享测试 fixtures — 项目级通用 Mock 与测试数据
"""
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 确保项目根目录在 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── 测试数据: 标准检索结果 ──────────────────────

@pytest.fixture
def sample_chunks() -> list[dict]:
    """构造标准检索结果（4 个 chunk，分数递减）"""
    return [
        {
            "content": "CT 扫描偶发图像伪影，表现为条状低密度阴影。排查发现探测器第3排第12通道灵敏度下降15%，更换该通道探测器模块后伪影消失。",
            "ticket_id": "GD-2026-03001",
            "source": "data/documents/tickets_01.txt",
            "score": 0.95,
            "chunk_index": 0,
        },
        {
            "content": "MRI 梯度线圈发出异常啸叫声，约2kHz。X轴梯度放大器直流母线滤波电容C12鼓包，更换同规格电容（450V 470μF）后恢复正常。",
            "ticket_id": "GD-2026-03002",
            "source": "data/documents/tickets_01.txt",
            "score": 0.82,
            "chunk_index": 1,
        },
        {
            "content": "透析液电导率偏高报警。A液泵管老化导致吸液量不准，更换泵管并重新定标后电导率恢复正常。建议每季度检查A/B液管路。",
            "ticket_id": "GD-2026-03003",
            "source": "data/documents/tickets_01.txt",
            "score": 0.68,
            "chunk_index": 0,
        },
        {
            "content": "CT 日常维护保养流程：每周清洁滤网并检查散热风扇运转情况，每月校准探测器各通道灵敏度并记录偏差值，每季度检查球管灯丝电流和旋转阳极状态。" * 12,
            "ticket_id": "GD-2026-03004",
            "source": "data/documents/tickets_02.txt",
            "score": 0.55,
            "chunk_index": 0,
        },
    ]


@pytest.fixture
def single_chunk() -> list[dict]:
    """单个 chunk"""
    return [{
        "content": "CT 图像伪影排查：探测器校准漂移是常见原因。",
        "ticket_id": "GD-2026-03001",
        "source": "data/documents/tickets_01.txt",
        "score": 0.90,
        "chunk_index": 0,
    }]


# ── 模拟 Embedding 向量 ─────────────────────────

@pytest.fixture
def fixed_embedding() -> list[float]:
    """固定 1536 维向量（用于 cosine 计算验证）"""
    random.seed(42)
    return [random.random() for _ in range(1536)]


@pytest.fixture
def another_embedding() -> list[float]:
    """另一个固定向量（与 fixed_embedding 不同）"""
    random.seed(99)
    return [random.random() for _ in range(1536)]


# ── Mock: Embedding Client ─────────────────────

@pytest.fixture
def mock_embedding_client():
    """Mock EmbeddingClient，返回固定向量"""
    with patch("src.embedding.embedding_client.EmbeddingClient") as mock:
        instance = mock.return_value
        random.seed(42)
        instance.embed.return_value = [random.random() for _ in range(1536)]
        yield instance


# ── Mock: Redis ─────────────────────────────────

@pytest.fixture
def mock_redis():
    """Mock Redis 客户端 — ping 返回 True"""
    with patch("redis.Redis") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        client.ping.return_value = True
        yield client


@pytest.fixture
def mock_redis_unavailable():
    """Mock Redis 客户端 — ping 抛异常（不可用）"""
    with patch("redis.Redis") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        client.ping.side_effect = ConnectionError("Redis not available")
        yield client


# ── Mock: LLM Client ────────────────────────────

@pytest.fixture
def mock_llm_client():
    """Mock LLMClient — 返回预设答案"""
    with patch("src.llm.llm_client.LLMClient") as mock_cls:
        instance = mock_cls.return_value
        instance.generate.return_value = "根据工单 GD-2026-03001，建议先检查探测器阵列校准状态。"
        instance.generate_stream.return_value = iter(["根据", "工单", "GD-", "2026-", "03001", "，", "建议", "检查", "探测器。"])
        yield instance


# ── Mock: Milvus Store ──────────────────────────

@pytest.fixture
def mock_milvus_store(sample_chunks):
    """Mock MilvusStore"""
    with patch("src.retrieval.milvus_client.MilvusStore") as mock_cls:
        instance = mock_cls.return_value
        instance.search.return_value = sample_chunks[:3]
        instance.sparse_search.return_value = sample_chunks[1:4]
        instance.hybrid_search.return_value = sample_chunks[:3]
        yield instance


# ── Mock: Memory Manager ────────────────────────

@pytest.fixture
def mock_memory_manager():
    """Mock MemoryManager"""
    with patch("src.memory.memory_manager.MemoryManager") as mock_cls:
        instance = mock_cls.return_value
        instance.available = True
        instance.add_message.return_value = {"turn": 1, "summary": None}
        instance.get_context.return_value = {
            "messages": [],
            "summary": None,
            "turn_count": 0,
            "has_memory": False,
        }
        instance.build_memory_prompt.return_value = ""
        yield instance


# ── Mock: QA Cache ──────────────────────────────

@pytest.fixture
def mock_qa_cache():
    """Mock QACache — 默认未命中"""
    with patch("src.memory.qa_cache.QACache") as mock_cls:
        instance = mock_cls.return_value
        instance.enabled = True
        instance.available = True
        instance.lookup.return_value = (False, None)
        instance.store.return_value = True
        yield instance


# ── Mock: Context Compressor ────────────────────

@pytest.fixture
def mock_compressor():
    """Mock ContextCompressor — 透传不压缩"""
    with patch("src.utils.context_compressor.ContextCompressor") as mock_cls:
        instance = mock_cls.return_value
        instance.compress.side_effect = lambda chunks, *a, **kw: chunks
        yield instance


# ── Mock: Reranker ──────────────────────────────

@pytest.fixture
def mock_reranker():
    """Mock Reranker"""
    with patch("src.retrieval.reranker.Reranker") as mock_cls:
        instance = mock_cls.return_value
        instance.rerank.side_effect = lambda q, cands, top_n: cands[:top_n]
        yield instance


# ── Mock: Query Processor ───────────────────────

@pytest.fixture
def mock_query_processor():
    """Mock QueryProcessor"""
    with patch("src.retrieval.query_processor.QueryProcessor") as mock_cls:
        instance = mock_cls.return_value
        instance.rewrite.return_value = "CT 图像伪影 探测器校准"
        yield instance
