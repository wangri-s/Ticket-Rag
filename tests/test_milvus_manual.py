"""
Milvus 连接与基本操作验证 — pytest / 直接运行 两用

PyCharm 右键 Run 自动用 pytest 发现用例；
终端 python tests/test_milvus_manual.py 直接运行也行。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.retrieval.milvus_client import MilvusStore


@pytest.fixture(scope="module")
def store():
    """所有测试共享同一个 Milvus 连接，避免重复初始化"""
    s = MilvusStore()
    # drop_if_exists=True 确保 schema 与当前代码一致
    s.create_collection(drop_if_exists=True)
    return s


def test_connection(store):
    """测试 1: Milvus 连接成功，能获取 collection 信息"""
    assert store.collection_exists()
    assert store.collection_name == "medical_chunks"
    print(f"\n  [OK] 连接成功，collection='{store.collection_name}'")


def test_has_collection(store):
    """测试 2: has_collection 返回 True"""
    assert store.collection_exists() is True


def test_insert_and_search(store):
    """测试 3: 写入一条向量 + 搜索验证"""
    content = "CT扫描过程中偶发图像伪影，排查发现探测器模块老化"
    ticket_id = "GD-TEST-00001"
    source = "测试文档.txt"
    chunk_index = 0

    # 先用 embedding 客户端生成向量
    from src.embedding.embedding_client import EmbeddingClient

    emb = EmbeddingClient()
    vec = emb.embed(content)

    # 写入
    count = store.insert(
        contents=[content],
        vectors=[vec],
        ticket_ids=[ticket_id],
        sources=[source],
        chunk_indices=[chunk_index],
    )
    assert count == 1
    print(f"\n  [OK] 已写入 {count} 条")

    # 搜索
    results = store.search(query_vector=vec, top_k=3)
    assert len(results) >= 1
    assert results[0]["content"] == content
    assert results[0]["ticket_id"] == ticket_id
    assert results[0]["source"] == source
    print(f"  [OK] 搜索命中 {len(results)} 条，top1 内容匹配")


def test_num_entities(store):
    """测试 4: num_entities 返回正确的数量"""
    count = store.num_entities
    assert count >= 1
    print(f"\n  [OK] collection 中共有 {count} 条向量")


def test_search_with_filter(store):
    """测试 5: 带标量过滤的搜索"""
    from src.embedding.embedding_client import EmbeddingClient

    emb = EmbeddingClient()
    vec = emb.embed("图像伪影")

    # 只搜测试工单
    results = store.search(
        query_vector=vec,
        top_k=3,
        expr='ticket_id == "GD-TEST-00001"',
    )
    assert len(results) >= 1
    for r in results:
        assert r["ticket_id"] == "GD-TEST-00001"
    print(f"\n  [OK] 过滤搜索命中 {len(results)} 条，全部属于 GD-TEST-00001")


def test_search_no_match_expression(store):
    """测试 6: 用不存在的工单号过滤，应返回空"""
    from src.embedding.embedding_client import EmbeddingClient

    emb = EmbeddingClient()
    vec = emb.embed("图像伪影")

    results = store.search(
        query_vector=vec,
        top_k=3,
        expr='ticket_id == "GD-NOTEXIST-99999"',
    )
    assert len(results) == 0
    print(f"\n  [OK] 不存在工单过滤返回空，符合预期")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
