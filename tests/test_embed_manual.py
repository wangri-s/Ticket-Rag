"""
Embedding 功能验证 — pytest / 直接运行 两用

PyCharm 右键 Run 自动用 pytest 发现用例；
终端 python tests/test_embed_manual.py 直接运行也行。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.embedding.embedding_client import EmbeddingClient


@pytest.fixture(scope="module")
def client():
    """所有测试共享同一个客户端实例，避免重复初始化"""
    return EmbeddingClient()


def test_single_embed(client):
    """测试 1: 单条文本向量化"""
    vec = client.embed("CT扫描过程中偶发图像伪影")
    assert len(vec) == 1536
    assert isinstance(vec[0], float)


def test_batch_embed(client):
    """测试 2: 批量文本向量化"""
    texts = ["MRI扫描梯度线圈异常啸叫", "透析液电导率偏高报警", "挂号系统数据库连接超时"]
    vecs = client.embed_batch(texts)
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 1536


def test_empty_text_raises(client):
    """测试 3: 空文本应抛出 ValueError"""
    with pytest.raises(ValueError, match="输入文本为空"):
        client.embed("")


def test_dimension_consistency(client):
    """测试 4: 多次调用输出的向量维度一致"""
    v1 = client.embed("测试文本A")
    v2 = client.embed("测试文本B")
    assert len(v1) == 1536
    assert len(v2) == 1536
    assert len(v1) == len(v2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
