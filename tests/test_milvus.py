"""
Milvus 数据写入与检索测试 — pytest / 直接运行 两用

用途:
  1. 写入模拟工单测试数据（write_test_data 函数可独立调用）
  2. 验证写入后检索功能正常

运行方式:
  pytest tests/test_milvus.py -v              # 全量测试
  python tests/test_milvus.py                 # 直接运行（写入+搜索）
  python tests/test_milvus.py --write-only    # 仅写入数据，跳过检索测试
  python tests/test_milvus.py --write-only --rebuild  # 删库重建写入
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding.embedding_client import EmbeddingClient
from src.retrieval.milvus_client import MilvusStore

# pytest 为可选依赖（--write-only 模式不需要）
try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False


# ── 模拟测试数据 ──────────────────────────────

TEST_RECORDS = [
    {
        "content": "【工单编号】GD-TEST-00001\n【设备型号】Siemens SOMATOM Force CT\n【故障现象】CT扫描过程中偶发图像伪影，表现为条状高密度影\n【处理方案】更换探测器模块第3通道，重新校准系统",
        "ticket_id": "GD-TEST-00001",
        "source": "模拟测试数据",
        "chunk_index": 0,
    },
    {
        "content": "【工单编号】GD-TEST-00002\n【设备型号】GE SIGNA Premier MRI\n【故障现象】梯度线圈切换时产生异常啸叫，影响患者体验\n【处理方案】紧固梯度线圈固定螺栓，更换减震垫片",
        "ticket_id": "GD-TEST-00002",
        "source": "模拟测试数据",
        "chunk_index": 1,
    },
    {
        "content": "【工单编号】GD-TEST-00003\n【设备型号】Mindray BS-800 生化分析仪\n【故障现象】样本针吸样量不准确，质控结果偏移超出范围\n【处理方案】清洗样本针管路，更换注射器密封圈，重新定标",
        "ticket_id": "GD-TEST-00003",
        "source": "模拟测试数据",
        "chunk_index": 2,
    },
    {
        "content": "【工单编号】GD-TEST-00004\n【设备型号】Philips EPIQ 7 超声\n【故障现象】开机后屏幕花屏，间歇性自动重启\n【处理方案】检查电源模块输出电压，更换故障电容，更新固件至v3.2.1",
        "ticket_id": "GD-TEST-00004",
        "source": "模拟测试数据",
        "chunk_index": 3,
    },
    {
        "content": "【工单编号】GD-TEST-00005\n【设备型号】Fresenius 4008S 血液透析机\n【故障现象】透析液电导率持续偏高报警，治疗被迫中断\n【处理方案】清洗电导率传感器探头，校准A/B液配比，更换细菌过滤器",
        "ticket_id": "GD-TEST-00005",
        "source": "模拟测试数据",
        "chunk_index": 4,
    },
]


# ── 数据写入（独立函数，不依赖 pytest）───────

def write_test_data(store: MilvusStore = None, rebuild: bool = False) -> int:
    """
    向 Milvus 写入模拟测试数据。

    参数:
      store:   MilvusStore 实例（可选，不传则自动创建）
      rebuild: True=先删库重建，False=增量追加

    返回: 写入的向量条数
    """
    if store is None:
        store = MilvusStore()

    # 建库
    store.create_collection(drop_if_exists=rebuild)

    before = store.num_entities
    print(f"\n  Collection: {store.collection_name}（当前 {before} 条）")

    # 向量化
    print("  [1/2] 向量化测试数据...")
    emb = EmbeddingClient()
    contents = [r["content"] for r in TEST_RECORDS]
    vectors = emb.embed_batch(contents)
    print(f"        完成: {len(vectors)} 条, 维度={len(vectors[0])}")

    # 写入
    print("  [2/2] 写入 Milvus...")
    written = store.insert(
        contents=contents,
        vectors=vectors,
        ticket_ids=[r["ticket_id"] for r in TEST_RECORDS],
        sources=[r["source"] for r in TEST_RECORDS],
        chunk_indices=[r["chunk_index"] for r in TEST_RECORDS],
    )
    print(f"        已写入: {written} 条")
    print(f"        当前总量: {store.num_entities} 条\n")
    return written


# ── pytest 测试用例（仅当 pytest 可用时注册）───

if _HAS_PYTEST:

    @pytest.fixture(scope="module")
    def store():
        """所有测试共享同一个 Milvus 连接，确保 collection 就绪"""
        s = MilvusStore()
        if not s.collection_exists():
            s.create_collection(drop_if_exists=False)
        return s

    class TestMilvusWrite:
        """写入 + 验证"""

        def test_write_data(self, store):
            """测试 1: 写入 5 条模拟工单数据"""
            written = write_test_data(store)
            assert written == len(TEST_RECORDS)
            print(f"  [OK] 成功写入 {written} 条测试数据")

        def test_count_after_write(self, store):
            """测试 2: 写入后 num_entities >= 5"""
            count = store.num_entities
            assert count >= len(TEST_RECORDS)
            print(f"  [OK] 当前总量: {count} 条")

    class TestMilvusSearch:
        """检索验证"""

        def test_search_ct_pseudo_image(self, store):
            """测试 3: 搜索"CT图像伪影"应命中 GD-TEST-00001"""
            emb = EmbeddingClient()
            query_vec = emb.embed("CT扫描图像伪影问题")
            results = store.search(query_vec, top_k=3)

            assert len(results) > 0
            ticket_ids = [r["ticket_id"] for r in results]
            print(f"  [OK] 命中 {len(results)} 条: {ticket_ids}")
            assert "GD-TEST-00001" in ticket_ids

        def test_search_mri_noise(self, store):
            """测试 4: 搜索"MRI异常啸叫"应命中 GD-TEST-00002"""
            emb = EmbeddingClient()
            query_vec = emb.embed("MRI梯度线圈异常啸叫")
            results = store.search(query_vec, top_k=3)

            assert len(results) > 0
            ticket_ids = [r["ticket_id"] for r in results]
            print(f"  [OK] 命中 {len(results)} 条: {ticket_ids}")
            assert "GD-TEST-00002" in ticket_ids

        def test_search_dialysis_conductivity(self, store):
            """测试 5: 搜索"透析液电导率"应命中 GD-TEST-00005"""
            emb = EmbeddingClient()
            query_vec = emb.embed("透析液电导率偏高报警")
            results = store.search(query_vec, top_k=3)

            assert len(results) > 0
            ticket_ids = [r["ticket_id"] for r in results]
            print(f"  [OK] 命中 {len(results)} 条: {ticket_ids}")
            assert "GD-TEST-00005" in ticket_ids

        def test_filter_by_ticket_id(self, store):
            """测试 6: 标量过滤 — 只搜 GD-TEST-00003"""
            emb = EmbeddingClient()
            query_vec = emb.embed("生化分析仪故障")
            results = store.search(
                query_vec, top_k=5,
                expr='ticket_id == "GD-TEST-00003"',
            )
            assert len(results) >= 1
            for r in results:
                assert r["ticket_id"] == "GD-TEST-00003"
            print(f"  [OK] 过滤命中 {len(results)} 条，全部属于 GD-TEST-00003")

        def test_search_with_score(self, store):
            """测试 7: 搜索结果包含相似度分数"""
            emb = EmbeddingClient()
            query_vec = emb.embed("医疗设备故障排查")
            results = store.search(query_vec, top_k=3)

            for r in results:
                assert "score" in r
                assert isinstance(r["score"], float)
            scores = [f"{r['score']:.4f}" for r in results]
            print(f"  [OK] 所有结果均含相似度分数: {scores}")

        def test_empty_search(self, store):
            """测试 8: 不存在的工单号过滤应返回空"""
            emb = EmbeddingClient()
            query_vec = emb.embed("图像伪影")
            results = store.search(
                query_vec, top_k=3,
                expr='ticket_id == "GD-NOTEXIST-99999"',
            )
            assert len(results) == 0
            print("  [OK] 不存在工单过滤返回空")


# ── 直接运行入口 ─────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Milvus 测试数据写入与检索")
    parser.add_argument("--write-only", action="store_true",
                        help="仅写入测试数据，不运行检索测试")
    parser.add_argument("--rebuild", action="store_true",
                        help="删库重建后写入")
    args = parser.parse_args()

    if args.write_only:
        print("=" * 56)
        print("  写入 Milvus 测试数据")
        print("=" * 56)
        written = write_test_data(rebuild=args.rebuild)
        print(f"[DONE] 写入完成: {written} 条")
    else:
        if not _HAS_PYTEST:
            print("错误: 运行测试需要 pytest，请先安装: pip install pytest")
            sys.exit(1)
        pytest.main([__file__, "-v", "-s"])
