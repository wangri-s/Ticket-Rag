"""
Milvus 向量数据库操作封装

基于 pymilvus MilvusClient（新 API），避免 ORM 风格连接。
职责:
  - 连接管理：uri 直连，无需手动 connect/disconnect
  - Collection 生命周期：创建/删除/检查是否存在
  - 数据写入：插入向量 + 元数据，自动创建索引
  - 数据检索：语义搜索，返回向量 + 元数据 + 相似度分数
"""

import logging
from typing import Optional

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)
from pymilvus.milvus_client.index import IndexParams

from src.config import MilvusConfig, get_config

logger = logging.getLogger(__name__)


def _build_schema(dim: int) -> CollectionSchema:
    """构建 Collection Schema（FieldSchema 对象列表）"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True, description="主键（自增）"),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000, description="文本块内容"),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim, description="稠密向量（语义）"),
        FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR, description="稀疏向量（BM25 关键字）"),
        FieldSchema(name="ticket_id", dtype=DataType.VARCHAR, max_length=32, description="工单编号"),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512, description="来源文件"),
        FieldSchema(name="chunk_index", dtype=DataType.INT64, description="块序号"),
    ]
    return CollectionSchema(fields, description="医疗设备运维工单知识库")


class MilvusStore:
    """
    Milvus 向量库操作封装

    用法:
      store = MilvusStore()
      store.create_collection()              # 首次使用调用一次
      store.insert(contents, vectors, ...)   # 写入向量
      results = store.search(query_vec)      # 语义搜索
    """

    def __init__(self, config: Optional[MilvusConfig] = None):
        cfg = config or get_config().milvus
        self.collection_name = cfg.collection_name
        self.vector_dim = cfg.vector_dim
        self.metric_type = cfg.metric_type
        self.index_type = cfg.index_type
        self.index_params = cfg.index_params
        self.search_params = cfg.search_params
        self.timeout = cfg.timeout
        self.max_retries = cfg.max_retries

        self._client = MilvusClient(
            uri=f"http://{cfg.host}:{cfg.port}",
            timeout=self.timeout,
        )

    # ── Collection 管理 ───────────────────────

    def collection_exists(self) -> bool:
        """检查 collection 是否已存在"""
        return self._client.has_collection(self.collection_name)

    def create_collection(self, drop_if_exists: bool = False) -> None:
        """
        创建 collection（若已存在且 drop_if_exists=True 则先删后建）。

        Schema 字段:
          id          — 主键（自增 int64）
          content     — 文本块内容（varchar 2000）
          embedding   — 向量（float vector, 1536 维）
          ticket_id   — 工单编号（varchar 32）
          source      — 来源文件（varchar 512）
          chunk_index — 块序号（int64）
        """
        if self.collection_exists():
            if drop_if_exists:
                logger.info(f"删除已有 collection: {self.collection_name}")
                self._client.drop_collection(self.collection_name)
            else:
                logger.info(f"Collection 已存在: {self.collection_name}")
                return

        # 构建 FieldSchema → CollectionSchema
        schema = _build_schema(self.vector_dim)

        # 创建
        self._client.create_collection(
            collection_name=self.collection_name,
            dimension=self.vector_dim,
            metric_type=self.metric_type,
            schema=schema,
        )

        # 创建稠密索引 (IVF_FLAT)
        index_params = IndexParams()
        index_params.add_index(
            field_name="embedding",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params=self.index_params,
        )
        self._client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )

        # 创建稀疏索引 (SPARSE_INVERTED_INDEX)
        sparse_index_params = IndexParams()
        sparse_index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        self._client.create_index(
            collection_name=self.collection_name,
            index_params=sparse_index_params,
        )

        # 加载到内存，否则无法搜索
        self._client.load_collection(self.collection_name)

        logger.info(
            f"Collection 已创建: {self.collection_name} "
            f"(dim={self.vector_dim}, index={self.index_type})"
        )

    # ── 数据写入 ──────────────────────────────

    def insert(
        self,
        contents: list[str],
        vectors: list[list[float]],
        ticket_ids: list[str],
        sources: list[str],
        chunk_indices: list[int],
        sparse_vectors: list[dict] = None,
    ) -> int:
        """
        批量写入向量 + 元数据。

        参数:
          sparse_vectors: 可选，BM25 稀疏向量列表 [{int(dim): float(val)}, ...]

        返回: 插入的行数
        """
        if not contents:
            return 0

        n = len(contents)

        # 构建 data 列表: [{"field": value}, ...]
        data = []
        for i in range(n):
            row = {
                "content": contents[i],
                "embedding": vectors[i],
                "ticket_id": ticket_ids[i],
                "source": sources[i],
                "chunk_index": int(chunk_indices[i]),
            }
            if sparse_vectors and i < len(sparse_vectors):
                row["sparse_embedding"] = sparse_vectors[i]
            data.append(row)

        result = self._client.insert(
            collection_name=self.collection_name,
            data=data,
        )

        count = result["insert_count"]

        # flush 确保数据持久化、对搜索可见
        self._client.flush(self.collection_name)

        logger.info(f"已写入 {count} 条向量到 {self.collection_name}")
        return count

    # ── 数据检索 ──────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        expr: Optional[str] = None,
    ) -> list[dict]:
        """
        语义搜索：输入一个向量，返回最相似的 top_k 个结果。

        参数:
          query_vector: 查询向量 (1536 维)
          top_k:        返回条数
          expr:         标量过滤，如 'ticket_id == "GD-2026-03001"'

        返回:
          [{id, score, content, ticket_id, source, chunk_index}, ...]
        """
        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="embedding",
            limit=top_k,
            filter=expr or "",
            output_fields=["content", "ticket_id", "source", "chunk_index"],
            search_params={"metric_type": self.metric_type, "params": self.search_params},
        )

        # results 格式: [[{id, distance, entity: {...}}, ...]]
        hits = []
        for batch in results:
            for hit in batch:
                entity = hit.get("entity", {})
                hits.append({
                    "id": hit["id"],
                    "score": hit["distance"],
                    "content": entity.get("content", ""),
                    "ticket_id": entity.get("ticket_id", ""),
                    "source": entity.get("source", ""),
                    "chunk_index": entity.get("chunk_index", -1),
                })

        return hits

    # ── 稀疏向量检索（关键字）───────────────

    def sparse_search(
        self,
        query_sparse_vector: dict,
        top_k: int = 5,
        expr: Optional[str] = None,
    ) -> list[dict]:
        """
        关键字检索：输入 BM25 稀疏向量，返回匹配结果。

        参数:
          query_sparse_vector: BM25 编码后的查询稀疏向量 {int(dim): float(val)}
          top_k:               返回条数
          expr:                标量过滤表达式

        返回:
          [{id, score, content, ticket_id, source, chunk_index}, ...]
        """
        results = self._client.search(
            collection_name=self.collection_name,
            data=[query_sparse_vector],
            anns_field="sparse_embedding",
            limit=top_k,
            filter=expr or "",
            output_fields=["content", "ticket_id", "source", "chunk_index"],
            search_params={"metric_type": "IP"},
        )

        hits = []
        for batch in results:
            for hit in batch:
                entity = hit.get("entity", {})
                hits.append({
                    "id": hit["id"],
                    "score": hit["distance"],
                    "content": entity.get("content", ""),
                    "ticket_id": entity.get("ticket_id", ""),
                    "source": entity.get("source", ""),
                    "chunk_index": entity.get("chunk_index", -1),
                })

        return hits

    # ── 混合检索（语义 + 关键字）─────────────

    def hybrid_search(
        self,
        query_vector: list[float],
        query_sparse_vector: dict,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        expr: Optional[str] = None,
    ) -> list[dict]:
        """
        混合检索：语义向量 + 关键字稀疏向量 加权重排。

        参数:
          query_vector:        稠密查询向量 (1536 维)
          query_sparse_vector: BM25 稀疏查询向量 {int(dim): float(val)}
          top_k:               返回条数
          dense_weight:        语义权重（0~1）
          sparse_weight:       关键字权重（0~1）
          expr:                标量过滤表达式

        返回:
          [{id, score, content, ticket_id, source, chunk_index}, ...]
        """
        from pymilvus import AnnSearchRequest, WeightedRanker

        # 稠密检索请求
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": self.metric_type, "params": self.search_params},
            limit=top_k * 2,
            expr=expr or "",
        )

        # 稀疏检索请求
        sparse_req = AnnSearchRequest(
            data=[query_sparse_vector],
            anns_field="sparse_embedding",
            param={"metric_type": "IP"},
            limit=top_k * 2,
            expr=expr or "",
        )

        # 加权重排
        ranker = WeightedRanker(dense_weight, sparse_weight)

        results = self._client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=top_k,
            output_fields=["content", "ticket_id", "source", "chunk_index"],
        )

        hits = []
        for batch in results:
            for hit in batch:
                entity = hit.get("entity", {})
                hits.append({
                    "id": hit["id"],
                    "score": hit["distance"],
                    "content": entity.get("content", ""),
                    "ticket_id": entity.get("ticket_id", ""),
                    "source": entity.get("source", ""),
                    "chunk_index": entity.get("chunk_index", -1),
                })

        return hits

    # ── 统计 ──────────────────────────────────

    @property
    def num_entities(self) -> int:
        """当前 collection 中的向量总数"""
        if not self.collection_exists():
            return 0
        stats = self._client.get_collection_stats(self.collection_name)
        return stats.get("row_count", 0)

    def drop_collection(self) -> None:
        """删除 collection（谨慎使用）"""
        if self.collection_exists():
            self._client.drop_collection(self.collection_name)
            logger.info(f"Collection 已删除: {self.collection_name}")
