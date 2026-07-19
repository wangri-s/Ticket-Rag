"""
BM25 稀疏向量生成器 — 用于关键字检索

分词链路: 原始文本 → jieba 分词 → BM25 词频统计 → 稀疏向量
  1. jieba.cut(text) → tokens 列表（中文分词）
  2. BM25EmbeddingFunction.fit(corpus) → 构建词表 + 计算 IDF
  3. encode_*(texts) → scipy sparse matrix → Milvus SPARSE_FLOAT_VECTOR

用法:
  embedder = BM25SparseEmbedder()
  embedder.fit(corpus)                          # 先训练（构建词表 + IDF）
  sparse_vecs = embedder.encode_documents(texts) # → list[dict[int, float]]
  query_vecs = embedder.encode_queries(queries)  # → list[dict[int, float]]
"""

import logging
from pathlib import Path
from typing import Optional

import jieba
import sys as _sys
from types import ModuleType as _ModuleType

# ── Windows DLL 兼容处理 ──────────────────────
# milvus_model/__init__.py 在模块级别执行:
#   DefaultEmbeddingFunction = dense.onnx.OnnxEmbeddingFunction
# dense/onnx 都是 LazyModule，逐级触发 importlib.import_module，最终加载
# onnx.py → import onnxruntime。部分 Windows 环境 DLL 初始化失败。
#
# 解决方案: 在 import milvus_model 之前，先把假的 dense/onnx 模块塞进
# sys.modules。这样 LazyModule._load() 触发 importlib.import_module 时，
# Python 发现模块已在 sys.modules 中，直接返回 stub，不碰真实文件。
# 同时把 onnxruntime 也 stub 掉，防止任何代码路径意外加载它。

# 1. 构建假 milvus_model.dense* 模块树，阻止加载 onnxruntime DLL
_onnx_stub = _ModuleType("milvus_model.dense.onnx")
_onnx_stub.OnnxEmbeddingFunction = type("OnnxEmbeddingFunction", (), {})

_dense_stub = _ModuleType("milvus_model.dense")
_dense_stub.onnx = _onnx_stub

# 2. 预注册到 sys.modules（必须在 import milvus_model 之前）
#    milvus_model/__init__.py: DefaultEmbeddingFunction = dense.onnx.OnnxEmbeddingFunction
#    → LazyModule._load() → importlib.import_module("milvus_model.dense")
#    → 发现 sys.modules 已有 → 直接返回 stub，不碰 onnx.py → 不碰 onnxruntime
_sys.modules["milvus_model.dense"] = _dense_stub
_sys.modules["milvus_model.dense.onnx"] = _onnx_stub

# 3. 安全导入 BM25
from milvus_model.sparse.bm25 import BM25EmbeddingFunction  # noqa: E402
from milvus_model.sparse.bm25.tokenizers import build_default_analyzer  # noqa: E402

logger = logging.getLogger(__name__)

# ── 医疗领域自定义词典 ────────────────────────

# 确保医学术语不被错误切分（如"电导率"不被切成"电/导率"）
_MEDICAL_TERMS = [
    # 设备名称
    "血液透析机", "生化分析仪", "超声诊断仪", "高压灭菌器",
    "麻醉机", "监护仪", "数字化X射线", "核磁共振",
    # 医学术语
    "电导率", "伪影", "定标", "透析液", "声学堆栈",
    "凸阵探头", "线阵探头", "梯度线圈", "球管",
    "呼吸回路", "真空泵", "探测器",
    # 故障术语
    "分辨率下降", "图像模糊", "数据断连", "漏气测试",
    "预览延迟", "响应超时", "灭菌程序",
]

_JIEBA_INITIALIZED = False


def _init_jieba():
    """加载医疗领域自定义词典，确保术语不被错误切分"""
    global _JIEBA_INITIALIZED
    if _JIEBA_INITIALIZED:
        return
    for term in _MEDICAL_TERMS:
        jieba.add_word(term)
    _JIEBA_INITIALIZED = True
    logger.info(f"jieba 分词器已加载 {len(_MEDICAL_TERMS)} 个医疗领域术语")


def _sparse_to_dict(sparse_vec) -> dict:
    """
    scipy sparse matrix → Milvus sparse dict {int(dim): float(val)}

    兼容 csr_array（encode_documents 输出）和 coo_array（encode_queries 输出）。
    """
    # csr_array: .indices / .data
    if hasattr(sparse_vec, "indices"):
        return {int(i): float(v) for i, v in zip(sparse_vec.indices, sparse_vec.data)}
    # coo_array: .col / .data
    return {int(c): float(v) for c, v in zip(sparse_vec.col, sparse_vec.data)}


class BM25SparseEmbedder:
    """
    BM25 稀疏向量生成器 — jieba 中文分词

    分词链路:
      文本 → jieba.cut() → tokens → BM25 词频 + IDF → 稀疏向量

    典型流程:
      embedder = BM25SparseEmbedder()
      embedder.fit(all_chunks)                        # 训练 IDF
      sparse_vecs = embedder.encode_documents(chunks) # 写入 Milvus

      # 检索时:
      q_vec = embedder.encode_queries(["CT伪影"])     # 查询向量化
      store.sparse_search(q_vec[0])
    """

    def __init__(self, language: str = "zh"):
        # 初始化 jieba + 加载医疗领域术语词典
        _init_jieba()

        # build_default_analyzer(language="zh") 底层使用 JiebaTokenizer
        # 等价于: jieba.cut(text) → stopword filter → lowercase
        self.analyzer = build_default_analyzer(language=language)
        self._bm25 = BM25EmbeddingFunction(self.analyzer)
        self._is_fitted = False
        logger.info(
            f"BM25 稀疏嵌入器已初始化: tokenizer=jieba, language={language}"
        )

    # ── 训练 ──────────────────────────────────

    def fit(self, corpus: list[str]) -> "BM25SparseEmbedder":
        """用全量语料训练 BM25（构建词表 + IDF）"""
        if not corpus:
            raise ValueError("训练语料不能为空")
        self._bm25.fit(corpus)
        self._is_fitted = True
        logger.info(f"BM25 训练完成: {len(corpus)} 篇文档")
        return self

    # ── 编码 ──────────────────────────────────

    def encode_documents(self, texts: list[str]) -> list[dict]:
        """文档编码 → Milvus 可用的 sparse dict 列表"""
        self._ensure_fitted()
        sparse_vecs = self._bm25.encode_documents(texts)
        return [_sparse_to_dict(v) for v in sparse_vecs]

    def encode_queries(self, queries: list[str]) -> list[dict]:
        """查询编码 → Milvus 可用的 sparse dict 列表"""
        self._ensure_fitted()
        sparse_vecs = self._bm25.encode_queries(queries)
        return [_sparse_to_dict(v) for v in sparse_vecs]

    # ── 内部 ──────────────────────────────────

    def _ensure_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("BM25 尚未训练，请先调用 fit()")
