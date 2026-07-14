"""
测试配置加载器 src/config.py
"""
import os
import tempfile

import pytest

# 确保项目根目录在 path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig, get_config, _resolve_env


class TestEnvResolution:
    """环境变量占位符解析"""

    def test_resolve_with_default(self):
        # .env 中可能已设置 MILVUS_HOST，先保存再清空
        saved = os.environ.pop("MILVUS_HOST", None)
        try:
            assert "localhost" == _resolve_env("${MILVUS_HOST:-localhost}")
        finally:
            if saved:
                os.environ["MILVUS_HOST"] = saved

    def test_resolve_from_env(self):
        os.environ["TEST_VAR"] = "hello"
        assert "hello" == _resolve_env("${TEST_VAR:-default}")
        del os.environ["TEST_VAR"]

    def test_resolve_without_default(self):
        os.environ["TEST_VAR"] = "world"
        assert "world" == _resolve_env("${TEST_VAR}")
        del os.environ["TEST_VAR"]

    def test_pure_value_unchanged(self):
        assert 42 == _resolve_env(42)
        assert "localhost" == _resolve_env("localhost")

    def test_nested_defaults(self):
        saved = os.environ.pop("MILVUS_HOST", None)
        saved_port = os.environ.pop("MILVUS_PORT", None)
        try:
            result = _resolve_env("${MILVUS_HOST:-localhost}:${MILVUS_PORT:-19530}")
            assert "localhost:19530" == result
        finally:
            if saved:
                os.environ["MILVUS_HOST"] = saved
            if saved_port:
                os.environ["MILVUS_PORT"] = saved_port


class TestConfigLoading:
    """配置文件加载"""

    def test_config_parses_without_error(self):
        cfg = AppConfig()
        assert cfg.milvus.collection_name == "medical_chunks"
        assert cfg.milvus.vector_dim == 1536
        assert cfg.embedding.model == "text-embedding-v1"
        assert cfg.llm.model == "qwen-max"

    def test_milvus_uri(self):
        cfg = AppConfig()
        assert cfg.milvus.uri.startswith("http://")
        assert "19530" in cfg.milvus.uri

    def test_retrieval_defaults(self):
        cfg = AppConfig()
        assert cfg.retrieval.top_k == 5
        assert cfg.retrieval.score_threshold == 0.0

    def test_chunking_config(self):
        cfg = AppConfig()
        assert cfg.chunking.chunk_size == 500
        assert cfg.chunking.chunk_overlap == 50
        assert cfg.chunking.strategy == "recursive"
        assert len(cfg.chunking.separators) >= 5

    def test_llm_temperature(self):
        cfg = AppConfig()
        assert cfg.llm.temperature == 0.1
        assert cfg.llm.max_tokens == 1500

    def test_singleton_same_instance(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_validate_detects_missing_key(self):
        cfg = AppConfig()
        # 如果没有设置真实 API Key，validate 应该返回非空
        missing = cfg.validate()
        # DASHSCOPE_API_KEY 是当前占位值时会报缺失
        if "sk-your-api-key-here" in cfg.embedding.dashscope_api_key:
            assert len(missing) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

