"""
应用配置加载器
- 从 .env 文件加载环境变量
- 解析 config.yml，替换 ${ENV_VAR:-default} 占位符
- 返回类型化的配置对象
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ---------- 项目根目录 ----------
ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """加载 .env 文件（如果存在），已存在的环境变量优先"""
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def _resolve_env(value: str) -> str:
    """替换字符串中的 ${VAR} 或 ${VAR:-default} 占位符"""
    if not isinstance(value, str):
        return value

    # 匹配 ${VAR} 或 ${VAR:-default}
    pattern = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")

    def _replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default if default is not None else "")

    return pattern.sub(_replacer, value)


def _walk_and_resolve(obj: Any) -> Any:
    """递归遍历配置字典，解析所有字符串中的环境变量"""
    if isinstance(obj, dict):
        return {k: _walk_and_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_resolve(v) for v in obj]
    if isinstance(obj, str):
        return _resolve_env(obj)
    return obj


def _parse_yml(path: Path) -> dict:
    """读取并解析 YAML 配置文件"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _walk_and_resolve(raw)


# ==================== 配置数据类 ====================

class MilvusConfig:
    def __init__(self, data: dict):
        self.host: str = data.get("host", "localhost")
        self.port: int = int(data.get("port", 19530))
        self.collection_name: str = data["collection_name"]
        self.vector_dim: int = data["vector_dim"]
        self.timeout: int = data.get("timeout", 30)
        self.max_retries: int = data.get("max_retries", 3)
        self.index_type: str = data.get("index_type", "IVF_FLAT")
        self.metric_type: str = data.get("metric_type", "IP")
        self.index_params: dict = data.get("index_params", {})
        self.search_params: dict = data.get("search_params", {"nprobe": 16})
        self.consistency_level: str = data.get("consistency_level", "Strong")

    @property
    def uri(self) -> str:
        return f"http://{self.host}:{self.port}"


class EmbeddingConfig:
    def __init__(self, data: dict):
        self.model: str = data["model"]
        self.provider: str = data.get("provider", "dashscope")
        self.dashscope_api_key: str = data["dashscope_api_key"]
        self.dimensions: int = data.get("dimensions", 1536)
        self.batch_size: int = data.get("batch_size", 32)
        self.max_retries: int = data.get("max_retries", 3)
        self.timeout: int = data.get("timeout", 60)
        self.sleep_interval: float = data.get("sleep_interval", 1.0)


class LLMConfig:
    def __init__(self, data: dict):
        self.model: str = data["model"]
        self.provider: str = data.get("provider", "dashscope")
        self.temperature: float = data["temperature"]
        self.max_tokens: int = data["max_tokens"]
        self.dashscope_api_key: str = data["dashscope_api_key"]


class RetrievalConfig:
    def __init__(self, data: dict):
        self.top_k: int = data["top_k"]
        self.score_threshold: float = data["score_threshold"]


class ChunkingConfig:
    def __init__(self, data: dict):
        self.strategy: str = data.get("strategy", "recursive")
        self.chunk_size: int = data["chunk_size"]
        self.chunk_overlap: int = data["chunk_overlap"]
        self.min_chunk_length: int = data["min_chunk_length"]
        self.separators: list[str] = data.get("separators", ["\n\n", "\n", "。", " "])


class AppConfig:
    """应用全局配置"""

    def __init__(self):
        _load_env()
        raw = _parse_yml(ROOT_DIR / "config.yml")

        self.milvus = MilvusConfig(raw["milvus"])
        self.embedding = EmbeddingConfig(raw["embedding"])
        self.llm = LLMConfig(raw["llm"])
        self.retrieval = RetrievalConfig(raw["retrieval"])
        self.chunking = ChunkingConfig(raw["chunking"])

    def validate(self) -> list[str]:
        """校验必填配置项，返回缺失项列表"""
        missing: list[str] = []

        if not self.embedding.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY (embedding & llm)")
        if not self.llm.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY (embedding & llm)")

        return missing


# ==================== 单例 ====================

_config: AppConfig | None = None


def get_config() -> AppConfig:
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config
