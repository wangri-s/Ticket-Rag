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
        # 生成控制
        self.top_p: float = data.get("top_p", 0.8)
        self.seed: int = data.get("seed", 42)
        self.stream: bool = data.get("stream", False)
        # 连接
        self.max_retries: int = data.get("max_retries", 3)
        self.timeout: int = data.get("timeout", 60)
        # RAG
        self.retrieval_top_k: int = data.get("retrieval_top_k", 5)
        self.system_prompt: str = data.get("system_prompt", "")
        self.rag_prompt_template: str = data.get("rag_prompt_template", "")
        self.json_prompt_template: str = data.get("json_prompt_template", "")
        self.fallback_answer: str = data.get("fallback_answer", "抱歉，未找到相关记录。")
        self.output_format: str = data.get("output_format", "text")


class QueryPreprocessConfig:
    def __init__(self, data: dict):
        self.enabled: bool = data.get("enabled", False)
        self.rewrite: bool = data.get("rewrite", True)
        self.multi_query: bool = data.get("multi_query", False)
        self.multi_query_n: int = data.get("multi_query_n", 3)


class RerankConfig:
    def __init__(self, data: dict):
        self.enabled: bool = data.get("enabled", False)
        self.model: str = data.get("model", "qwen3-rerank")
        self.dashscope_api_key: str = data.get("dashscope_api_key", "")
        self.oversample_factor: int = data.get("oversample_factor", 3)
        self.max_retries: int = data.get("max_retries", 3)
        self.timeout: int = data.get("timeout", 30)


class RetrievalConfig:
    def __init__(self, data: dict):
        self.top_k: int = data["top_k"]
        self.score_threshold: float = data.get("score_threshold", 0.0)
        self.default_mode: str = data.get("default_mode", "hybrid")
        # 混合检索权重
        hybrid = data.get("hybrid", {})
        self.hybrid_dense_weight: float = hybrid.get("dense_weight", 0.5)
        self.hybrid_sparse_weight: float = hybrid.get("sparse_weight", 0.5)
        # 查询预处理
        self.query_preprocess = QueryPreprocessConfig(data.get("query_preprocess", {}))
        # 重排序
        self.rerank = RerankConfig(data.get("rerank", {}))


class ChunkingConfig:
    def __init__(self, data: dict):
        self.strategy: str = data.get("strategy", "recursive")
        self.chunk_size: int = data["chunk_size"]
        self.chunk_overlap: int = data["chunk_overlap"]
        self.min_chunk_length: int = data["min_chunk_length"]
        self.separators: list[str] = data.get("separators", ["\n\n", "\n", "。", " "])


class RedisConfig:
    def __init__(self, data: dict):
        self.host: str = data.get("host", "localhost")
        self.port: int = int(data.get("port", 6379))
        self.db: int = int(data.get("db", 0))
        self.password: str = data.get("password", "") or None
        self.ttl: int = int(data.get("ttl", 86400))
        self.max_messages: int = int(data.get("max_messages", 20))
        self.key_prefix: str = data.get("key_prefix", "rag:session")

    @property
    def uri(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class MySQLConfig:
    def __init__(self, data: dict):
        self.host: str = data.get("host", "localhost")
        self.port: int = int(data.get("port", 3306))
        self.user: str = data.get("user", "root")
        self.password: str = data.get("password", "")
        self.database: str = data.get("database", "rag_memory")
        self.charset: str = data.get("charset", "utf8mb4")
        self.pool_size: int = int(data.get("pool_size", 5))
        self.max_retries: int = int(data.get("max_retries", 3))
        self.retry_base_delay: float = float(data.get("retry_base_delay", 0.5))
        self.retry_max_delay: float = float(data.get("retry_max_delay", 5.0))


class KafkaConfig:
    def __init__(self, data: dict):
        self.bootstrap_servers: str = data.get("bootstrap_servers", "localhost:9092")
        self.topic_messages: str = data.get("topic_messages", "rag.conversation.messages")
        self.topic_summaries: str = data.get("topic_summaries", "rag.conversation.summaries")
        self.consumer_group: str = data.get("consumer_group", "rag-memory-consumer")
        self.max_retries: int = int(data.get("max_retries", 3))
        self.retry_base_delay: float = float(data.get("retry_base_delay", 0.5))
        self.retry_max_delay: float = float(data.get("retry_max_delay", 5.0))


class SummaryConfig:
    def __init__(self, data: dict):
        self.enabled: bool = data.get("enabled", True)
        self.trigger_turns: int = int(data.get("trigger_turns", 5))
        self.model: str = data.get("model", "qwen-max")
        self.max_summary_length: int = int(data.get("max_summary_length", 300))


class CacheConfig:
    def __init__(self, data: dict):
        self.enabled: bool = data.get("enabled", True)
        self.similarity_threshold: float = float(data.get("similarity_threshold", 0.95))
        self.max_entries: int = int(data.get("max_entries", 100))
        self.ttl: int = int(data.get("ttl", 3600))


class RateLimitConfig:
    def __init__(self, data: dict):
        self.enabled: bool = data.get("enabled", True)
        # 全局限流
        self.global_rpm: int = int(data.get("global_rpm", 60))
        self.global_burst: int = int(data.get("global_burst", 15))
        # 用户级限流
        self.user_rpm: int = int(data.get("user_rpm", 20))
        self.user_burst: int = int(data.get("user_burst", 5))
        # LLM 成本
        self.cost_enabled: bool = self.enabled
        self.llm_max_tokens_per_hour: int = int(data.get("llm_max_tokens_per_hour", 100000))
        self.llm_max_requests_per_hour: int = int(data.get("llm_max_requests_per_hour", 50))
        # Redis 后端
        self.redis_enabled: bool = data.get("redis_enabled", True)


class MemoryConfig:
    def __init__(self, data: dict):
        self.redis = RedisConfig(data.get("redis", {}))
        self.mysql = MySQLConfig(data.get("mysql", {}))
        self.kafka = KafkaConfig(data.get("kafka", {}))
        self.summary = SummaryConfig(data.get("summary", {}))


class AppConfig:
    """应用全局配置"""

    def __init__(self):
        _load_env()
        raw = _parse_yml(ROOT_DIR / "config.yml")

        self.milvus = MilvusConfig(raw["milvus"])
        self.embedding = EmbeddingConfig(raw["embedding"])
        self.llm = LLMConfig(raw["llm"])
        self.retrieval = RetrievalConfig(raw["retrieval"])
        self.rerank = self.retrieval.rerank  # 快捷访问
        # Reranker 共用 LLM 的 API Key
        if not self.rerank.dashscope_api_key:
            self.rerank.dashscope_api_key = self.llm.dashscope_api_key
        self.chunking = ChunkingConfig(raw["chunking"])
        self.memory = MemoryConfig(raw.get("memory", {}))
        self.cache = CacheConfig(raw.get("cache", {}))
        self.rate_limit = RateLimitConfig(raw.get("rate_limit", {}))

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
