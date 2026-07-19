"""
Redis 短期记忆客户端

存储结构:
  session:{sid}:messages → List<JSON>  （最近 N 条消息，自动 trim）
  session:{sid}:summary  → String      （最新中期摘要）

用法:
  redis_mem = RedisMemoryStore()
  redis_mem.add_message(session_id, {"role": "user", "content": "..."})
  messages = redis_mem.get_messages(session_id)  # → list[dict]
  redis_mem.update_summary(session_id, "摘要文本")
"""

import json
import logging
from typing import Optional

import redis

from src.config import RedisConfig, get_config

logger = logging.getLogger(__name__)


class RedisMemoryStore:
    """
    Redis 短期记忆存储。

    职责:
      - 存储当前会话的最近 N 条消息（List 结构，自动 trim 到 max_messages）
      - 存储最新中期摘要
      - TTL 自动过期（默认 24h）
    """

    def __init__(self, config: Optional[RedisConfig] = None):
        cfg = config or get_config().memory.redis
        self.key_prefix = cfg.key_prefix
        self.max_messages = cfg.max_messages
        self.ttl = cfg.ttl

        self._client = redis.Redis(
            host=cfg.host,
            port=cfg.port,
            db=cfg.db,
            password=cfg.password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        # 连接检查
        try:
            self._client.ping()
            logger.info(f"Redis 已连接: {cfg.host}:{cfg.port}/{cfg.db}")
        except Exception as e:
            logger.warning(f"Redis 连接失败（将降级为无记忆模式）: {e}")
            self._client = None

    # ── Key 生成 ──────────────────────────────

    def _msg_key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}:messages"

    def _summary_key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}:summary"

    # ── 消息操作 ──────────────────────────────

    def add_message(self, session_id: str, message: dict) -> bool:
        """
        追加一条消息到会话的消息列表。

        参数:
          session_id: 会话 ID
          message:    {"role": "user|assistant", "content": "...",
                       "turn": int, "timestamp": "ISO8601"}

        返回: 是否写入成功（Redis 不可用时返回 False，调用方降级处理）
        """
        if self._client is None:
            return False

        try:
            payload = json.dumps(message, ensure_ascii=False)
            key = self._msg_key(session_id)

            # 追加到列表右侧
            self._client.rpush(key, payload)

            # Trim 到 max_messages（保留最新 N 条）
            self._client.ltrim(key, -self.max_messages, -1)

            # 续期 TTL
            self._client.expire(key, self.ttl)

            return True
        except Exception as e:
            logger.error(f"Redis 写入消息失败: {e}")
            return False

    def get_messages(self, session_id: str) -> list[dict]:
        """
        获取会话的最近消息列表。

        返回: list[dict]，Redis 不可用或无数据时返回空列表
        """
        if self._client is None:
            return []

        try:
            key = self._msg_key(session_id)
            raw_list = self._client.lrange(key, 0, -1)
            messages = []
            for raw in raw_list:
                try:
                    messages.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
            return messages
        except Exception as e:
            logger.error(f"Redis 读取消息失败: {e}")
            return []

    def get_turn_count(self, session_id: str) -> int:
        """获取当前会话的对话轮数（一组 user+assistant 算 1 轮）"""
        messages = self.get_messages(session_id)
        user_count = sum(1 for m in messages if m.get("role") == "user")
        return user_count

    # ── 摘要操作 ──────────────────────────────

    def update_summary(self, session_id: str, summary: str) -> bool:
        """更新会话的中期摘要"""
        if self._client is None:
            return False

        try:
            key = self._summary_key(session_id)
            self._client.set(key, summary)
            self._client.expire(key, self.ttl)
            return True
        except Exception as e:
            logger.error(f"Redis 写入摘要失败: {e}")
            return False

    def get_summary(self, session_id: str) -> Optional[str]:
        """获取会话的最新中期摘要"""
        if self._client is None:
            return None

        try:
            key = self._summary_key(session_id)
            val = self._client.get(key)
            return val if val else None
        except Exception as e:
            logger.error(f"Redis 读取摘要失败: {e}")
            return None

    # ── 生命周期 ──────────────────────────────

    def clear_session(self, session_id: str) -> bool:
        """清除某个会话的全部记忆"""
        if self._client is None:
            return False
        try:
            self._client.delete(self._msg_key(session_id), self._summary_key(session_id))
            return True
        except Exception as e:
            logger.error(f"Redis 清除会话失败: {e}")
            return False

    @property
    def available(self) -> bool:
        return self._client is not None
