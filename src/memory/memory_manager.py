"""
三级记忆编排器 — 统一管理短期/中期/长期记忆

架构:
  ┌─────────────────────────────────────────────────────┐
  │  MemoryManager                                       │
  │                                                      │
  │  add_message(session_id, role, content)              │
  │    ├─ 1. Redis: 写入短期记忆（同步，毫秒级）          │
  │    ├─ 2. Kafka: 异步发布到 topic（不阻塞）            │
  │    └─ 3. 每 5 轮 → LLM 生成摘要 → Redis + Kafka      │
  │                                                      │
  │  get_context(session_id) → 对话历史 + 摘要            │
  │    ├─ 1. Redis: 读取最近 N 条消息（短期）             │
  │    └─ 2. Redis: 读取最新摘要（中期）                  │
  └─────────────────────────────────────────────────────┘

用法:
  manager = MemoryManager()
  manager.add_message(session_id, "user", "CT伪影怎么处理？")
  manager.add_message(session_id, "assistant", "根据工单...")
  context = manager.get_context(session_id)
  # → {"messages": [...], "summary": "...", "turn_count": 3}
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.config import get_config
from src.memory.redis_client import RedisMemoryStore
from src.memory.mysql_client import MySQLMemoryStore
from src.memory.kafka_client import KafkaProducerClient
from src.memory.summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)

# ── UTC+8 时区 ────────────────────────────────

_UTC8 = timezone(__import__("datetime").timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(_UTC8).isoformat(timespec="seconds")


class MemoryManager:
    """
    三级记忆编排器。

    职责:
      - 短期记忆: Redis 读写（会话消息列表）
      - 中期记忆: LLM 摘要（每 N 轮触发一次）
      - 长期记忆: Kafka → MySQL（异步持久化管道）

    所有外部依赖不可用时自动降级，不影响核心问答功能。
    """

    def __init__(self):
        cfg = get_config().memory

        # 短期记忆（必需，不可用时降级）
        self._redis = RedisMemoryStore(cfg.redis)

        # 长期记忆（直接写，Kafka 不可用时的 fallback）
        self._mysql = MySQLMemoryStore(cfg.mysql)
        if self._mysql.available:
            self._mysql.init_tables()

        # 消息队列（异步管道，不可用时直接写 MySQL）
        self._kafka = KafkaProducerClient(cfg.kafka)

        # 摘要生成器
        self._summarizer = ConversationSummarizer(cfg.summary)

        logger.info(
            f"MemoryManager 初始化: redis={self._redis.available} "
            f"mysql={self._mysql.available} kafka={self._kafka.available} "
            f"summary={self._summarizer.enabled}"
        )

    # ── 公开接口 ──────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> dict:
        """
        添加一条消息到记忆系统。

        参数:
          session_id: 会话 ID
          role:       "user" | "assistant"
          content:    消息文本

        返回: 本次消息的元信息 {"turn": int, "summary": str|None}
        """
        # 1. 写入 Redis 短期记忆（同步，必需）
        turn = self._redis.get_turn_count(session_id)
        if role == "user":
            turn += 1  # 新的一轮从 user 消息开始

        message = {
            "role": role,
            "content": content,
            "turn": turn,
            "timestamp": _now_iso(),
        }

        self._redis.add_message(session_id, message)

        # 2. Kafka 异步管道 → MySQL；Kafka 不可用时直接写 MySQL
        if self._kafka.available:
            self._kafka.send_message(session_id, message)
        elif self._mysql.available:
            self._mysql.insert_message(session_id, message)

        # 3. 检查是否需要生成中期摘要
        summary_text = None
        if role == "assistant" and self._summarizer.should_summarize(turn):
            summary_text = self._maybe_summarize(session_id, turn)

        return {"turn": turn, "summary": summary_text}

    def get_context(self, session_id: str) -> dict:
        """
        获取会话的完整记忆上下文。

        返回:
          {
            "messages":    list[dict],   # 最近 N 条消息
            "summary":     str | None,   # 最新中期摘要
            "turn_count":  int,          # 当前轮数
            "has_memory":  bool,         # 是否有历史记忆
          }
        """
        messages = self._redis.get_messages(session_id)
        summary = self._redis.get_summary(session_id)
        turn_count = sum(1 for m in messages if m.get("role") == "user")

        return {
            "messages": messages,
            "summary": summary,
            "turn_count": turn_count,
            "has_memory": len(messages) > 0,
        }

    def build_memory_prompt(self, session_id: str) -> str:
        """
        将记忆上下文格式化为 Prompt 片段，注入到 LLM 对话中。

        返回: 可直接拼接到 system_prompt 或 user_message 的文本。
              无历史时返回空字符串。
        """
        ctx = self.get_context(session_id)
        if not ctx["has_memory"]:
            return ""

        parts = []

        # 中期摘要（优先放前面，让模型先理解整体背景）
        if ctx["summary"]:
            parts.append(f"【历史对话摘要】\n{ctx['summary']}")

        # 短期记忆（最近几轮对话）
        if ctx["messages"]:
            lines = []
            for m in ctx["messages"]:
                role_label = "用户" if m.get("role") == "user" else "助手"
                content = m.get("content", "")[:300]  # 每条截断
                lines.append(f"{role_label}: {content}")
            parts.append("【最近对话】\n" + "\n".join(lines))

        return "\n\n".join(parts)

    def clear_session(self, session_id: str) -> bool:
        """清除某个会话的全部记忆"""
        return self._redis.clear_session(session_id)

    # ── 内部 ──────────────────────────────────

    def _maybe_summarize(self, session_id: str, turn: int) -> Optional[str]:
        """
        检查是否满足摘要触发条件，如果满足则生成摘要。

        条件: 当前轮数是 trigger_turns 的整数倍（默认每 5 轮）
        """
        # 获取最近 trigger_turns 轮的消息用于摘要
        recent = self._redis.get_messages(session_id)
        if not recent:
            return None

        summary = self._summarizer.generate(recent)
        if summary is None:
            return None

        # 计算摘要覆盖的轮次范围
        start_turn = max(1, turn - self._summarizer.trigger_turns + 1)
        turn_range = f"{start_turn}-{turn}"

        # 存入 Redis 短期记忆
        self._redis.update_summary(session_id, summary)

        # Kafka → MySQL；Kafka 不可用时直接写 MySQL
        if self._kafka.available:
            self._kafka.send_summary(session_id, summary, turn_range)
        elif self._mysql.available:
            self._mysql.insert_summary(session_id, summary, turn_range)

        logger.info(
            f"中期摘要已存储: session={session_id[:12]}... "
            f"turns={turn_range} len={len(summary)}"
        )
        return summary

    # ── 生命周期 ──────────────────────────────

    @property
    def available(self) -> bool:
        """至少短期记忆可用"""
        return self._redis.available

    def close(self):
        self._kafka.close()
