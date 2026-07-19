"""
对话摘要生成器 — LLM 驱动的中期记忆

每 N 轮对话触发一次（默认 5 轮），将最近的对话压缩为结构化摘要。
摘要作为"中期记忆"存入 Redis + Kafka → MySQL，桥接短/长期记忆。

用法:
  summarizer = ConversationSummarizer()
  summary = summarizer.generate(messages)  # → "用户咨询了CT伪影问题..."
"""

import logging
from typing import Optional

from src.config import SummaryConfig, get_config

logger = logging.getLogger(__name__)

# ── 摘要生成 Prompt ──────────────────────────

SUMMARY_SYSTEM_PROMPT = """你是一个对话摘要助手。将以下医疗设备运维对话压缩为简洁的结构化摘要。
只输出摘要本身，不要添加解释或前缀。"""

SUMMARY_USER_TEMPLATE = """请将以下 {turn_count} 轮对话压缩为摘要（不超过 {max_length} 字）。

要求:
1. 提炼用户的核心问题和设备类型
2. 记录助手给出的关键诊断结论和处理方案
3. 标注未解决的问题（如有）
4. 保留工单编号引用

对话记录:
{conversation}

摘要:"""


class ConversationSummarizer:
    """
    LLM 驱动的对话摘要生成器。

    每 trigger_turns 轮生成一次摘要，将最近对话压缩为结构化摘要文本。
    """

    def __init__(self, config: Optional[SummaryConfig] = None):
        cfg = config or get_config().memory.summary
        self.enabled = cfg.enabled
        self.trigger_turns = cfg.trigger_turns
        self.model = cfg.model
        self.max_summary_length = cfg.max_summary_length

        self._llm_api_key: Optional[str] = None

    def _get_api_key(self) -> str:
        """懒加载 API Key（从全局 LLM 配置获取）"""
        if self._llm_api_key:
            return self._llm_api_key
        from src.config import get_config
        self._llm_api_key = get_config().llm.dashscope_api_key
        return self._llm_api_key

    def generate(self, messages: list[dict]) -> Optional[str]:
        """
        对对话消息生成中期摘要。

        参数:
          messages: [{"role": "user|assistant", "content": "..."}, ...]

        返回: 摘要字符串，失败时返回 None
        """
        if not self.enabled or not messages:
            return None

        # 格式化对话记录
        turn_count = sum(1 for m in messages if m.get("role") == "user")
        lines = []
        for m in messages:
            role_label = "👤 用户" if m.get("role") == "user" else "🤖 助手"
            content = m.get("content", "")[:500]  # 每条截断 500 字
            lines.append(f"{role_label}: {content}")

        conversation_text = "\n".join(lines)

        user_prompt = SUMMARY_USER_TEMPLATE.format(
            turn_count=turn_count,
            max_length=self.max_summary_length,
            conversation=conversation_text,
        )

        try:
            from dashscope import Generation

            resp = Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=self._get_api_key(),
                temperature=0.1,
                max_tokens=min(self.max_summary_length * 2, 600),
                seed=42,
            )

            if resp.status_code == 200 and resp.output and resp.output.text:
                summary = resp.output.text.strip()
                logger.info(
                    f"中期摘要已生成: {len(summary)} 字符 "
                    f"({turn_count} 轮对话 → 摘要)"
                )
                return summary
            else:
                logger.warning(f"摘要生成 API 返回异常: code={resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"摘要生成失败: {e}")
            return None

    def should_summarize(self, turn_count: int) -> bool:
        """判断是否应该触发摘要（每 N 轮一次）"""
        if not self.enabled or turn_count < 1:
            return False
        return turn_count % self.trigger_turns == 0
