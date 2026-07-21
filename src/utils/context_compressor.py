"""
上下文压缩器 — 防止 Prompt 超过 LLM context window

问题场景:
  RAG 检索返回过多 chunk，或对话历史过长，导致拼接后的 Prompt
  超出 LLM 的 context window（Qwen-Max: 8K/32K tokens）。
  超限时 LLM API 会截断输入或直接报错。

压缩策略（按优先级，逐级递进）:
  ① Score-based truncation — 按相关性分数丢弃低分 chunk（零额外成本）
  ② Content truncation   — 每个 chunk 内容截断到 max_chars（零额外成本）
  ③ Deduplication        — 合并内容高度重叠的相邻 chunk（零额外成本）
  ④ LLM summarization    — 用 LLM 将上下文压缩为摘要（最后手段，+1次LLM调用）

用法:
  compressor = ContextCompressor()
  compressed = compressor.compress(
      chunks, system_prompt, question, memory_context="..."
  )
  # compressed 是保证 token 数 ≤ max_context_tokens 的 chunk 列表
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import ContextCompressionConfig, get_config

logger = logging.getLogger(__name__)

# ── token 估算 ─────────────────────────────────

# 中英文混合文本的 token 估算系数
# Qwen tokenizer: 中文 ~1.5 chars/token, 英文 ~4 chars/token
# 取保守值 1.2 chars/token（高估 token 数，避免截断不足）
_CHARS_PER_TOKEN = 1.2


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量。

    保守策略: 低估 chars_per_token → 高估 token 数 → 宁可多截断一点，不可超限。
    """
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_chunk_tokens(chunk: dict) -> int:
    """估算单个 chunk 在 Prompt 中的 token 数"""
    content = chunk.get("content", "")
    # 加上格式化开销: "─── 工单 N ───\n编号: ...  |  来源: ...\n" ≈ 40 chars
    overhead = 40
    return estimate_tokens(content) + estimate_tokens(str(overhead))


# ── 内容去重 ───────────────────────────────────

def _deduplicate_chunks(chunks: list[dict], threshold: float = 0.7) -> list[dict]:
    """
    合并内容高度重叠的相邻 chunk。

    使用简单的 Jaccard 相似度（基于字符 3-gram），
    相邻 chunk 重叠度过高时保留分数更高的那条。
    """
    if len(chunks) <= 1:
        return chunks

    def _jaccard(a: str, b: str) -> float:
        """基于 3-gram 的 Jaccard 相似度"""
        def ngrams(s, n=3):
            return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}
        set_a = ngrams(a)
        set_b = ngrams(b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = result[-1].get("content", "")
        curr = chunks[i].get("content", "")
        if _jaccard(prev, curr) > threshold:
            # 保留分数更高的
            if chunks[i].get("score", 0) > result[-1].get("score", 0):
                result[-1] = chunks[i]
            logger.debug(f"去重合并 chunk[{i}], Jaccard={_jaccard(prev, curr):.2f}")
        else:
            result.append(chunks[i])

    return result


# ═══════════════════════════════════════════════
# ContextCompressor
# ═══════════════════════════════════════════════

class ContextCompressor:
    """
    上下文压缩器。

    确保拼接后的 Prompt 不超过 LLM context window。
    压缩策略从低成本到高成本逐级递进。

    用法:
      compressor = ContextCompressor()
      chunks = compressor.compress(chunks, system_prompt, question)
      # 然后用压缩后的 chunks 构建 Prompt
    """

    def __init__(self, config: Optional[ContextCompressionConfig] = None):
        cfg = config or get_config().context_compression
        self._cfg = cfg
        self._llm = None

    # ── 公开接口 ──────────────────────────────

    def compress(
        self,
        chunks: list[dict],
        system_prompt: str = "",
        question: str = "",
        memory_context: str = "",
    ) -> list[dict]:
        """
        压缩上下文，保证总 token 数 ≤ max_context_tokens。

        参数:
          chunks:          检索返回的 chunk 列表（已按分数降序）
          system_prompt:   System Prompt 文本
          question:        用户问题
          memory_context:  对话记忆上下文文本（可选）

        返回:
          压缩后的 chunk 列表（可能比输入少）
        """
        if not chunks:
            return chunks

        max_tokens = self._cfg.max_context_tokens
        reserve_tokens = self._cfg.llm_response_reserve_tokens

        # 固定开销（不随 chunks 变化的部分）
        fixed_text = system_prompt + question + memory_context
        fixed_tokens = estimate_tokens(fixed_text)

        available = max_tokens - fixed_tokens - reserve_tokens
        if available <= 0:
            logger.warning(
                f"固定开销已超预算: fixed={fixed_tokens} "
                f"max={max_tokens} reserve={reserve_tokens}"
            )
            # 截断到至少保留 1 个 chunk
            return chunks[:1]

        # 估算当前总 token
        current_tokens = sum(estimate_chunk_tokens(c) for c in chunks)
        total_estimated = fixed_tokens + current_tokens

        if total_estimated <= available + fixed_tokens:
            logger.debug(
                f"上下文无需压缩: {total_estimated}/{max_tokens} tokens "
                f"(chunks={len(chunks)})"
            )
            return chunks

        logger.info(
            f"上下文需压缩: {total_estimated}/{max_tokens} tokens "
            f"(fixed={fixed_tokens}, chunks={len(chunks)}×{current_tokens // max(1, len(chunks))})"
        )

        # ── Level 1: Score truncation ──
        chunks = self._truncate_by_score(chunks, available, fixed_tokens)
        if self._fits(chunks, fixed_tokens, available + fixed_tokens):
            logger.info(f"压缩 L1 (score truncation): {len(chunks)} chunks 保留")
            return chunks

        # ── Level 2: Content truncation ──
        chunks = self._truncate_content(chunks, available, fixed_tokens)
        if self._fits(chunks, fixed_tokens, available + fixed_tokens):
            logger.info(f"压缩 L2 (content truncation): {len(chunks)} chunks, "
                        f"max={self._cfg.max_chunk_chars}chars/chunk")
            return chunks

        # ── Level 3: Deduplication ──
        chunks = _deduplicate_chunks(chunks, threshold=self._cfg.dedup_threshold)
        if self._fits(chunks, fixed_tokens, available + fixed_tokens):
            logger.info(f"压缩 L3 (dedup): {len(chunks)} chunks 保留")
            return chunks

        # ── Level 4: LLM summarization (last resort) ──
        if self._cfg.llm_summarize_enabled:
            compressed = self._summarize_with_llm(chunks, question)
            if compressed:
                logger.info(f"压缩 L4 (LLM summarization): 压缩上下文为 {len(compressed)} chunks")
                return compressed

        # ── Fallback: aggressive truncation ──
        logger.warning("所有压缩策略耗尽，激进截断")
        return self._aggressive_truncate(chunks, available, fixed_tokens)

    # ── 内部: 各级策略 ──────────────────────

    def _fits(self, chunks: list[dict], fixed_tokens: int, budget: int) -> bool:
        """检查 chunks + 固定开销是否在预算内"""
        chunk_tokens = sum(estimate_chunk_tokens(c) for c in chunks)
        return fixed_tokens + chunk_tokens <= budget

    def _truncate_by_score(
        self, chunks: list[dict], available: int, fixed_tokens: int
    ) -> list[dict]:
        """按分数从低到高丢弃 chunk，直到满足预算"""
        budget = available + fixed_tokens
        kept = []
        running = fixed_tokens
        for c in chunks:
            cost = estimate_chunk_tokens(c)
            if running + cost <= budget:
                kept.append(c)
                running += cost
            else:
                logger.debug(
                    f"丢弃 chunk score={c.get('score', 0):.4f} "
                    f"(tokens={cost}, budget剩余={budget - running})"
                )
        return kept

    def _truncate_content(
        self, chunks: list[dict], available: int, fixed_tokens: int
    ) -> list[dict]:
        """截断每个 chunk 的 content"""
        budget = available + fixed_tokens
        max_chars = self._cfg.max_chunk_chars

        result = []
        running = fixed_tokens
        for c in chunks:
            content = c.get("content", "")
            truncated = c.copy()
            if len(content) > max_chars:
                # 找最近的完整句号截断
                cut = content.rfind("。", 0, max_chars)
                if cut == -1 or cut < max_chars // 2:
                    cut = max_chars
                else:
                    cut += 1  # 保留句号
                truncated["content"] = content[:cut] + "…"
                logger.debug(f"截断 chunk: {len(content)}→{cut} chars")

            cost = estimate_chunk_tokens(truncated)
            if running + cost <= budget:
                result.append(truncated)
                running += cost
            else:
                break

        return result

    def _summarize_with_llm(
        self, chunks: list[dict], question: str,
    ) -> Optional[list[dict]]:
        """
        用 LLM 将多段上下文压缩为一段精简摘要。

        失败时返回 None，由上层 fallback 处理。
        """
        try:
            from src.llm.llm_client import LLMClient

            if self._llm is None:
                self._llm = LLMClient()

            # 拼接上下文
            context_text = "\n\n---\n\n".join(
                f"[工单 {c.get('ticket_id', '?')}] {c.get('content', '')[:400]}"
                for c in chunks[:10]  # 最多送 10 个 chunk 去压缩
            )

            prompt = (
                f"请将以下多条工单记录压缩为一条精简摘要，"
                f"保留与用户问题最相关的关键信息（故障原因、处理方案、关键结论）。\n\n"
                f"【用户问题】\n{question}\n\n"
                f"【工单记录】\n{context_text}\n\n"
                f"【压缩摘要】（不超过 500 字）"
            )

            summary = self._llm.generate(
                user_message=prompt,
                system_prompt="你是一个专业的文本压缩助手，只输出压缩后的摘要。",
            )

            if summary and len(summary.strip()) > 10:
                return [{
                    "content": summary.strip(),
                    "ticket_id": "compressed",
                    "source": "context_compressor",
                    "score": 1.0,
                }]

            logger.warning("LLM 压缩返回空或过短")
            return None

        except Exception as e:
            logger.warning(f"LLM 压缩失败: {e}")
            return None

    def _aggressive_truncate(
        self, chunks: list[dict], available: int, fixed_tokens: int
    ) -> list[dict]:
        """
        激进截断：每个 chunk 只保留前 150 字符 + 只取前 3 个。
        作为所有策略耗尽后的最终兜底。
        """
        budget = available + fixed_tokens
        result = []
        running = fixed_tokens
        for c in chunks[:3]:
            content = c.get("content", "")[:150]
            truncated = {**c, "content": content + "…"}
            cost = estimate_chunk_tokens(truncated)
            if running + cost <= budget:
                result.append(truncated)
                running += cost
            else:
                break
        return result
