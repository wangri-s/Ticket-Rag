"""
智能工单 RAG 系统 — 提示词管理

提供:
  - build_rag_prompt(context, question) → 拼接完整的 RAG 用户消息
  - get_system_prompt()                  → 返回预设医疗专家 system prompt
  - format_chunks(chunks)                → 将检索结果格式化为 Prompt 可读文本
"""

from src.config import get_config


# ── 格式化检索结果 ───────────────────────────

def format_chunks(chunks: list[dict]) -> str:
    """
    将检索返回的 chunk 列表格式化为 Prompt 可读文本。

    参数:
      chunks: [{"content": "...", "ticket_id": "...", "source": "...", "chunk_index": 0}, ...]

    返回:
        格式化的文本，每条以工单编号和来源标注
    """
    if not chunks:
        return "（无参考工单）"

    lines = []
    for i, c in enumerate(chunks, 1):
        tid = c.get("ticket_id", "?")
        src = c.get("source", "?")
        content = c.get("content", "").strip()

        # 去掉工单内容中的分隔线，节省 token
        content = content.replace("═", "").replace("─", "").strip()

        lines.append(f"─── 工单 {i} ───")
        lines.append(f"编号: {tid}  |  来源: {src}")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


# ── 构建 RAG Prompt ──────────────────────────

def build_rag_prompt(
    question: str,
    chunks: list[dict],
    ticket_id_filter: str = None,
) -> str:
    """
    构建 RAG 的完整用户消息：检索结果 + 用户问题。

    参数:
      question:         用户原始问题
      chunks:           检索到的 chunk 列表
      ticket_id_filter: 如果用户指定了工单号，可在此注明

    返回:
        可直接传给 LLMClient.generate() 的 user_message 字符串
    """
    cfg = get_config().llm

    # 无检索结果 → 返回 None，由调用方用 fallback_answer
    if not chunks:
        return None

    # 格式化上下文
    context = format_chunks(chunks)

    # 拼接过滤说明
    filter_note = ""
    if ticket_id_filter:
        filter_note = f"\n（用户指定仅查询工单: {ticket_id_filter}）"

    # 填入模板
    prompt = cfg.rag_prompt_template.format(
        context=context,
        question=question + filter_note,
    )

    return prompt


# ── 获取 System Prompt ───────────────────────

def get_system_prompt() -> str:
    """返回 config.yml 中预设的医疗运维专家 system prompt"""
    return get_config().llm.system_prompt


def get_fallback_answer() -> str:
    """返回无检索结果时的兜底回答"""
    return get_config().llm.fallback_answer


# ── 快捷函数 ─────────────────────────────────

def build_full_prompt(
    question: str,
    chunks: list[dict],
    ticket_id_filter: str = None,
) -> tuple[str | None, str]:
    """
    一次性构建 system_prompt + user_message。

    返回: (system_prompt, user_message)
      - 有检索结果: 返回 (medical_expert_prompt, rag_formatted_message)
      - 无检索结果: 返回 (None, None)，调用方使用 get_fallback_answer()
    """
    user_message = build_rag_prompt(question, chunks, ticket_id_filter)
    if user_message is None:
        return None, None

    return get_system_prompt(), user_message
