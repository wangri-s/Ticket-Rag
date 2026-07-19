"""
智能工单 RAG 系统 — 提示词管理

提供:
  - build_rag_prompt(context, question) → 拼接完整的 RAG 用户消息
  - get_system_prompt()                  → 返回预设医疗专家 system prompt
  - format_chunks(chunks)                → 将检索结果格式化为 Prompt 可读文本
"""

from typing import Literal, Optional

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


def build_json_prompt(
    question: str,
    chunks: list[dict],
    ticket_id_filter: str = None,
) -> str:
    """
    构建 JSON 结构化输出的 Prompt：检索结果 + 用户问题 + JSON Schema 约束。

    与 build_rag_prompt 相同，但使用 json_prompt_template 模板，
    强制模型返回可解析的 JSON 对象，适合 API 集成场景。

    参数:
      question:         用户原始问题
      chunks:           检索到的 chunk 列表
      ticket_id_filter: 如果用户指定了工单号，可在此注明

    返回:
        可直接传给 LLM 的 user_message 字符串，或 None（无检索结果时）
    """
    cfg = get_config().llm

    if not chunks:
        return None

    context = format_chunks(chunks)

    filter_note = ""
    if ticket_id_filter:
        filter_note = f"\n（用户指定仅查询工单: {ticket_id_filter}）"

    prompt = cfg.json_prompt_template.format(
        context=context,
        question=question + filter_note,
    )

    return prompt


OutputFormat = Literal["text", "json"]


# ── 获取 System Prompt ───────────────────────

def get_system_prompt() -> str:
    """返回 config.yml 中预设的医疗运维专家 system prompt"""
    return get_config().llm.system_prompt


def get_fallback_answer() -> str:
    """返回无检索结果时的兜底回答"""
    return get_config().llm.fallback_answer


def get_json_fallback() -> str:
    """返回无检索结果时的 JSON 兜底回答"""
    import json
    return json.dumps({
        "question": "",
        "has_reference": False,
        "analysis": {
            "symptom_breakdown": "",
            "possible_causes": [],
            "reasoning_chain": "知识库中未找到相关历史工单",
        },
        "references": [],
        "recommendations": {
            "steps": ["尝试用不同关键词描述故障现象", "联系设备厂家技术支持"],
            "precautions": [],
            "urgency": "medium",
        },
    }, ensure_ascii=False)


# ── 快捷函数 ─────────────────────────────────

def build_full_prompt(
    question: str,
    chunks: list[dict],
    ticket_id_filter: str = None,
    output_format: OutputFormat = "text",
) -> tuple[Optional[str], Optional[str]]:
    """
    一次性构建 system_prompt + user_message。

    参数:
      question:         用户问题
      chunks:           检索到的 chunk 列表
      ticket_id_filter: 按工单号过滤（可选）
      output_format:    "text" → 自然语言（含 Few-shot + CoT）
                        "json" → JSON Schema 约束输出

    返回: (system_prompt, user_message)
      - 有检索结果: 返回 (medical_expert_prompt, formatted_message)
      - 无检索结果: 返回 (None, None)，调用方使用 fallback
    """
    if output_format == "json":
        user_message = build_json_prompt(question, chunks, ticket_id_filter)
    else:
        user_message = build_rag_prompt(question, chunks, ticket_id_filter)

    if user_message is None:
        return None, None

    return get_system_prompt(), user_message
