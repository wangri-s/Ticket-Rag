"""
RAG 智能工单问答 — Streamlit 对话式前端（流式输出）

启动方式:
  streamlit run src/ui/app.py
  python -m streamlit run src/ui/app.py
"""

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src.llm.rag_chain import RAGChain
from src.config import get_config

# ── 页面配置 ──────────────────────────────────

st.set_page_config(
    page_title="智能工单 RAG 问答",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS 微调 ──────────────────────────────────

st.markdown("""
<style>
    .source-content {
        font-size: 0.85rem;
        color: #555;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .meta-bar {
        font-size: 0.8rem;
        color: #999;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# ── RAGChain ──────────────────────────────────

@st.cache_resource
def init_chain() -> RAGChain:
    return RAGChain()


# ── 侧边栏 ────────────────────────────────────

def render_sidebar() -> dict:
    """返回当前参数，用于下一次提问"""
    st.sidebar.markdown("## ⚙️ 检索设置")

    params = {}

    params["mode"] = st.sidebar.selectbox(
        "检索模式",
        options=["hybrid", "semantic", "keyword"],
        format_func=lambda x: {
            "hybrid": "🔀 混合检索",
            "semantic": "🧠 语义检索",
            "keyword": "🔑 关键字检索",
        }[x],
        index=0,
    )

    params["top_k"] = st.sidebar.slider(
        "检索条数", min_value=1, max_value=10, value=5, step=1,
    )

    params["query_expansion"] = st.sidebar.checkbox(
        "启用查询扩展 (Query Expansion)",
        value=False,
        help="LLM 将口语改写为专业关键词再检索。例如'片子不清楚'→'CT 图像伪影 分辨率下降'。增加约 0.5s 延迟。",
    )

    params["rerank"] = st.sidebar.checkbox(
        "启用重排序 (Rerank)",
        value=False,
        help="初检多召回 → qwen3-rerank 交叉编码精排 → 取 Top-K。更精准，但增加约 0.5-1s 延迟。",
    )

    params["output_format"] = st.sidebar.selectbox(
        "输出格式",
        options=["text", "json"],
        format_func=lambda x: {
            "text": "📝 自然语言（Few-shot + CoT）",
            "json": "🔧 JSON 结构化（API 场景）",
        }[x],
        index=0,
        help="text: 含思维链推理和格式示例的自然语言回答。json: Schema 约束的 JSON 对象，适合下游系统解析。",
    )

    _tid = st.sidebar.text_input(
        "按工单号过滤（可选）",
        value="",
        placeholder="如 GD-2026-03001",
    )
    params["ticket_id"] = _tid.strip() if _tid.strip() else None

    from src.retrieval.metadata_filter import KNOWN_DEVICE_TYPES
    _dtype = st.sidebar.selectbox(
        "按设备类型过滤（可选）",
        options=["(全部)"] + KNOWN_DEVICE_TYPES,
        index=0,
    )
    params["device_type"] = _dtype if _dtype != "(全部)" else None

    if st.sidebar.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        # 生成新 session_id = 开启新会话记忆
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.rerun()

    # 会话记忆状态
    sid = st.session_state.get("session_id", "")
    if sid:
        st.sidebar.caption(f"💾 会话记忆: `{sid[:8]}...` (每5轮自动摘要)")

    st.sidebar.divider()

    # 快捷问题
    st.sidebar.markdown("### 💡 快捷问题")
    quick_questions = [
        "CT扫描图像伪影是什么原因？",
        "透析液电导率偏高报警怎么处理？",
        "MRI设备啸叫声怎么排查？",
        "超声探头成像模糊是什么故障？",
        "监护仪数据断连怎么处理？",
        "麻醉机呼吸回路漏气怎么办？",
        "DR系统曝光后预览延迟严重",
        "挂号窗口响应超时怎么办？",
    ]
    for i, q in enumerate(quick_questions):
        if st.sidebar.button(q, key=f"qq_{i}", use_container_width=True):
            st.session_state.quick_input = q
            st.rerun()

    st.sidebar.divider()

    # 系统信息
    st.sidebar.markdown("### 📊 系统信息")
    cfg = get_config()
    with st.sidebar.expander("配置详情", expanded=False):
        st.markdown(f"""
| 项目 | 值 |
|------|-----|
| Milvus | `{cfg.milvus.uri}` |
| Collection | `{cfg.milvus.collection_name}` |
| Embedding | `{cfg.embedding.model}` |
| LLM | `{cfg.llm.model}` |
| Chunk | `{cfg.chunking.chunk_size}` 字符 |
        """)

    return params


# ── 渲染来源引用 ──────────────────────────────

def render_sources(sources: list, expanded: bool = False):
    """在聊天消息内渲染引用来源折叠面板"""
    if not sources:
        return
    with st.expander(f"📚 引用来源（{len(sources)} 条）", expanded=expanded):
        for j, s in enumerate(sources):
            score = s.get("score", 0.0)
            bar_len = max(1, int(score * 20))
            bar = "█" * bar_len + "░" * (20 - bar_len)

            st.markdown(
                f"**[{j+1}] {s.get('ticket_id', '?')}** "
                f"`{bar}` {score:.4f}"
            )
            st.caption(f"来源: {s.get('source', '?')}")
            st.markdown(
                f'<div class="source-content">{s.get("content", "")[:600]}</div>',
                unsafe_allow_html=True,
            )
            if len(s.get("content", "")) > 600:
                st.caption("... (内容过长，已截断)")
            if j < len(sources) - 1:
                st.divider()


# ── 主界面 ────────────────────────────────────

def main():
    # 初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "quick_input" not in st.session_state:
        st.session_state.quick_input = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]

    chain = init_chain()
    params = render_sidebar()

    # ── 顶部标题 ──
    st.title("🏥 智能工单 RAG 问答")
    st.caption("基于 Milvus + Qwen-Max 的医疗设备运维工单检索增强生成 · 流式输出")

    # ── 渲染历史消息 ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                # 回答正文（历史消息直接渲染，不再流式）
                st.markdown(msg["answer"])

                # 元信息
                badge = "✅ 命中" if msg.get("has_answer") else "⚠️ 兜底"
                st.caption(
                    f"{badge} &nbsp;|&nbsp; "
                    f"模式 `{msg.get('mode', '?')}` &nbsp;|&nbsp; "
                    f"来源 {len(msg.get('sources', []))} 条 &nbsp;|&nbsp; "
                    f"耗时 {msg.get('latency', 0):.0f}ms"
                )

                # 来源（历史消息默认折叠）
                render_sources(msg.get("sources", []), expanded=False)
            else:
                st.markdown(msg["content"])

    # ── 处理快捷问题注入 ──
    pending = st.session_state.quick_input
    if pending:
        st.session_state.quick_input = None

    # ── 输入框 ──
    prompt = st.chat_input("输入你的问题，按 Enter 发送...")

    question = prompt or pending
    if not question:
        return

    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 执行 RAG + 流式输出
    with st.chat_message("assistant"):
        # 先做检索（阻塞）
        status_placeholder = st.empty()
        status_placeholder.caption("🔍 正在检索...")

        t0 = time.perf_counter()
        try:
            result = chain.ask_stream(
                question=question,
                mode=params["mode"],
                top_k=params["top_k"],
                ticket_id_filter=params["ticket_id"],
                device_type_filter=params.get("device_type"),
                rerank=params.get("rerank", False),
                query_expansion=params.get("query_expansion", False),
                output_format=params.get("output_format", "text"),
                session_id=st.session_state.get("session_id"),
            )
        except Exception as e:
            status_placeholder.empty()
            st.error(f"❌ 检索失败：{e}")
            st.session_state.messages.append({
                "role": "assistant",
                "answer": f"❌ 检索失败：{e}",
                "sources": [],
                "has_answer": False,
                "mode": params["mode"],
                "latency": 0,
            })
            return

        sources = result["sources"]
        has_answer = result["has_answer"]

        # 检索完成，开始流式生成
        status_placeholder.caption(
            f"✍️ 正在生成... &nbsp; ({len(sources)} 条来源)"
        )

        # st.write_stream 接收一个生成器，逐 chunk 渲染
        answer_placeholder = st.empty()
        full_answer = ""

        try:
            # 流式写入（打字机效果）
            for chunk in result["stream"]:
                full_answer += chunk
                answer_placeholder.markdown(full_answer + "▌")

            # 去掉光标
            answer_placeholder.markdown(full_answer)

            # JSON 格式：额外显示格式化 JSON 块
            if result.get("output_format") == "json":
                import json as _json
                try:
                    parsed = _json.loads(full_answer)
                    with st.expander("📋 JSON 结构化视图", expanded=True):
                        st.json(parsed)
                except _json.JSONDecodeError:
                    st.caption("⚠️ LLM 返回的不是合法 JSON，已显示原始文本")
        except Exception as e:
            answer_placeholder.error(f"❌ 生成失败：{e}")
            full_answer = f"❌ 生成失败：{e}"

        latency = (time.perf_counter() - t0) * 1000
        status_placeholder.empty()

        # 元信息
        badge = "✅ 命中" if has_answer else "⚠️ 兜底"
        st.caption(
            f"{badge} &nbsp;|&nbsp; "
            f"模式 `{result['mode']}` &nbsp;|&nbsp; "
            f"来源 {len(sources)} 条 &nbsp;|&nbsp; "
            f"耗时 {latency:.0f}ms"
        )

        # 来源（当前回答默认折叠）
        render_sources(sources, expanded=False)

    # 存入历史
    st.session_state.messages.append({
        "role": "assistant",
        "answer": full_answer,
        "sources": sources,
        "has_answer": has_answer,
        "mode": result["mode"],
        "latency": latency,
    })


if __name__ == "__main__":
    main()
