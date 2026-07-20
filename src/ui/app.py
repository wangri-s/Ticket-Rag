"""
RAG 智能工单问答 — Streamlit 对话式前端（流式输出）

启动方式:
  streamlit run src/ui/app.py
  python -m streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import streamlit.components.v1 as components

from src.llm.rag_chain import RAGChain
from src.config import get_config

# ── 页面配置 ──────────────────────────────────────

st.set_page_config(
    page_title="智能工单 RAG 问答",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "医疗设备运维工单 RAG 问答系统 · Milvus + Qwen-Max",
    },
)

# ═══════════════════════════════════════════════════════════
# CSS 样式
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── 全局 ── */
:root {
    --primary: #1a73e8;
    --success: #0d904f;
    --warning: #e37400;
    --danger: #d93025;
    --bg-card: #f8f9fa;
    --border: #e0e0e0;
}

/* ── 欢迎页 ── */
.welcome-container {
    text-align: center;
    padding: 3rem 1rem;
    max-width: 700px;
    margin: 0 auto;
}
.welcome-container h1 {
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
}
.welcome-container .subtitle {
    color: #5f6368;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.welcome-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 0.75rem;
    margin-top: 1.5rem;
}
.welcome-card {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: left;
    cursor: pointer;
    transition: all 0.15s;
}
.welcome-card:hover {
    border-color: var(--primary);
    box-shadow: 0 2px 8px rgba(26,115,232,0.12);
    transform: translateY(-1px);
}
.welcome-card .card-icon { font-size: 1.3rem; margin-right: 0.4rem; }
.welcome-card .card-text { font-size: 0.9rem; color: #202124; }

/* ── 聊天消息 ── */
.chat-row {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.5rem 0;
}
.chat-avatar {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}
.chat-avatar.user { background: #e8f0fe; }
.chat-avatar.assistant { background: #e6f4ea; }
.chat-bubble {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    max-width: 100%;
    overflow-x: auto;
}

/* ── 状态指示器 ── */
.status-box {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}
.status-box.searching { background: #e8f0fe; color: #1967d2; }
.status-box.generating { background: #fef7e0; color: #b06000; }
.status-box.done { background: #e6f4ea; color: #137333; }
.status-pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse 1.2s infinite;
}
.status-box.searching .status-pulse { background: #1967d2; }
.status-box.generating .status-pulse { background: #f9ab00; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── 元信息标签栏 ── */
.meta-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.6rem;
    font-size: 0.78rem;
}
.meta-tag {
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-weight: 500;
    white-space: nowrap;
}
.meta-tag.hit    { background: #e6f4ea; color: #137333; }
.meta-tag.miss   { background: #fce8e6; color: #c5221f; }
.meta-tag.cache  { background: #e8f0fe; color: #1967d2; }
.meta-tag.mode   { background: #f1f3f4; color: #5f6368; }
.meta-tag.latency-fast  { background: #e6f4ea; color: #137333; }
.meta-tag.latency-mid   { background: #fef7e0; color: #b06000; }
.meta-tag.latency-slow  { background: #fce8e6; color: #c5221f; }

/* ── 来源卡片 ── */
.source-card {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    background: var(--bg-card);
}
.source-card .source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.source-card .source-id {
    font-weight: 600;
    font-size: 0.9rem;
    color: #202124;
}
.source-card .source-score {
    font-family: monospace;
    font-size: 0.8rem;
    color: #5f6368;
}
.source-score-bar {
    height: 4px;
    border-radius: 2px;
    background: #e0e0e0;
    margin: 0.4rem 0 0.5rem 0;
    overflow: hidden;
}
.source-score-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s;
}
.source-content-text {
    font-size: 0.85rem;
    color: #3c4043;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 150px;
    overflow-y: auto;
}
.source-meta {
    font-size: 0.75rem;
    color: #80868b;
    margin-top: 0.3rem;
}

/* ── 侧边栏优化 ── */
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.95rem;
    margin-top: 1.2rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #e0e0e0;
}
.sidebar-section {
    margin-bottom: 0.5rem;
}

/* ── 打字光标 ── */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}
.typing-cursor {
    animation: blink 0.8s infinite;
    color: var(--primary);
    font-weight: bold;
}

/* ── 响应式 ── */
@media (max-width: 768px) {
    .welcome-grid { grid-template-columns: 1fr; }
    .meta-tags { gap: 0.25rem; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 懒加载
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def init_chain() -> RAGChain:
    return RAGChain()


# ═══════════════════════════════════════════════════════════
# 辅助组件
# ═══════════════════════════════════════════════════════════

def _latency_class(ms: float) -> str:
    """根据延迟返回 CSS class"""
    if ms < 1000:
        return "latency-fast"
    elif ms < 3000:
        return "latency-mid"
    else:
        return "latency-slow"


def _score_color(score: float) -> str:
    """分数 → 颜色"""
    if score >= 0.85:
        return "#0d904f"
    elif score >= 0.7:
        return "#e37400"
    else:
        return "#5f6368"


def render_source_card(source: dict, index: int):
    """单张来源卡片"""
    score = source.get("score", 0.0)
    ticket_id = source.get("ticket_id", "?")
    content = source.get("content", "")
    src_file = source.get("source", "?")
    color = _score_color(score)

    st.markdown(f"""
    <div class="source-card">
        <div class="source-header">
            <span class="source-id">📄 [{index}] {ticket_id}</span>
            <span class="source-score" style="color:{color}">★ {score:.4f}</span>
        </div>
        <div class="source-score-bar">
            <div class="source-score-fill" style="width:{score * 100}%;background:{color}"></div>
        </div>
        <div class="source-content-text">{content[:500]}</div>
        <div class="source-meta">📁 {src_file}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sources(sources: list, expanded: bool = False):
    """渲染引用来源列表"""
    if not sources:
        return
    label = f"📚 引用来源（{len(sources)} 条）"
    if len(sources) <= 3:
        expanded = True  # 来源少时默认展开
    with st.expander(label, expanded=expanded):
        for j, s in enumerate(sources, 1):
            render_source_card(s, j)


def render_meta_tags(
    has_answer: bool,
    from_cache: bool,
    mode: str,
    latency_ms: float,
    source_count: int,
):
    """渲染元信息标签行"""
    tags = []

    if from_cache:
        tags.append(('<span class="meta-tag cache">⚡ 缓存命中</span>'))
    else:
        if has_answer:
            tags.append((f'<span class="meta-tag hit">✅ 命中 · {source_count}条来源</span>'))
        else:
            tags.append(('<span class="meta-tag miss">⚠️ 未找到相关工单</span>'))

    tags.append((f'<span class="meta-tag mode">🔀 {mode}</span>'))
    tags.append((
        f'<span class="meta-tag {_latency_class(latency_ms)}">⏱ {latency_ms:.0f}ms</span>'
    ))

    html = '<div class="meta-tags">' + ''.join(tags) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 欢迎页
# ═══════════════════════════════════════════════════════════

WELCOME_QUESTIONS = [
    ("🩻", "CT扫描图像伪影是什么原因？"),
    ("🩸", "透析液电导率偏高报警怎么处理？"),
    ("🔊", "MRI设备啸叫声怎么排查？"),
    ("📡", "超声探头成像模糊是什么故障？"),
    ("💓", "监护仪数据断连怎么处理？"),
    ("🫁", "麻醉机呼吸回路漏气怎么办？"),
    ("🩻", "DR系统曝光后预览延迟严重"),
    ("💻", "挂号窗口响应超时怎么办？"),
]


def render_welcome():
    """空会话时的欢迎引导页"""
    st.markdown("""
    <div class="welcome-container">
        <h1>🏥 智能工单 RAG 问答</h1>
        <p class="subtitle">
            基于 <strong>Milvus 向量检索</strong> + <strong>Qwen-Max 大模型</strong><br>
            医疗设备运维工单 · 检索增强生成 · 流式输出
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 能力卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with st.container(border=True):
            st.markdown("🔍 **混合检索**")
            st.caption("语义 + 关键词加权融合")
    with col2:
        with st.container(border=True):
            st.markdown("🧠 **CoT 推理**")
            st.caption("思维链引导鉴别诊断")
    with col3:
        with st.container(border=True):
            st.markdown("💾 **会话记忆**")
            st.caption("三级记忆自动摘要")
    with col4:
        with st.container(border=True):
            st.markdown("⚡ **语义缓存**")
            st.caption("相似问题秒级命中")

    st.markdown("<br>", unsafe_allow_html=True)

    # 快捷问题网格
    st.markdown("#### 💡 试试这些问题")

    # 两行四列
    rows = [WELCOME_QUESTIONS[i:i + 4] for i in range(0, len(WELCOME_QUESTIONS), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for i, (icon, q) in enumerate(row):
            with cols[i]:
                if st.button(
                    f"{icon} {q}",
                    key=f"welcome_{q[:15]}",
                    use_container_width=True,
                ):
                    st.session_state.quick_input = q
                    st.rerun()

    st.divider()
    st.caption("💬 在下方输入框键入问题，或点击上方快捷问题开始")


# ═══════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """返回当前检索参数"""
    with st.sidebar:
        st.markdown("## ⚙️ 检索设置")

        params = {}

        # ── 检索模式 ──
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        params["mode"] = st.selectbox(
            "检索模式",
            options=["hybrid", "semantic", "keyword"],
            format_func=lambda x: {
                "hybrid":  "🔀 混合检索（推荐）",
                "semantic": "🧠 语义检索",
                "keyword":  "🔑 关键字检索",
            }[x],
            index=0,
        )

        params["top_k"] = st.slider(
            "返回条数", 1, 10, 5,
            help="检索结果数量，越多召回越全但噪声也越多",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 高级选项 ──
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        with st.expander("🔧 高级选项", expanded=False):
            params["query_expansion"] = st.checkbox(
                "查询扩展",
                value=False,
                help="LLM 将口语改写为专业关键词再检索（+0.5s）",
            )
            params["rerank"] = st.checkbox(
                "重排序 (Rerank)",
                value=False,
                help="qwen3-rerank 交叉编码精排（+0.5~1s）",
            )
            params["output_format"] = st.selectbox(
                "输出格式",
                options=["text", "json"],
                format_func=lambda x: "📝 自然语言" if x == "text" else "🔧 JSON 结构",
                index=0,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 过滤 ──
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        with st.expander("🎯 过滤条件", expanded=False):
            tid = st.text_input(
                "工单编号",
                value="",
                placeholder="如 GD-2026-03001",
                label_visibility="collapsed",
            )
            params["ticket_id"] = tid.strip() if tid.strip() else None

            from src.retrieval.metadata_filter import KNOWN_DEVICE_TYPES
            dtype = st.selectbox(
                "设备类型",
                options=["(全部)"] + KNOWN_DEVICE_TYPES,
                index=0,
                label_visibility="collapsed",
            )
            params["device_type"] = dtype if dtype != "(全部)" else None
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # ── 会话管理 ──
        st.markdown("### 💬 会话")
        sid = st.session_state.get("session_id", "")
        if sid:
            msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.caption(
                f"🔑 `{sid[:8]}...` · 已对话 **{msg_count}** 轮\n\n"
                "每 5 轮自动生成摘要"
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = uuid.uuid4().hex[:12]
                st.rerun()
        with c2:
            if st.button("🔄 新会话", use_container_width=True):
                st.session_state.session_id = uuid.uuid4().hex[:12]
                st.toast("已创建新会话", icon="🆕")

        st.divider()

        # ── 系统状态 ──
        st.markdown("### 📊 系统状态")
        cfg = get_config()
        try:
            from src.utils.rate_limiter import get_rate_limiter
            limiter = get_rate_limiter()
            rl_status = "✅ 已启用" if limiter.enabled else "⏸️ 已禁用"
        except Exception:
            rl_status = "⚠️ 未加载"

        status_items = [
            ("LLM", cfg.llm.model),
            ("Embedding", cfg.embedding.model),
            ("Collection", cfg.milvus.collection_name),
            ("Chunk", f"{cfg.chunking.chunk_size}字符"),
            ("限流", rl_status),
        ]
        for label, value in status_items:
            st.caption(f"• **{label}**: {value}")

    return params


# ═══════════════════════════════════════════════════════════
# 主界面
# ═══════════════════════════════════════════════════════════

def main():
    # ── 初始化 session ──
    defaults = {
        "messages": [],
        "quick_input": None,
        "session_id": uuid.uuid4().hex[:12],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    chain = init_chain()
    params = render_sidebar()

    # ── 空会话 → 欢迎页 ──
    if not st.session_state.messages:
        render_welcome()

    # ── 渲染历史消息 ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(msg["answer"])

                # JSON 块
                if msg.get("output_format") == "json" and msg.get("answer"):
                    import json as _json
                    try:
                        parsed = _json.loads(msg["answer"])
                        with st.expander("📋 JSON 视图", expanded=False):
                            st.json(parsed)
                    except _json.JSONDecodeError:
                        pass

                render_meta_tags(
                    has_answer=msg.get("has_answer", False),
                    from_cache=msg.get("from_cache", False),
                    mode=msg.get("mode", "hybrid"),
                    latency_ms=msg.get("latency", 0),
                    source_count=len(msg.get("sources", [])),
                )
                render_sources(msg.get("sources", []))
            else:
                st.markdown(msg["content"])

    # ── 快捷问题注入 ──
    pending = st.session_state.quick_input
    if pending:
        st.session_state.quick_input = None

    # ── 输入框 ──
    prompt = st.chat_input(
        '描述设备故障现象，如 CT扫描图像有伪影怎么排查？'
    )
    question = prompt or pending
    if not question:
        return

    # ── 用户消息 ──
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # ── RAG + 流式 ──
    with st.chat_message("assistant"):
        # 阶段 1: 检索
        status = st.empty()
        status.markdown(
            '<div class="status-box searching">'
            '<span class="status-pulse"></span>'
            '正在检索知识库...'
            '</div>',
            unsafe_allow_html=True,
        )

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
            status.empty()
            st.error(f"❌ 检索失败：{e}")
            st.session_state.messages.append({
                "role": "assistant",
                "answer": f"❌ 检索失败：{e}",
                "sources": [],
                "has_answer": False,
                "from_cache": False,
                "mode": params["mode"],
                "output_format": params.get("output_format", "text"),
                "latency": 0,
            })
            return

        sources = result["sources"]
        has_answer = result["has_answer"]
        from_cache = result.get("from_cache", False)

        # 阶段 2: 生成
        if from_cache:
            status.markdown(
                '<div class="status-box done">'
                '⚡ 语义缓存命中 · 跳过 LLM 调用'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            status.markdown(
                '<div class="status-box generating">'
                '<span class="status-pulse"></span>'
                f'检索完成（{len(sources)} 条）· 正在生成回答...'
                '</div>',
                unsafe_allow_html=True,
            )

        # 流式输出
        answer_placeholder = st.empty()
        full_answer = ""

        try:
            for chunk in result["stream"]:
                full_answer += chunk
                answer_placeholder.markdown(
                    full_answer
                    + '<span class="typing-cursor">|</span>',
                    unsafe_allow_html=True,
                )
            # 去掉光标
            answer_placeholder.markdown(full_answer)
        except Exception as e:
            answer_placeholder.error(f"❌ 生成失败：{e}")
            full_answer = f"❌ 生成失败：{e}"

        latency = (time.perf_counter() - t0) * 1000
        status.empty()

        # JSON 视图
        if result.get("output_format") == "json":
            import json as _json
            try:
                parsed = _json.loads(full_answer)
                with st.expander("📋 JSON 结构化视图", expanded=True):
                    st.json(parsed)
            except _json.JSONDecodeError:
                st.caption("⚠️ 返回内容非合法 JSON，已显示原始文本")

        # 元信息
        render_meta_tags(has_answer, from_cache, result["mode"], latency, len(sources))

        # 来源
        if not from_cache:
            render_sources(sources)

    # ── 存入历史 ──
    st.session_state.messages.append({
        "role": "assistant",
        "answer": full_answer,
        "sources": sources,
        "has_answer": has_answer,
        "from_cache": from_cache,
        "mode": result["mode"],
        "output_format": result.get("output_format", "text"),
        "latency": latency,
    })


if __name__ == "__main__":
    main()
