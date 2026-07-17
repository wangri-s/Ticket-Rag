"""
RAG 问答 API — 检索 + LLM 生成全链路

端点:
  POST /api/ask         — RAG 问答（检索 + 生成）
  POST /api/ask/stream  — RAG 问答流式输出（SSE）
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.llm.rag_chain import RAGChain
from src.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG 问答"])


# ── 请求/响应模型 ────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户问题",
        examples=["CT扫描图像伪影是什么原因？"],
    )
    mode: str = Field(
        default="hybrid",
        pattern=r"^(semantic|keyword|hybrid)$",
        description="检索模式: semantic | keyword | hybrid",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="检索条数",
    )
    ticket_id: Optional[str] = Field(
        default=None,
        max_length=50,
        description="按工单号过滤（如 GD-2026-03001）",
    )


class SourceItem(BaseModel):
    ticket_id: str
    content: str
    score: float
    source: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    mode: str
    has_answer: bool
    latency_ms: float = Field(description="总耗时（毫秒）")


# ── 懒加载 RAGChain（模块级单例）──────────────

_chain: Optional[RAGChain] = None


def _get_chain() -> RAGChain:
    global _chain
    if _chain is None:
        logger.info("正在初始化 RAGChain...")
        _chain = RAGChain()
        logger.info("RAGChain 初始化完成")
    return _chain


# ── 端点 ─────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
def rag_ask(req: AskRequest):
    """
    RAG 问答（检索 + LLM 生成）。

    流程：用户问题 → 向量检索（语义/关键字/混合）→ 构建 Prompt → LLM 生成 → 返回答案 + 引用来源。

    无相关结果时返回兜底回答（has_answer=false, sources=[]）。
    """
    t0 = time.perf_counter()
    chain = _get_chain()

    result = chain.ask(
        question=req.question,
        mode=req.mode,          # type: ignore
        top_k=req.top_k,
        ticket_id_filter=req.ticket_id,
    )

    latency = (time.perf_counter() - t0) * 1000
    logger.info(f"RAG ask: mode={req.mode} latency={latency:.0f}ms "
                f"sources={len(result['sources'])} has_answer={result['has_answer']}")

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        mode=result["mode"],
        has_answer=result["has_answer"],
        latency_ms=round(latency, 1),
    )


@router.post("/ask/stream")
def rag_ask_stream(req: AskRequest):
    """
    RAG 问答 — SSE 流式输出。

    适用于前端逐字展示回答。事件格式：
      event: source  → 引用来源（JSON）
      event: delta   → 回答片段（文本，逐 token 输出）
      event: done    → 结束信号

    使用方式（curl）:
      curl -N -X POST http://localhost:8000/api/ask/stream \
        -H "Content-Type: application/json" \
        -d '{"question":"CT伪影怎么处理？","mode":"hybrid","top_k":3}'
    """
    chain = _get_chain()

    # 先检索
    result = chain.ask(
        question=req.question,
        mode=req.mode,          # type: ignore
        top_k=req.top_k,
        ticket_id_filter=req.ticket_id,
    )

    async def event_stream():
        import json

        # 1. 发送来源信息
        sources_data = json.dumps(
            [{"ticket_id": s["ticket_id"], "score": s["score"],
              "content": s["content"][:200]} for s in result["sources"]],
            ensure_ascii=False,
        )
        yield f"event: sources\ndata: {sources_data}\n\n"

        # 2. 逐字发送回答（模拟流式，LLM 未开启 stream 时一次性发送）
        answer = result["answer"]
        chunk_size = 10  # 每次发送 10 个字符
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i + chunk_size]
            yield f"event: delta\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # 3. 结束
        has_answer = "true" if result["has_answer"] else "false"
        yield f"event: done\ndata: {json.dumps({'has_answer': result['has_answer']}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/ask/health")
def ask_health():
    """检查 RAGChain 是否已初始化"""
    if _chain is not None:
        return {"status": "ready", "chain": "initialized"}
    return {"status": "warming", "chain": "not initialized yet"}
