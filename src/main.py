"""
RAG 智能工单系统 — FastAPI 应用入口

启动方式:
  uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
  python -m src.main                              # 直接运行（默认 8000 端口）
"""

import logging
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.search import router as search_router
from src.api.ask import router as ask_router

# ── 日志 ──────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")


# ── 生命周期 ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时执行"""
    logger.info("=" * 50)
    logger.info("智能工单 RAG 系统 API 启动中...")
    logger.info("=" * 50)
    yield
    logger.info("智能工单 RAG 系统 API 已关闭")


# ── 创建应用 ──────────────────────────────────

app = FastAPI(
    title="智能工单 RAG 系统",
    description="""
基于 Milvus + Qwen-Max 的医疗设备运维工单检索增强生成系统。

## 核心能力

- **RAG 问答**：检索 + LLM 生成，支持 3 种检索模式
- **单独检索**：不经过 LLM，直接返回匹配的工单片段
- **流式输出**：SSE 逐字返回回答（适合前端）

## 检索模式

| 模式 | 原理 | 适用场景 |
|------|------|---------|
| `semantic` | 稠密向量语义相似度 | 自然语言描述的问题 |
| `keyword` | BM25 稀疏向量关键词匹配 | 精确术语、工单号搜索 |
| `hybrid` | 语义 + 关键字加权重排 | 通用场景（推荐） |
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── 中间件 ────────────────────────────────────

# CORS — 允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志 + 耗时统计
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({elapsed:.0f}ms)"
    )
    return response


# ── 全局异常处理 ──────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"未处理的异常: {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "服务器内部错误，请稍后重试",
            "detail": str(exc) if app.debug else None,
        },
    )


# ── 注册路由 ──────────────────────────────────

app.include_router(search_router)
app.include_router(ask_router)


# ── 根路由 ────────────────────────────────────

@app.get("/")
async def root():
    """API 导航页"""
    return {
        "service": "智能工单 RAG 系统",
        "version": "1.0.0",
        "endpoints": {
            # RAG 问答
            "rag_ask": "POST /api/ask",
            "rag_ask_stream": "POST /api/ask/stream",
            # 单独检索
            "search_semantic": "POST /api/search/semantic",
            "search_keyword": "POST /api/search/keyword",
            "search_hybrid": "POST /api/search/hybrid",
            # 运维
            "health": "GET /health",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "rag-ticket-system"}


# ── 直接运行入口 ──────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print()
    print("=" * 55)
    print("  智能工单 RAG 系统 API")
    print("  Swagger UI: http://localhost:8000/docs")
    print("  ReDoc:      http://localhost:8000/redoc")
    print("  健康检查:    http://localhost:8000/health")
    print("=" * 55)
    print()

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
