"""
RAG 智能工单系统 — FastAPI 应用入口
"""

from fastapi import FastAPI

app = FastAPI(
    title="智能工单 RAG 系统",
    description="基于 Milvus + Qwen-Max 的医疗设备运维工单检索增强生成系统",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "rag-ticket-system"}


@app.get("/")
async def root():
    return {"message": "智能工单 RAG 系统已启动", "docs": "/docs"}
