# ============================================================
# RAG 智能工单系统 — Dockerfile
# ============================================================

FROM python:3.11-slim

# ---------- 工作目录 ----------
WORKDIR /app

# ---------- Python 环境配置 ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ---------- 安装依赖 ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 复制源码 ----------
COPY . .

# ---------- 端口 ----------
EXPOSE 8000

# ---------- 启动命令（可在 docker-compose 中覆盖） ----------
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
