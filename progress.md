# 项目进度记录

## 步骤 1：连接远程 Git 仓库
- **时间**：2026-07-13
- **操作**：
  - 初始化本地 Git 仓库：`git init`
  - 添加远程仓库：`git remote add origin https://github.com/wangri-s/Ticket-Rag.git`
  - 执行 `git fetch origin`（远程仓库当前为空）
- **状态**：✅ 完成

## 步骤 2：创建 progress.md 文件
- **时间**：2026-07-13
- **操作**：在项目根目录下创建 `progress.md`，用于记录项目每一步操作
- **状态**：✅ 完成

## 步骤 3：创建 docker-compose.yml
- **时间**：2026-07-13
- **操作**：创建 `docker-compose.yml`，包含以下服务：
  - **etcd** — Milvus 元数据存储（quay.io/coreos/etcd:v3.5.5）
  - **minio** — Milvus 对象存储（minio/minio）
  - **milvus-standalone** — 向量数据库主服务（milvusdb/milvus:latest），依赖 etcd 和 minio 健康检查通过
  - **rag-app** — RAG 应用容器，从项目 Dockerfile 构建，依赖 milvus 服务，环境变量从 `.env` 文件读取
  - 所有服务运行在 `rag-network` 桥接网络中
  - 持久化卷：`etcd_data`、`minio_data`、`milvus_data`
- **状态**：✅ 完成

## 步骤 4：创建 config.yml（应用配置文件）
- **时间**：2026-07-13
- **操作**：创建 `config.yml`，包含应用运行所需的全部配置项：
  - **milvus** — 连接参数（host/port 通过环境变量读取）、collection 名 `medical_chunks`、向量维度 1536、索引类型 IVF_FLAT
  - **embedding** — 嵌入模型 `text-embedding-v1`（DashScope），API Key 从环境变量读取
  - **llm** — 对话模型 `qwen-max`，temperature 0.1，max_tokens 1500
  - **retrieval** — 检索 top_k=5，score_threshold=0.0
  - **chunking** — 分块大小 500，重叠 50，最小长度 20
- **状态**：✅ 完成

## 步骤 5：创建 .env.example 和环境变量读取代码
- **时间**：2026-07-13
- **操作**：
  - 创建 `.env.example` — 环境变量模板，包含 `MILVUS_HOST`、`MILVUS_PORT`、`DASHSCOPE_API_KEY`
  - 创建 `src/config.py` — 配置加载模块，核心功能：
    - 通过 `python-dotenv` 加载 `.env` 文件
    - 正则解析 `config.yml` 中 `${ENV_VAR:-default}` 占位符并递归替换
    - 提供 `MilvusConfig`、`EmbeddingConfig`、`LLMConfig`、`RetrievalConfig`、`ChunkingConfig` 五个类型化配置类
    - `AppConfig` 统一入口 + `get_config()` 单例
    - `validate()` 方法校验必填环境变量
  - 创建 `requirements.txt` — Python 依赖清单
- **状态**：✅ 完成

## 步骤 6：完善 requirements.txt
- **时间**：2026-07-13
- **操作**：补充完整项目依赖，按功能分组：
  - **LLM & Embedding** — `dashscope`
  - **向量数据库** — `pymilvus`
  - **LangChain 生态** — `langchain`、`langchain-community`
  - **文档处理** — `pypdf`、`tiktoken`
  - **配置** — `pyyaml`、`python-dotenv`
  - **Web 框架** — `fastapi`、`uvicorn`
  - **UI** — `streamlit`
  - **网络请求** — `requests`、`sseclient-py`
- **状态**：✅ 完成

## 步骤 7：创建 Dockerfile
- **时间**：2026-07-13
- **操作**：创建 `Dockerfile`，主要内容：
  - 基础镜像：`python:3.11-slim`
  - 安装系统依赖 `curl`（健康检查用）
  - 设置 `PYTHONPATH=/app`、禁止字节码、无缓冲输出
  - 分层复制：先 `requirements.txt` → pip install，再复制源码（利用 Docker 缓存层）
  - 暴露端口 `8000`
  - 默认启动命令：`uvicorn src.main:app --host 0.0.0.0 --port 8000`
- **状态**：✅ 完成

## 步骤 8：生成项目目录结构
- **时间**：2026-07-13
- **操作**：创建智能工单 RAG 系统的完整目录结构：
  - `src/ingestion/` — 文档加载、解析、分块
  - `src/embedding/` — 文本向量化（DashScope）
  - `src/retrieval/` — Milvus 语义检索
  - `src/llm/` — LLM 调用封装（qwen-max）
  - `src/api/` — FastAPI REST 接口
  - `src/ui/` — Streamlit 前端界面
  - `src/utils/` — 通用工具函数
  - `src/config.py` — 配置加载器
  - `data/documents/` — 原始文档存放
  - `data/vectors/` — 向量数据备份
  - `tests/` — 单元测试
- **状态**：✅ 完成

## 步骤 9：创建 RAG 知识库工单文档
- **时间**：2026-07-13
- **操作**：创建 `data/documents/工单知识库.txt`，作为 RAG 系统的检索知识源：
  - 包含 10 份模拟医疗设备运维工单（GD-2026-07001 ~ GD-2026-07010）
  - 覆盖设备：CT、MRI、生化分析仪、超声、监护仪、血液透析、DR、麻醉机、灭菌器、HIS
  - 每份工单包含：编号、时间、设备型号、科室、故障现象、排查步骤、处理方案、处理结果
  - 故障类型多样化：硬件老化、传感器漂移、网络故障、软件性能、耗材更换
- **状态**：✅ 完成
