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
  - 包含 10 份模拟医疗设备运维工单（GD-2026-03001 ~ GD-2026-03010）
  - 覆盖设备：CT、MRI、生化分析仪、超声、监护仪、血液透析、DR、麻醉机、灭菌器、HIS
  - 每份工单包含：编号、时间、设备型号、科室、故障现象、排查步骤、处理方案、处理结果
  - 故障类型多样化：硬件老化、传感器漂移、网络故障、软件性能、耗材更换
- **状态**：✅ 完成

## 步骤 10：完善 config.yml chunking 配置
- **时间**：2026-07-14
- **操作**：细化 `config.yml` 中 `chunking` 配置，为文档分块做准备：
  - 新增 `strategy: recursive` — 递归字符分割策略
  - 新增 `separators` — 7 级分隔符优先级（工单分隔线 → 字段标签 → 步骤 → 段落 → 句子）
  - 同步更新 `src/config.py` 中 `ChunkingConfig` 类，新增 `strategy`、`separators` 两个字段
  - 后续：去掉 `encoding`，改用纯字符计数分块（500 中文字符远在 text-embedding-v1 的 2048 token 限制内）
- **状态**：✅ 完成

## 步骤 11：创建文本分割器 chunker.py
- **时间**：2026-07-14
- **操作**：创建 `src/ingestion/chunker.py`，`MedicalWorkOrderChunker` 类：
  - 基于 `RecursiveCharacterTextSplitter`，使用 config.yml 中 7 级分隔符递归分块
  - `length_function=len` 纯字符计数，不依赖 token 编码器
  - `keep_separator=True` 保留分隔符，提升检索上下文可读性
  - `split_text()` — 纯文本分块，自动过滤短于 `min_chunk_length` 的碎片
  - `split_documents()` — LangChain Document 对象分块，继承 metadata 并追加 `chunk_index`
  - 重构：`split_text_with_overlap_control()` 替换为 `split_with_stats()`，返回 chunks + count + avg/min/max 尺寸 + ticket_ids
  - 新增 `extract_ticket_id()` 函数，正则匹配工单编号（GD-YYYY-MMDDD），自动注入每个块的 metadata
  - 新增 `group_by_ticket()` 方法，将分块结果按工单编号分组，支持检索结果溯源
  - `split_documents()` 现在自动为每块追加 `ticket_id` 到 metadata
- **状态**：✅ 完成（重构）

## 步骤 12：创建文档加载器 + 摄入流水线
- **时间**：2026-07-14
- **操作**：
  - 创建 `src/ingestion/loader.py` — `DocumentLoader` 统一加载器：
    - `load()` 单文件加载，根据后缀自动选策略（TXT → TextLoader / PDF → PyPDFLoader / MD → 按文本）
    - `load_directory()` 批量加载目录，递归扫描，自动跳过非文档文件
    - 每个文档携带 `file_type`、`source` 等 metadata
  - 创建 `src/ingestion/pipeline.py` — `IngestionPipeline` 摄入流水线：
    - `run(source)` 编排完整流程：加载 → 分块 → 统计输出
    - `run_with_stats(source)` 返回结构化统计数据（total_chunks、ticket_ids、avg/min/max）
    - `group_by_ticket()` 按工单编号分组，支持溯源
    - **实测验证**：工单知识库 → 1 文档 7118 字符 → 22 块，avg=322 字符/块，max=487（未超 500），覆盖全部 10 张工单
- **状态**：✅ 完成

## 步骤 13：创建命令行工具 cli.py
- **时间**：2026-07-14
- **操作**：创建 `src/cli.py`，三个子命令：
  - `ingest -s <路径>` — 摄入文档（加载+分块+统计），支持 `--json` 供脚本调用
  - `chunk-preview -f <文件> -n <数量>` — 预览分块效果，显示每块内容+工单编号+字符数
  - `stats` — 显示系统配置摘要，校验必填环境变量
- **状态**：✅ 完成

## 步骤 14：创建测试文件
- **时间**：2026-07-14
- **操作**：创建 4 个测试文件，覆盖已完成的全部模块：
  - `tests/test_config.py` — 9 个测试：环境变量解析（含 .env 覆盖）、配置加载、单例、必填校验
  - `tests/test_chunker.py` — 14 个测试：工单编号提取、分块/过滤/分隔符优先级、元数据继承、统计、分组
  - `tests/test_loader.py` — 6 个测试：TXT 加载、文件不存在/格式不支持错误、目录加载、元数据
  - `tests/test_pipeline.py` — 5 个测试：目录/单文件摄入、stats 统计、空目录、分组
  - 使用 `pytest`，**37 tests passed, 0 failed**
- **状态**：✅ 完成
