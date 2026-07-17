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

## 步骤 15：完善 config.yml Milvus + Embedding 配置
- **时间**：2026-07-14
- **操作**：扩展 `config.yml` 和 `src/config.py` 中 Milvus 和 Embedding 配置段：
  - **Milvus 新增** — `timeout`（30s）、`max_retries`（3次）、`search_params.nprobe`（16，搜索探测聚类数）、`consistency_level`（Strong，强一致性读）
  - **Embedding 新增** — `dimensions`（1536，与向量维度校验）、`max_retries`（3次）、`timeout`（60s）、`sleep_interval`（1s，批量间冷却防限流）
- **状态**：✅ 完成

## 步骤 16：修复 chunker 工单编号继承
- **时间**：2026-07-14
- **操作**：修复 `split_documents()` 中工单编号丢失问题：
  - **问题**：一张工单被切成 2-3 块时，只有包含 `【工单编号】` 的块能提取到 ticket_id，后续块（处理方案/结果）丢失编号，被归入 `__unmatched__`
  - **修复**：引入 `last_ticket_id` 变量，顺序遍历分块时记住最近一次提取到的编号，当前块提取不到则继承
  - **效果**：`__unmatched__` 从 12 块降至 1 块（仅文档标题），10 张工单每张正确关联 2-3 块
- **状态**：✅ 完成

## 步骤 17：创建 Embedding 客户端
- **时间**：2026-07-14
- **操作**：创建 `src/embedding/embedding_client.py`，`EmbeddingClient` 类：
  - `embed(text)` — 单条文本 → 1536 维向量
  - `embed_batch(texts)` — 批量向量化，自动按 `batch_size` 分批，批次间 `sleep_interval` 防限流
  - `_call_with_retry()` — 失败自动重试（max_retries=3），指数退避
  - `_validate_dim()` — 校验输出维度与配置一致
  - **实测验证**：`embed("CT扫描过程中偶发图像伪影")` → 1536 维向量，API 调用成功
- **状态**：✅ 完成

## 步骤 18：创建 Milvus 操作客户端
- **时间**：2026-07-15
- **操作**：创建 `src/retrieval/milvus_client.py`，`MilvusStore` 类：
  - 基于新版 `pymilvus.MilvusClient`（非 ORM 风格，避免 PyMilvus 3.1 弃用）
  - `create_collection()` — 创建 collection，6 字段 schema（id/content/embedding/ticket_id/source/chunk_index），自动建 IVF_FLAT 索引
  - `insert()` — 批量写入向量+元数据，自动 flush 持久化
  - `search()` — 语义搜索，返回 top_k 结果含 id/score/content/ticket_id，支持标量过滤
  - `collection_exists()` / `num_entities` / `drop_collection()` — 工具方法
  - 同步修复：`.env` 中 `MILVUS_HOST` 改为 `localhost`（本机开发），`docker-compose.yml` 中 `environment` 覆盖为 `milvus-standalone`（容器内 DNS）
- **状态**：✅ 完成

## 步骤 19：创建 ETL 构建脚本 + Milvus 测试验证
- **时间**：2026-07-15
- **操作**：
  - 创建 `build_milvus.py` — 端到端知识库构建脚本，串联全部已开发模块：
    - `[1/4]` 文档加载（DocumentLoader）
    - `[2/4]` 文本分块（MedicalWorkOrderChunker）
    - `[3/4]` 批量向量化（EmbeddingClient，自动分批+限流）
    - `[4/4]` 写入 Milvus（MilvusStore，首次建库/增量追加）
    - 支持 `--rebuild`（删库重建）、`--dry-run`（只分块预览）、`--source`（指定文档目录）
    - **实测**：工单知识库 1 文档 → 22 块 → 22 条向量 → Milvus 存储，总耗时 5.2s
  - 创建 `tests/test_milvus_manual.py` — 6 个 pytest 用例验证 Milvus 连接、增删查、标量过滤
    - **实测**：6 passed, 0 failed
  - **修复 `milvus_client.py` 3 个 API 兼容问题**：
    - dict schema → `FieldSchema` + `CollectionSchema` 对象（新版 pymilvus 拒绝 dict）
    - `create_index(params=...)` → `IndexParams().add_index(...)`（参数接口变更）
    - `insert()` 后加 `flush()` 持久化 + `create_collection()` 后加 `load_collection()` 加载到内存
- **状态**：✅ 完成

## 步骤 20：部署 Attu + 创建 test_milvus.py 测试文件
- **时间**：2026-07-17
- **操作**：
  - **部署 Attu**（Milvus 官方 GUI 管理工具）：
    - 在 `docker-compose.yml` 中新增 `attu` 服务（`zilliz/attu:latest`），映射端口 `3000:3000`
    - Attu 通过内部网络 `rag-network` 直连 `milvus-standalone:19530`
    - 访问 `http://localhost:3000` 即可管理 Milvus
  - **修复端口冲突**：Windows 系统保留 9091 端口，将 `milvus-standalone` 的 9091 映射改为 `19091:9091`
  - 创建 `tests/test_milvus.py` — 独立写入 + 检索验证测试文件：
    - `TEST_RECORDS` — 5 条模拟工单数据，覆盖 CT/MRI/生化/超声/血透五种设备
    - `write_test_data()` — 独立写入函数，向量化 → Milvus 一条龙，支持 `--rebuild`
    - **8 个 pytest 用例**：写入验证 x2、语义搜索 x3（CT伪影/MRI啸叫/透析电导率）、标量过滤、分数校验、空结果
    - `--write-only` 模式不依赖 pytest，可直接 `python tests/test_milvus.py --write-only` 运行
  - **实测**：Docker 4 服务全部 healthy → 写入 5 条 1536 维向量 → 8 tests passed, 0 failed
- **状态**：✅ 完成

## 步骤 21：运行构建脚本，chunks 向量化存入 Milvus
- **时间**：2026-07-17
- **操作**：
  - 安装缺失依赖 `langchain`、`langchain-community`、`pypdf`、`tiktoken`
  - 执行 `python build_milvus.py --rebuild` 端到端 ETL：
    - `[1/4]` 文档加载：1 文档（工单知识库.txt，7118 字符）
    - `[2/4]` 文本分块：22 块（avg=322, min=89, max=487 字符）
    - `[3/4]` 向量化：22 条 × 1536 维（DashScope text-embedding-v1）
    - `[4/4]` 写入 Milvus：删库重建，22 条全部写入
  - **实测**：总耗时 4.4s，Milvus 当前总量 22 条
- **状态**：✅ 完成

## 步骤 22：Schema 增加 sparse_vector，实现 BM25 关键字检索 + 混合检索
- **时间**：2026-07-17
- **操作**：
  - 安装 `milvus-model`（BM25 稀疏向量）、`jieba`（中文分词）
  - 新增 `src/embedding/sparse_embedder.py` — `BM25SparseEmbedder` 类：
    - 基于 `pymilvus.model.sparse.BM25EmbeddingFunction` + `jieba` 中文分词
    - `fit(corpus)` 训练词表 + IDF
    - `encode_documents(texts)` → Milvus sparse dict 列表
    - `encode_queries(queries)` → 查询稀疏向量
    - 兼容 scipy csr_array（encode_documents）和 coo_array（encode_queries）
  - 改造 `src/retrieval/milvus_client.py`：
    - Schema 新增 `sparse_embedding` 字段（`SPARSE_FLOAT_VECTOR`）
    - `create_collection()` 新增稀疏索引 `SPARSE_INVERTED_INDEX`
    - `insert()` 新增可选参数 `sparse_vectors`
    - 新增 `sparse_search()` — 纯关键字检索（BM25 稀疏向量匹配）
    - 新增 `hybrid_search()` — 语义 + 关键字加权重排（`WeightedRanker`）
  - 改造 `build_milvus.py`：
    - ETL 流程从 4 步扩展为 5 步：[3/5] BM25 训练+编码
    - 写入时同时传入稠密向量和稀疏向量
  - **实测**：
    - 构建：1 文档 → 22 块 → 22 稠密 + 22 稀疏向量 → 5.4s
    - 关键字检索 "CT伪影" → top1: GD-2026-03001（CT 工单）✅
    - 混合检索 "MRI啸叫" → top1-2: GD-2026-03002（MRI 工单）✅
- **状态**：✅ 完成

## 步骤 23：实现检索接口 + 简单测试
- **时间**：2026-07-17
- **操作**：
  - 新增 `src/api/search.py` — 检索 API 路由，三个端点：
    - `POST /api/search/semantic` — 语义检索（稠密向量 IP 相似度）
    - `POST /api/search/keyword` — 关键字检索（BM25 稀疏向量，首次调用自动训练）
    - `POST /api/search/hybrid` — 混合检索（WeightedRanker 加权重排）
    - 均支持 `ticket_id` 过滤 + `top_k` 控制
    - 使用 `@lru_cache` 懒加载 EmbeddingClient / BM25SparseEmbedder / MilvusStore
    - Pydantic 模型做请求校验和响应序列化
  - 更新 `src/main.py` — 注册 search_router，首页列出可用端点
  - 安装 `fastapi`、`uvicorn`
  - **实测**：`uvicorn src.main:app --port 8000` 启动后，4 组测试全部 200 ✅
- **状态**：✅ 完成

## 步骤 24：扩展 config.yml LLM 配置
- **时间**：2026-07-17
- **操作**：
  - `config.yml` LLM 段从 5 个基础字段扩展为 4 个区块：
    - **生成控制**：新增 `top_p`（0.8）、`seed`（42）、`stream`（false）
    - **连接**：新增 `max_retries`（3）、`timeout`（60s）
    - **RAG**：新增 `retrieval_top_k`（5）、`system_prompt`（307 字医疗运维专家人设+4 条安全守则）、`rag_prompt_template`（{context}/{question} 占位符）、`fallback_answer`（无结果兜底回复）
  - `src/config.py` `LLMConfig` 类同步新增 9 个属性，均带默认值
  - **实测**：`get_config().llm` 全部字段加载正确 ✅
- **状态**：✅ 完成

## 步骤 25：创建 llm_client.py 封装千问 LLM 调用
- **时间**：2026-07-17
- **操作**：
  - 新建 `src/llm/llm_client.py` — `LLMClient` 类，封装 DashScope 千问大模型：
    - `generate(user_message, system_prompt?)` — 单轮对话，自动注入 config.yml 预设的医疗专家 system prompt
    - `chat(messages)` — 多轮对话，调用 `Generation.call(model='qwen-max', ...)`
    - `_call_with_retry()` — 失败自动重试 3 次，指数退避（1s→2s→4s）
    - 每次调用记录 input/output token 用量
    - 所有参数（temperature/max_tokens/top_p/seed/timeout）从 `LLMConfig` 读取
  - **实测**：`generate("CT伪影原因")` 返回医疗分析 ✅，`chat([...])` 返回正确回答 ✅
- **状态**：✅ 完成

## 步骤 26：创建 prompts.py 智能工单提示词模块
- **时间**：2026-07-17
- **操作**：
  - 新建 `src/llm/prompts.py` — RAG 提示词管理模块，四个核心函数：
    - `format_chunks(chunks)` — 将检索结果格式化为可读文本（编号+工单号+来源+内容，去掉分隔线省 token）
    - `build_rag_prompt(question, chunks)` — 填入 config.yml 的 `rag_prompt_template`，{context}/{question} 占位符替换，无结果返回 None
    - `get_system_prompt()` / `get_fallback_answer()` — 读取 config 预设值
    - `build_full_prompt(question, chunks)` — 一次性返回 (system_prompt, user_message)，无结果返回 (None, None)
  - **实测**：2 个 chunk 格式化为 338 字符 user_message ✅，空 chunks 正确处理 ✅
- **状态**：✅ 完成

## 步骤 27：创建 rag_chain.py RAG 核心链
- **时间**：2026-07-17
- **操作**：
  - 新建 `src/llm/rag_chain.py` — `RAGChain` 类，串联检索→Prompt→LLM 全流程：
    - `ask(question, mode, top_k, ticket_id_filter)` — 主入口，返回 {"answer", "sources", "has_answer", "mode"}
    - 支持三种检索模式：`semantic` / `keyword` / `hybrid`
    - 组件懒加载：EmbeddingClient / BM25SparseEmbedder / MilvusStore / LLMClient 各只初始化一次
    - 按 `retrieval.score_threshold` 过滤低相关 chunk
    - 无结果时走 `fallback_answer`，不调 LLM
  - **实测**：
    - 混合检索 "CT伪影排查步骤" → 5 sources + 1263 字回答，引用 GD-2026-03001 ✅
    - 关键字检索 "透析液电导率" → 2 sources + 1218 字回答，引用 GD-2026-03006 ✅
    - 无匹配 "心脏搭桥手术" → 分数过滤生效，LLM 正确拒绝回答 ✅
- **状态**：✅ 完成

## 步骤 28：扩展 retrieval 配置 + 消除硬编码
- **时间**：2026-07-17
- **操作**：
  - `config.yml` retrieval 段从 2 个字段扩展为：
    - `top_k`（5）、`score_threshold`（0.0）、`default_mode`（hybrid）
    - 新增 `hybrid` 子段：`dense_weight`（0.5）、`sparse_weight`（0.5）
  - `src/config.py` `RetrievalConfig` 类同步新增 4 个属性
  - `src/llm/rag_chain.py` 消除 2 处硬编码：
    - 默认检索模式 → `cfg.retrieval.default_mode`
    - 混合检索权重 → `cfg.retrieval.hybrid_dense_weight / hybrid_sparse_weight`
  - 至此 retrieval 和 llm 配置全部通过 `get_config()` 驱动，无需改代码
- **状态**：✅ 完成

## 步骤 29：扩展 cli.py 增加 RAG 问答命令
- **时间**：2026-07-17
- **操作**：
  - `src/cli.py` 新增 `ask` 子命令：
    - **命令行模式**：`python -m src.cli ask -q "CT伪影怎么处理？" -m hybrid -k 3`
      - `-q` 指定问题，不传则进入交互模式
      - `-m` 检索模式（semantic/keyword/hybrid）
      - `-k` 检索条数
      - `--ticket-id` 按工单号过滤
    - **交互模式**：`python -m src.cli ask` 进入 REPL，逐条对话，输入 quit 退出
    - `_do_ask()` 统一格式化输出：回答 + 引用来源（工单号、score、内容预览）
  - **实测**：`ask -q "CT伪影排查步骤" -m hybrid` → 4 段回答 + 3 条引用来源 ✅
- **状态**：✅ 完成

## 步骤 30：创建 interactive.py 交互式问答终端
- **时间**：2026-07-17
- **操作**：
  - 新建 `src/interactive.py` — 独立交互式 RAG 问答终端：
    - **一次性初始化 6 个组件**：配置→Embedding→Milvus→BM25→LLM→RAGChain，启动时全数就绪，后续零延迟
    - **循环交互**：while True 循环，用户反复提问无需重启
    - **RAG 全链路**：嵌入问题→检索 chunks→拼接 Prompt→LLM 生成→打印回答+引用来源
    - **内置命令**：`mode` 切换检索模式、`topk` 调检索条数、`filter` 按工单过滤、`stats` 查看设置、`help` 帮助
    - **四层异常保护**：KeyboardInterrupt/EOFError 优雅退出，RAG 错误打印后继续，未知异常不崩溃
  - **实测**：`python -m src.interactive` → 6/6 组件加载 ✅
- **状态**：✅ 完成

## 步骤 31：生成 README.md 项目介绍文档
- **时间**：2026-07-17
- **操作**：
  - 新建 `README.md`，包含 9 个章节：
    - 项目概述 + 核心能力 + 技术栈
    - 完整目录结构说明
    - 快速开始（5 步从零跑通）
    - 三种检索模式对比
    - API 接口文档
    - RAG 问答流程图
    - 安全设计说明（4 条守则）
    - 测试覆盖（43 tests）
    - 配置参考
- **状态**：✅ 完成

## 步骤 32：合并交互模式到 cli.py，删除 interactive.py
- **时间**：2026-07-17
- **操作**：
  - 将 `interactive.py` 的功能全部合并进 `src/cli.py` 的 `ask` 子命令：
    - 无 `-q` 参数 → 交互模式（while True 循环 + 内置命令）
    - 内置命令：`mode`/`topk`/`filter`/`clear`/`help`/`stats`/`quit`
    - 交互状态（模式、条数、过滤）在对话中可变
    - 命令行模式保持原样 `ask -q "问题"`
  - 删除 `src/interactive.py`（不再需要单独文件）
  - `python -m src.cli ask` 入口统一
- **状态**：✅ 完成

## 步骤 33：修复 test_rag.py — 补断言 + 加边界测试
- **时间**：2026-07-18
- **操作**：审查并修复 `tests/test_rag.py`，共 6 处改动：
  - **修复 1** — `test_no_result_fallback`（原 174-183 行）：补上 3 个 assert（has_answer=False、sources=[]、兜底文本含"未找到"），原来只打印不校验
  - **修复 2** — `test_keyword_matches_ticket`（原 114-122 行）：补上 assert 验证 GD-2026-03006 在结果中，原来只打印 ticket_ids
  - **修复 3** — `test_ticket_filter`（原 157-168 行）：if 条件断言改为先 assert sources 非空再遍历，防止无结果时静默通过
  - **修复 4** — `test_hybrid_scores_descending`（原 141-151 行）：加 has_answer=True 断言，防止无结果时降序检查无意义
  - **修复 5** — 新增 `TestRAGEdgeCasesMore` 类：
    - `test_empty_question_no_crash` — 空字符串问题不崩溃
    - `test_invalid_mode_falls_back_to_hybrid` — 无效模式静默降级 hybrid
  - **修复 6** — 测试编号从 12 个扩展到 14 个，修正重复编号
  - **修复 7** — `src/cli.py` `_print_stats()` 函数缺少 `from src.config import get_config` 导入，补上局部导入（与其他函数 `cmd_stats`、`cmd_ask` 风格一致）
- **状态**：✅ 完成
