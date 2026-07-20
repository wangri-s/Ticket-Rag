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

## 步骤 34：适配 pymilvus 3.0 — 修复 sparse_embedder 导入路径
- **时间**：2026-07-18
- **操作**：
  - **问题**：pymilvus 3.0 移除了 `pymilvus.model` 子包，BM25 相关 API 迁移到独立包 `milvus_model`，导致 `from pymilvus.model.sparse import BM25EmbeddingFunction` 报 `ModuleNotFoundError`
  - **修复**：`src/embedding/sparse_embedder.py` 中 `pymilvus.model.sparse` → `milvus_model.sparse`，`pymilvus.model.sparse.bm25.tokenizers` → `milvus_model.sparse.bm25.tokenizers`
  - `requirements.txt` 补上缺失的 `milvus-model` 依赖
- **状态**：✅ 完成

## 步骤 35：修复 BM25 训练路径 — 相对路径 → 项目根绝对路径
- **时间**：2026-07-18
- **操作**：
  - **问题**：`rag_chain.py` `_get_sparse_embedder()` 中 `DocumentLoader().load_directory("data/documents")` 使用相对路径，PyCharm/pytest 从 `tests/` 目录运行测试时 cwd 不是项目根，导致 `NotADirectoryError`
  - **修复**：导入 `config.py` 的 `ROOT_DIR`（`Path(__file__).resolve().parent.parent`），改为 `load_directory(str(ROOT_DIR / "data/documents"))`
- **状态**：✅ 完成

## 步骤 36：修复 test_rag.py 两个用例预期与实际行为对齐
- **时间**：2026-07-18
- **操作**：
  - **问题 1** — `test_no_result_fallback`："心脏搭桥手术麻醉方案"中"手术"关键词会命中工单中的"手术室"等内容，keyword BM25 实际有命中导致 `has_answer=True`，断言 `False` 失败
  - **修复 1**：改用"Python编程语言装饰器用法详解"，与医疗工单完全无交集的查询，确保 keyword 零命中走兜底
  - **问题 2** — `test_empty_question_no_crash`：`EmbeddingClient.embed()` 对空字符串主动抛 `ValueError("输入文本为空")`，原测试期望不崩溃与实际行为不符
  - **修复 2**：改为 `pytest.raises(ValueError, match="输入文本为空")` 匹配预期抛异常
  - **实测**：14 tests passed, 0 failed ✅
- **状态**：✅ 完成

## 步骤 37：生成 FastAPI 完整接口（RAG 问答 + 流式 + 中间件）
- **时间**：2026-07-18
- **操作**：
  - **新建 `src/api/ask.py`** — RAG 问答 API（检索 + LLM 生成全链路）：
    - `POST /api/ask` — RAG 问答，请求体含 question/mode/top_k/ticket_id，返回 answer + sources + latency_ms
    - `POST /api/ask/stream` — SSE 流式输出，前端可逐字展示回答
    - `GET /api/ask/health` — 检查 RAGChain 初始化状态
    - 请求模型 `AskRequest` 含参数校验（question 1-2000字符、mode 正则校验、top_k 1-20）
    - RAGChain 模块级懒加载单例，首次调用初始化，后续零等待
  - **修复 `src/api/search.py`** — 两处改动：
    - 补 `from src.config import ROOT_DIR` 导入
    - `_get_sparse_embedder()` 中 `load_directory("data/documents")` → `load_directory(str(ROOT_DIR / "data/documents"))`，修复非项目根目录启动时路径错误（与步骤 35 同款 bug）
  - **重写 `src/main.py`** — 应用入口升级：
    - 新增 CORS 中间件（`allow_origins=["*"]`），允许前端跨域
    - 新增请求日志中间件（记录 method + path + status + 耗时）
    - 新增全局异常捕获（500 → JSON 友好错误信息）
    - 新增 lifespan 生命周期日志
    - 注册 `search_router` + `ask_router`
    - 更新 `/` 根路由，列出全部 7 个端点
    - `__main__` 直接运行入口（`uvicorn.run` with reload）
  - **依赖**：`fastapi` + `uvicorn` 已在 requirements.txt 中 ✅
- **状态**：✅ 完成

## 步骤 38：生成 Streamlit 前端页面
- **时间**：2026-07-18
- **操作**：
  - **新建 `src/ui/app.py`** — 简洁 RAG 问答前端：
    - **侧边栏**：检索模式（下拉）、top_k（滑块 1-10）、工单号过滤（输入框）、系统配置（折叠面板）
    - **主区域**：标题 + 5 个快捷测试问题按钮（点击填入输入框）+ 提问/清空按钮
    - **回答区**：耗时 + 模式 + 来源数 + 命中状态 + LLM 回答（命中绿/兜底黄）
    - **来源区**：每条折叠面板，含分数进度条可视化、完整内容展示
    - **性能**：`@st.cache_resource` 缓存 RAGChain，刷新不复初始化
    - **异常处理**：try/except 捕获 RAG 链路错误
  - **依赖**：`streamlit>=1.35.0` 已在 requirements.txt ✅
- **状态**：✅ 完成

## 步骤 39：重写 Streamlit 为经典 LLM 对话界面
- **时间**：2026-07-18
- **操作**：
  - **重写 `src/ui/app.py`** — 从表单式改为对话式：
    - **双气泡对话**：`st.chat_message("user")` + `st.chat_message("assistant")`，用户右/助手左
    - **对话历史**：`st.session_state.messages` 保存完整记录，刷新不丢失
    - **底部输入框**：`st.chat_input` 固定在底部，Enter 发送
    - **即时反馈**：先显示 "⏳ 正在检索+生成..." 占位，完成后替换为正式回答
    - **元信息**：每条助手回复下方灰色小字显示命中状态、检索模式、来源数、耗时
    - **来源折叠**：每条助手消息内嵌 `expander`，展开查看来源 + 分数进度条
    - **侧边栏快捷问题**：8 个按钮点击即发送，同时保留检索参数设置
    - **清空对话按钮**：侧边栏一键重置历史
  - **实测**：UI 布局与 ChatGPT/Claude 风格一致 ✅
- **状态**：✅ 完成

## 步骤 40：LLM 流式输出 + Streamlit 打字机效果
- **时间**：2026-07-18
- **操作**：
  - **`src/llm/llm_client.py`** — 新增 `generate_stream()` 方法：
    - 使用 DashScope `stream=True + incremental_output=True` 逐 token 获取增量
    - `_call_stream()` 内部方法，yield 每个文本增量
    - 自动统计输出字符数
  - **`src/llm/rag_chain.py`** — 新增 `ask_stream()` 方法：
    - 检索逻辑与 `ask()` 一致（三种模式 + 阈值过滤 + 兜底）
    - 返回 `{"stream": generator, "sources": [...], ...}`，调用方迭代 stream 字段获取增量
  - **`src/ui/app.py`** — 流式打字机效果：
    - 用 `chain.ask_stream()` 代替 `chain.ask()`
    - 先显示 "🔍 正在检索..." → 检索完成显示 "✍️ 正在生成..."
    - `for chunk in result["stream"]` 逐段渲染，末尾带闪烁光标 `▌`
    - 生成完成后光标消失，追加元信息 + 来源折叠
- **状态**：✅ 完成

## 步骤 41：更新 README.md
- **时间**：2026-07-18
- **操作**：
  - 更新项目结构（移除已删除的 `interactive.py`，新增 `api/ask.py`、`ui/app.py`、`tests/test_rag.py`）
  - 更新技术栈（milvus-model、Streamlit、流式输出）
  - 新增 API 接口文档（`/api/ask` + `/api/ask/stream` + 请求/响应示例）
  - 新增前端界面章节（Streamlit 对话式 UI）
  - 修正快速开始中的命令（`src.interactive` → `src.cli ask`）
  - 更新测试说明（加入 RAG 全链路集成测试）
- **状态**：✅ 完成

## 步骤 42：检索增加 Reranker 重排序阶段（qwen3-rerank 交叉编码器）
- **时间**：2026-07-18
- **操作**：
  - **优化方式**：**两阶段检索（粗召回 + 精排）**
    - **Stage 1 — 粗召回**：双路混合检索（稠密 IVF_FLAT + 稀疏 BM25），多召回 top_k × 3 条候选（上限 20）
    - **Stage 2 — 精排**：DashScope **qwen3-rerank 交叉编码器**对每个候选 (query, chunk) 对独立打分（非向量余弦），按 relevance_score 降序取 top_k
    - **与双编码器区别**：embedding 模型分别编码 query 和 doc，通过向量余弦计算相似度；交叉编码器将 query+doc 拼接后过 Transformer，直接输出相关性分数，显著更精准
  - **新增 `src/retrieval/reranker.py`**：
    - `Reranker` 类：封装 `dashscope.TextReRank.call(model="qwen3-rerank")`
    - `rerank(query, chunks, top_n)` 方法：传入初检结果，返回精排后列表（score 替换为 relevance_score）
    - 失败降级：API 异常时回退原始向量排序，保证可用性
    - 内容截断：单条 >3000 字符自动截断，避免超 4000 token 上限
  - **更新 `config.yml`** — `retrieval` 下新增 `rerank` 配置节：
    - `enabled: false`（默认关闭，按需开启）
    - `model: qwen3-rerank` / `oversample_factor: 3` / `max_retries: 3` / `timeout: 30`
  - **更新 `src/config.py`**：
    - 新增 `RerankConfig` 数据类（enabled/model/api_key/oversample_factor/max_retries/timeout）
    - `RetrievalConfig` 新增 `self.rerank = RerankConfig(...)`
    - `AppConfig` 新增 `self.rerank` 快捷访问，自动注入 DASHSCOPE_API_KEY
  - **更新 `src/llm/rag_chain.py`**：
    - 新增 `_get_reranker()` 懒加载 + `_retrieve_with_rerank()` 两阶段检索方法
    - `ask()` 和 `ask_stream()` 新增 `rerank: bool = None` 参数（None=取 config 默认值）
    - 日志增加 `rerank=True/False` 标记
  - **更新 `src/api/ask.py`** — `AskRequest` 新增 `rerank: bool = False`
  - **更新 `src/ui/app.py`** — 侧边栏检索参数区新增"启用重排序"复选框（默认关闭，hover 提示说明原理）
- **实测**：rerank=True 时检索日志显示 `oversample_k=15 → rerank candidates=15 → top_n=5` ✅
- **状态**：✅ 完成

## 步骤 43：增加 Query 预处理（查询改写 + 多路召回扩展点）
- **时间**：2026-07-18
- **操作**：
  - **优化方式**：**检索前查询预处理**，弥合用户口语化描述与知识库专业术语之间的差距
    - **Query Rewrite（查询改写）**：LLM 将口语问题改写为检索友好的关键词组合
      - Prompt 设计：指定医疗设备领域角色 + 4 条规则 + 3 个示例（Few-shot）
      - 示例："机器拍出来的片子不清楚"→"CT 图像伪影 分辨率下降 探测器校准"
      - 改写结果只用于检索，LLM 生成仍用原始问题（保证回答匹配用户原意）
      - 失败降级：rewrite 失败时自动回退原查询
    - **Multi-Query（多路召回）**：保留扩展点（LLM 生成 N 个变体各自检索合并去重），默认关闭
  - **新增 `src/retrieval/query_processor.py`**：
    - `QueryProcessor` 类：封装 LLM 查询改写/扩展
    - `rewrite(question)→str`：口语→关键词，使用独立 LLMClient 实例（低温度）
    - `expand(question, n)→list[str]`：生成 N 个语义等价变体
    - 两个 Prompt 模板：QUERY_REWRITE_PROMPT（Few-shot）、MULTI_QUERY_PROMPT
  - **更新 `config.yml`** — `retrieval` 下新增 `query_preprocess` 配置节：
    - `enabled: false`（默认关闭）/ `rewrite: true` / `multi_query: false` / `multi_query_n: 3`
  - **更新 `src/config.py`** — 新增 `QueryPreprocessConfig`
  - **更新 `src/llm/rag_chain.py`**：
    - 新增 `_get_query_processor()` 懒加载 + `_expand_query()` 方法
    - `ask()` 和 `ask_stream()` 新增 `query_expansion` 参数
    - 检索前先改写 query，Prompt 仍用原始问题
  - **更新 `src/api/ask.py`** — `AskRequest` 新增 `query_expansion: bool = False`
  - **更新 `src/ui/app.py`** — 侧边栏新增"启用查询扩展"复选框
- **实测**：`question="机器拍出来的片子不清楚"` → `rewrite="CT 图像伪影 分辨率下降 探测器校准"` ✅
- **状态**：✅ 完成

## 步骤 44：分词器显式切换为 jieba + 加载医疗领域自定义词典
- **时间**：2026-07-20
- **操作**：
  - **背景**：`build_default_analyzer(language="zh")` 底层已是 `JiebaTokenizer`，但代码中不可见
  - **改为显式**：`import jieba` + 注释标明分词链路（原始文本→jieba.cut→tokens→BM25→稀疏向量）
  - **新增医疗领域自定义词典**：`_MEDICAL_TERMS` 列表包含 25 个医学术语，通过 `jieba.add_word()` 预加载
    - 设备名称：血液透析机、生化分析仪、超声诊断仪、高压灭菌器等
    - 医学术语：电导率、伪影、定标、透析液、声学堆栈等
    - 故障术语：分辨率下降、图像模糊、数据断连、漏气测试等
  - **作用**：确保专业术语不被错误切分（如"电导率"不会切成"电/导率"，"血液透析机"不会切成"血液/透析/机"）
  - **初始化**：`BM25SparseEmbedder.__init__()` 中调用 `_init_jieba()`，日志输出 "jieba 分词器已加载 25 个医疗领域术语"
- **状态**：✅ 完成

## 步骤 45：增加元数据过滤（设备类型 + 表达式构建器）
- **时间**：2026-07-20
- **操作**：
  - **新增字段**：Milvus schema 增加 `device_type` VARCHAR(64) 字段，所有 search 方法输出中包含
  - **chunker.py** — 新增 `extract_device_type()` 正则提取（匹配`【设备类型】xxx`），`split_documents()` 中自动提取并存入 metadata（含继承逻辑）
  - **milvus_client.py** — schema 加字段、`insert()` 加 `device_types` 参数、3 个 search 方法的 hit 提取加 `device_type`
  - **新增 `src/retrieval/metadata_filter.py`** — 过滤表达式构建工具：
    - `build_filter_expr(ticket_id, device_type)` → Milvus filter 表达式
    - 多条件自动 AND 拼接，None/空值跳过
    - `KNOWN_DEVICE_TYPES` 列表（10 种设备）供 UI 下拉使用
  - **build_milvus.py** — 传入 `device_types` 列表
  - **rag_chain.py** — `ask()`/`ask_stream()` 新增 `device_type_filter` 参数，用 `build_filter_expr()` 统一构建表达式
  - **api/ask.py** — `AskRequest` 新增 `device_type` 字段，两个端点传递 `device_type_filter`
  - **ui/app.py** — 侧边栏新增"按设备类型过滤"下拉框（10 种设备 + 全部）
  - **⚠️ 注意**：schema 变更需重建 collection `python build_milvus.py --rebuild`
- **状态**：✅ 完成

## 步骤 46：提示词工程增强（Few-shot + Chain-of-Thought + JSON Schema）
- **时间**：2026-07-20
- **操作**：
  - **三项增强**：
    1. **少样本示例（Few-shot）**：在 `rag_prompt_template` 中加入一个完整的带标注问答示例（透析液电导率报警），展示标准回答格式和详细程度，提升输出格式稳定性
    2. **思维链（Chain-of-Thought）**：在 `system_prompt` 中添加 5 步推理框架（拆解现象→列举可能原因→对照工单证据→排除与确认→形成建议），并附带具体推理示例。对于鉴别诊断类问题引导模型先分析症状再给建议
    3. **JSON Schema 约束输出**：新增 `json_prompt_template`，用 JSON Schema 约束模型输出（含字段：question、has_reference、analysis、references、recommendations），附带完整 JSON 示例。API 场景下避免下游解析失败
  - **config.yml** — `system_prompt` 新增【思考方式—思维链】章节；`rag_prompt_template` 新增 Few-shot 示例；新增 `json_prompt_template`；新增 `output_format` 默认配置
  - **config.py** — `LLMConfig` 新增 `output_format`（默认 "text"）、`json_prompt_template`
  - **prompts.py** — 新增 `build_json_prompt()`、`get_json_fallback()`；`build_full_prompt()` 新增 `output_format` 参数按格式分发
  - **rag_chain.py** — `ask()`/`ask_stream()` 新增 `output_format` 参数；无结果兜底区分 text/json
  - **api/ask.py** — `AskRequest`/`AskResponse` 新增 `output_format` 字段（pattern: text|json）
  - **ui/app.py** — 侧边栏新增"输出格式"下拉框（📝 自然语言 / 🔧 JSON 结构化）；JSON 模式自动渲染 `st.json()` 格式化视图
- **面试要点**：
  1. **Few-shot 为什么有效**：大模型对格式指令的遵循度不稳定，但在 Prompt 中给 1-2 个高质量示例后，模型通过 in-context learning 会自动对齐输出格式和详细程度
  2. **Chain-of-Thought 的价值**：医疗运维中很多故障是鉴别诊断问题（如"图像模糊"可能来自探测器、球管、或重建算法）。5 步推理链引导模型逐步分析，比直接给答案更可靠，也便于工程师追溯推理过程
  3. **JSON Schema 的应用场景**：API 场景下下游系统需要结构化数据。在 Prompt 中嵌入 JSON Schema 让模型直接返回可解析的 JSON（含 urgency 紧急程度评估），避免下游用正则拆分自然语言
  4. **为什么用 Prompt 约束而非 function calling**：当前模型的 function calling 支持有限，用 JSON Schema 嵌入 Prompt 的方式更可控、可调试，且不依赖特定模型的 function calling 实现
- **状态**：✅ 完成

## 步骤 47：三级对话记忆系统（Redis 短期 + Kafka 管道 + MySQL 长期 + LLM 中期摘要）
- **时间**：2026-07-20
- **架构**：
  ```
  用户消息 → Redis（短期，毫秒级）
           ↘ Kafka Producer（异步，不阻塞）
               → Kafka Topic → Kafka Consumer（后台线程）
                   → MySQL（长期，永久存储）

  每 5 轮对话 → LLM 摘要 → Redis（中期记忆）+ Kafka → MySQL
  ```
- **操作**：
  - **新增 `src/memory/` 模块**（6 个文件）：
    - `__init__.py` — 模块说明
    - `redis_client.py` — **短期记忆**：List 结构存储最近 N 条消息（默认 20 条），TTL 24h 自动过期；`add_message()` 自动 trim + 续期；`get_messages()`/`get_summary()` 读取；连接失败 → 降级为无记忆模式
    - `mysql_client.py` — **长期记忆**：两张表 `rag_conversations`（消息）+ `rag_summaries`（摘要）；自动建库建表；DictCursor 返回字典；连接池支持
    - `kafka_client.py` — **消息队列**：`KafkaProducerClient` 完全异步发送（不等待 ack）；`KafkaConsumerClient` 后台线程消费写入 MySQL；JSON 序列化
    - `summarizer.py` — **LLM 摘要生成器**：每 5 轮触发一次；DashScope 调用 qwen-max；Prompt 要求提炼设备类型+诊断结论+工单引用+未解决问题；失败返回 None 不阻塞
    - `memory_manager.py` — **三级记忆编排器**：`add_message()` 同时写入 Redis + 异步 Kafka；`get_context()` 返回消息+摘要；`build_memory_prompt()` 格式化为 LLM 可用的上下文
  - **config.yml** — 新增 `memory` 配置段（redis/mysql/kafka/summary 四部分）
  - **config.py** — 新增 `RedisConfig`、`MySQLConfig`、`KafkaConfig`、`SummaryConfig`、`MemoryConfig` 五个配置类
  - **rag_chain.py** — `ask()`/`ask_stream()` 新增 `session_id` 参数；自动保存 user/assistant 消息；Prompt 注入对话历史背景；流式模式包装生成器，流结束后保存完整回答
  - **api/ask.py** — `AskRequest`/`AskResponse` 新增 `session_id` 字段
  - **ui/app.py** — 自动生成 session_id（`uuid4().hex[:12]`）；侧边栏显示会话记忆状态；清空对话时生成新 session_id
- **设计要点**：
  1. **三级记忆为什么这样划分**：短期用 Redis（毫秒读写，适合实时对话）；长期用 MySQL（持久可靠，支持复杂查询）；中期用 LLM 摘要（每 5 轮压缩，既控制 token 消耗又不丢失上下文）
  2. **为什么加 Kafka**：直接写 MySQL 会阻塞聊天响应（网络延迟+事务）。通过 Kafka 解耦：写 Redis → 发 Kafka → 立即返回，后台 consumer 慢慢写 MySQL。Kafka 不可用时不影响核心功能
  3. **优雅降级**：所有外部依赖（Redis/Kafka/MySQL）不可用时只记录 warning 日志，不影响 RAG 问答。`session_id` 不传就是无记忆模式，向后兼容
  4. **摘要触发时机**：选在 `role == "assistant"` 且 `turn % 5 == 0` 时触发，确保新一轮对话完整后才生成摘要
  5. **流式记忆保存**：`ask_stream()` 包装生成器，在所有 chunk 产生后拼接完整回答再写入记忆，避免流式过程中的多次写入
- **状态**：✅ 完成

## 步骤 48：MySQL + Kafka 增加指数退避重试机制
- **时间**：2026-07-20
- **操作**：
  - **config.yml / config.py** — MySQL/Kafka 配置新增 `retry_base_delay`（默认 0.5s）、`retry_max_delay`（默认 5.0s）
  - **mysql_client.py** — 全面重写重试逻辑：
    - 新增 `_retry_with_backoff()` 通用重试包装器（指数退避: 0.5s→1s→2s→4s，上限 5s）
    - 新增 `_is_retryable()` — 区分可重试错误码（2002/2003/2006/2013/1205/1213）vs 不可重试错误
    - 新增 `_reset_connection()` — 写操作失败时强制关闭旧连接 + 重建
    - 新增 `_sleep_backoff()` — 指数退避等待
    - 写操作（`insert_message`/`insert_summary`/`init_tables`）：最多 3 次重试，死连接自动重连
    - 读操作（`get_history`/`get_summaries`）：最多 2 次轻量重试，不重连
    - `_get_conn()` 增加 `ping(reconnect=False)` 检测连接存活
    - 连接增加 `connect_timeout=5`、`read_timeout=10`、`write_timeout=10`
  - **kafka_client.py** — Producer 重试增强：
    - 新增 `_ensure_producer()` — producer 失效时指数退避重连（最多 3 次）
    - 新增 `_send_with_retry()` — 统一发送重试逻辑：
      - 先确保 producer 可用 → 再发送
      - `KafkaError.retriable` → 区分可重试 vs 不可重试
      - 可重试错误：关闭旧 producer → 退避 → 重连 → 重发
      - 不可重试错误：立即返回 False
      - 发送改用 `future.get(timeout=10)` 同步等待结果，确保捕获异常
    - Consumer 消费循环增加连续异常计数（连续 5 次异常自动退出）+ 指数退避
    - Consumer `_process()` 中 MySQL 写入失败不再崩溃（内部已有重试），仅 warning 日志
  - **设计要点**：
    1. **指数退避 vs 固定延迟**：指数退避（0.5→1→2→4→上限 5s）在快速恢复和避免雪崩之间取得平衡
    2. **可重试错误码分类**：2002/2003（连接错误）、2006/2013（断连）、1205/1213（锁超时/死锁）适合重试；语法错误/约束冲突等应立即失败
    3. **读写分离**：写操作重试时先重连再试（可能是连接过期）；读操作不重连（数据本身不存在不应重试）
    4. **Kafka 同步等待**：从完全异步（`future.add_callback`）改为 `future.get(timeout=10)`，确保在重试循环中能感知发送失败
- **状态**：✅ 完成

## 步骤 49：语义缓存 — 相同/相似问题跳过 LLM
- **时间**：2026-07-20
- **原理**：
  1. 用户问题 → DashScope embedding 向量（1536 维）
  2. 与 Redis 中缓存的所有问题向量做**余弦相似度**比较
  3. 相似度 ≥ 0.95 → 命中，直接返回缓存答案（跳过检索+LLM）
  4. 相似度 < 0.95 → 未命中，正常 RAG → 缓存新 Q&A
  - Redis 存储：`rag:cache:qa` → List\<JSON\> `{q, a, embedding, ts}`
  - LRU 淘汰：超过 `max_entries`(100) 时从左侧丢弃旧条目
  - TTL：默认 1 小时
- **操作**：
  - **config.yml / config.py** — 新增 `CacheConfig`（enabled/threshold/max_entries/ttl）
  - **新增 `src/memory/qa_cache.py`** — 语义缓存核心：
    - `lookup(question)` → 余弦相似度匹配，返回 `(hit, answer)`
    - `store(question, answer)` → LRU + TTL 管理
    - `_embed(text)` → 懒加载 EmbeddingClient
    - `_cosine(a, b)` → 余弦相似度计算
  - **rag_chain.py** — `ask()`/`ask_stream()` 在检索前调用 `_qa_cache.lookup()`，命中直接返回（`from_cache=True`）；回答后调用 `_qa_cache.store()`
  - **api/ask.py** — `AskResponse` 新增 `from_cache: bool` 字段
- **验证结果**：同一问题第 1 次 24014ms（LLM），第 2 次 2721ms（缓存命中），9x 加速，答案完全一致
- **Bug 修复**：`store()` 用 key `"a"` 但 `lookup()` 读 `"answer"` → 字段名不匹配导致缓存永远不命中，已修正
- **状态**：✅ 完成

---

## 步骤 50：限流与配额管理（Token Bucket + LLM 成本追踪）
- **时间**：2026-07-20
- **背景**：
  - 系统对外暴露 API 后存在以下风险：
    1. 无 QPS 限制 —— 恶意/意外高频调用可打垮服务，LLM API 费用不可控
    2. 无用户维度隔离 —— 所有请求一视同仁，无法区分高/低优先级用户
    3. 无 LLM 预算管控 —— 单用户可以无限调用 LLM，费用无上限
- **方案**：三层限流架构

  **① 全局限流 — Token Bucket (Redis Lua + 内存兜底)**
  - 算法：Token Bucket（capacity=burst, refill_rate=rpm/60）
  - Redis 模式：Lua 脚本保证 `HGET → refill → 判断 → HMSET` 原子性，适合多进程部署
  - 内存模式：`threading.Lock` + time-based refill，Redis 不可用时自动降级
  - 默认：60 req/min，burst=15

  **② 用户级限流 — 同算法，独立桶**
  - 用户标识提取优先级：`X-API-Key` > `X-User-ID` > `session_id` > 客户端 IP
  - 默认：20 req/min，burst=5
  - 每个用户独立 Token Bucket，互不影响

  **③ LLM 成本控制 — Sliding Window (Redis Sorted Set)**
  - 按小时窗口统计每用户的 token 消耗 + 请求次数
  - 两维度预算：max_tokens_per_hour（默认 100K）+ max_requests_per_hour（默认 50）
  - `ZADD` 记录每次 LLM 调用 → `ZREMRANGEBYSCORE` 清理过期数据 → 超限抛 `BudgetExceeded`

  **④ 用户 ID 传播 — contextvars**
  - 问题：API 层知道 user_id，LLMClient 需要 user_id 上报成本，但不想改函数签名
  - 方案：`contextvars.ContextVar` 在 API 入口 `set_current_user()`，LLMClient 内部 `get_current_user()`
  - 优势：不侵入 RAGChain → LLMClient → prompts 的调用链，协程安全

- **操作**：
  - **config.yml** — 新增 `rate_limit` 配置节：
    ```yaml
    rate_limit:
      enabled: true
      global_rpm: 60         # 全局 QPS ≈ 1
      global_burst: 15       # 允许短时突发
      user_rpm: 20           # 单用户 QPS
      user_burst: 5
      llm_max_tokens_per_hour: 100000
      llm_max_requests_per_hour: 50
      redis_enabled: true    # 分布式后端开关
    ```
  - **config.py** — 新增 `RateLimitConfig` 数据类
  - **新增 `src/utils/rate_limiter.py`** — 限流核心（~430 行）：
    - `InMemoryTokenBucket` — `threading.Lock` + time-based refill
    - `RedisTokenBucket` — Lua 脚本原子 Token Bucket
    - `RateLimiter` — 统一接口，Redis→内存自动 fallback，`check_global()` + `check_user()`
    - `CostTracker` — Redis Sorted Set 滑动窗口，`check_budget()` + `record_usage()`
    - `get_user_id(request)` — 用户标识提取（API-Key > User-ID > session > IP）
    - `set_current_user()` / `get_current_user()` — contextvars 传播
  - **main.py** — 注册异常处理：
    - `RateLimitExceeded` → 429 + `Retry-After` + `X-RateLimit-*` 头
    - `BudgetExceeded` → 429 + `Retry-After`
  - **api/ask.py** — `rag_ask` / `rag_ask_stream` 注入限流检查：
    - 提取 user_id → 全局限流 → 用户限流 → LLM 预算检查 → `set_current_user()`
    - 响应携带 `X-RateLimit-*` 头（Limit/Remaining/Reset，全局+用户各一套）
  - **llm_client.py** — `_call_api()` 成功后调用 `_report_cost()`，自动上报 token 用量
- **验证结果**：
  - 所有 32 个已有测试通过，无回归
  - 内存模式限流验证：burst=15 → 前 15 次放行，第 16 次触发 `RateLimitExceeded`
  - Redis 不可用时自动打印一条 Warning，退回内存模式
  - 成本上报失败不影响主流程（全 try-catch）
- **状态**：✅ 完成

---

## 步骤 51：Streamlit 前端界面重构优化
- **时间**：2026-07-21
- **背景**：
  - 旧版前端仅有基础 CSS 微调，体验接近"开发原型"
  - 缺乏欢迎引导页，新用户不知道从何下手
  - 检索/生成状态提示简陋（纯文本 caption）
  - 元信息和来源卡片排版粗糙，信息密度低
  - 侧边栏控件杂乱堆叠，缺乏分组层次
- **方案**：全面重构 Streamlit UI，保持功能完整，提升视觉品质

  **① 欢迎引导页**
  - 空会话时展示欢迎页：标题 + 能力卡片（4 列网格：混合检索/CoT 推理/会话记忆/语义缓存）
  - 8 个快捷问题以按钮网格布局（2 行 × 4 列），替代侧边栏的垂直按钮列表
  - 减少首次使用的认知负担

  **② 状态指示器**
  - 三阶段视觉反馈：
    - 检索中 → 蓝色脉冲动画 + `status-pulse` CSS
    - 生成中 → 黄色脉冲动画
    - 缓存命中 → 绿色常亮
  - 用 CSS `@keyframes pulse` 实现呼吸灯效果

  **③ 元信息标签栏**
  - 彩色圆角标签（pill badge）展示关键指标：
    - 命中/未命中/缓存命中（绿/红/蓝）
    - 检索模式（灰）
    - 延迟自动分级：绿 <1s / 黄 1-3s / 红 >3s
  - CSS `flex-wrap` 自适应换行

  **④ 来源卡片重构**
  - 卡片式设计：工单编号 + 分数（颜色渐变） + 进度条 + 文本块（最大高度限制） + 来源文件
  - 来源 ≤3 条默认展开，>3 条默认折叠

  **⑤ 侧边栏分层**
  - 三组 Expander：基本设置（始终展开）/ 高级选项 / 过滤条件
  - 新增"新建会话"按钮
  - 系统状态新增限流状态展示

  **⑥ CSS 设计系统**
  - CSS 变量（颜色/圆角/阴影）、动画（pulse/blink）、hover 效果、响应式

- **操作**：
  - 重写 `src/ui/app.py`
  - 新增组件函数：`render_welcome()`、`render_source_card()`、`render_meta_tags()`、`_latency_class()`、`_score_color()`
  - 状态指示器：蓝/黄/绿三阶段
  - 历史消息新增 `from_cache` 和 `output_format` 字段
- **验证**：语法检查通过，32 个已有测试全部通过
- **状态**：✅ 完成