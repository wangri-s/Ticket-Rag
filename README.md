# RAG 智能工单问答系统

基于 **Milvus + 千问大模型** 的医疗设备运维工单检索增强生成（RAG）系统。

---

## 项目概述

面向医院设备科的智能运维助手。通过向量检索从历史工单知识库中查找相关案例，结合千问大模型生成包含故障分析、参考案例、处理步骤和安全提醒的专业回答。

### 核心能力

| 能力 | 说明 |
|------|------|
| **语义检索** | 稠密向量（1536 维），理解同义词和上下文 |
| **关键字检索** | BM25 稀疏向量 + jieba 医疗分词 + 倒排索引，精确匹配术语和型号 |
| **混合检索** | 语义 + 关键字加权重排，互补盲区 |
| **重排序** | qwen3-rerank 交叉编码器精排，初检多召回→重排→取 Top-K |
| **查询扩展** | LLM 口语→关键词改写（"片子不清楚"→"CT 图像伪影 分辨率下降"） |
| **元数据过滤** | 按设备类型（10 种）+ 工单号过滤，Milvus 标量索引加速 |
| **RAG 生成** | 检索→Prompt 拼接→LLM 生成，回答带工单引用来源 |
| **流式输出** | LLM 逐 token 返回，前端打字机效果 |
| **提示词工程** | Few-shot 示例 + Chain-of-Thought 推理 + JSON Schema 结构化输出 |
| **三级对话记忆** | Redis 短期 + Kafka 管道 + MySQL 长期 + LLM 中期摘要（每 5 轮） |
| **语义缓存** | 相同/相似问题 embedding 余弦匹配，跳过 LLM（9x 加速） |
| **限流配额** | 全局限流 + 用户限流（Token Bucket）+ LLM 成本控制（Sliding Window） |
| **安全兜底** | 医疗专家 System Prompt + 安全守则 + 无结果回退 |
| **优雅降级** | Redis/Kafka/MySQL/限流 任一不可用自动降级，不影响核心问答 |

### 技术栈

```
检索:     Milvus 2.x (IVF_FLAT + SPARSE_INVERTED_INDEX)
嵌入:     DashScope text-embedding-v1 (1536d)
分词:     jieba + 27 个医疗领域自定义术语
LLM:      DashScope Qwen-Max + qwen3-rerank
记忆:     Redis 7.x + Kafka + MySQL 8.0
限流:     Token Bucket + Sliding Window（Redis Lua + 内存兜底）
框架:     FastAPI + LangChain + pymilvus 3.0
前端:     Streamlit（对话式 UI，流式输出 + 欢迎引导页）
```

---

## 项目结构

```
智能工单/
├── config.yml                 # 应用配置（Milvus / Embedding / LLM / Retrieval / Chunking）
├── .env                       # 环境变量（API Key 等，不入 git）
├── docker-compose.yml         # Docker 编排（etcd + minio + milvus + attu + rag-app）
├── Dockerfile                 # 应用容器
├── requirements.txt           # Python 依赖
├── README.md                  # 本文件
├── build_milvus.py            # ETL 构建脚本（加载→分块→向量化→入库）
│
├── data/
│   └── documents/
│       └── 工单知识库.txt      # 10 张模拟医疗设备运维工单
│
├── src/
│   ├── config.py              # 配置加载器（.env + config.yml → 类型化对象）
│   ├── main.py                # FastAPI 应用入口（CORS + 日志 + 异常处理）
│   ├── cli.py                 # 命令行工具（ingest / preview / stats / ask + 交互模式）
│   │
│   ├── ingestion/             # 文档摄入
│   │   ├── loader.py          #   文档加载器（TXT/PDF）
│   │   ├── chunker.py         #   工单文本分割器（7 级分隔符）
│   │   └── pipeline.py        #   摄入流水线
│   │
│   ├── embedding/             # 向量化
│   │   ├── embedding_client.py  #  稠密向量（DashScope）
│   │   └── sparse_embedder.py   #  稀疏向量（BM25 + jieba）
│   │
│   ├── retrieval/             # 检索引擎
│   │   ├── milvus_client.py   #   Milvus 操作封装（增删查 + 双路检索 + 标量过滤）
│   │   ├── reranker.py        #   重排序（qwen3-rerank 交叉编码器）
│   │   ├── query_processor.py #   查询预处理（口语→关键词改写）
│   │   └── metadata_filter.py #   元数据过滤表达式构建器
│   │
│   ├── memory/                # 三级对话记忆
│   │   ├── memory_manager.py  #   编排层（Redis + Kafka + MySQL + LLM摘要）
│   │   ├── redis_client.py    #   短期记忆（List 存储，TTL 24h）
│   │   ├── mysql_client.py    #   长期记忆（持久化 + 指数退避重试）
│   │   ├── kafka_client.py    #   消息队列（Producer/Consumer + 重试）
│   │   ├── summarizer.py      #   中期摘要（每 5 轮 LLM 压缩）
│   │   └── qa_cache.py        #   语义缓存（embedding 余弦匹配）
│   │
│   ├── llm/                   # 大模型
│   │   ├── llm_client.py      #   千问 API 封装（重试 + 退避 + 流式）
│   │   ├── prompts.py         #   提示词管理（Few-shot + CoT + JSON Schema）
│   │   └── rag_chain.py       #   RAG 核心链（检索→Prompt→生成 + 流式 + 记忆 + 缓存）
│   │
│   ├── api/                   # REST API
│   │   ├── search.py          #   检索接口（semantic / keyword / hybrid）
│   │   └── ask.py             #   RAG 问答接口（/api/ask + /api/ask/stream）
│   │
│   ├── utils/                  # 工具
│   │   └── rate_limiter.py     #   限流配额（Token Bucket + Sliding Window + LLM成本）
│   │
│   └── ui/                    # 前端
│       └── app.py             #   Streamlit 对话界面（欢迎页 + 流式 + 状态指示器）
│
└── tests/                     # 测试
    ├── test_config.py
    ├── test_chunker.py
    ├── test_loader.py
    ├── test_pipeline.py
    ├── test_milvus.py
    ├── test_milvus_manual.py
    ├── test_embed_manual.py
    └── test_rag.py            # RAG 全链路集成测试
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/wangri-s/Ticket-Rag.git
cd Ticket-Rag

# 创建虚拟环境
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制环境变量模板，填入 DashScope API Key
cp .env.example .env
# 编辑 .env: DASHSCOPE_API_KEY=sk-your-key
```

`config.yml` 包含全部应用参数，可按需调整检索模式、模型温度、Prompt 模板等。

### 3. 启动基础设施

```bash
# Milvus（向量数据库，必需）
docker compose -p rag up -d etcd minio milvus-standalone

# Redis（对话记忆 + 语义缓存 + 分布式限流）
docker run -d --name redis-rag -p 6379:6379 redis:7-alpine

# Zookeeper + Kafka（异步记忆管道，可选）
docker start zookeeper kafka
```

> Redis/Kafka 不可用时自动降级，不影响核心问答。

验证服务状态：

```bash
docker ps --filter "name=milvus"
# milvus-standalone  healthy  :19530
# milvus-attu        running  :3000  (GUI 管理工具)
# milvus-etcd        healthy
# milvus-minio       healthy
```

### 4. 构建知识库

```bash
python build_milvus.py --rebuild
```

输出：

```
[1/5] 加载文档    1 文档, 7118 字符
[2/5] 文本分块    22 块 (avg=322 字符)
[3/5] BM25 训练   22 个稀疏向量
[4/5] 稠密向量化  22 条 × 1536 维
[5/5] 写入 Milvus 22 条
[DONE] 总耗时 5.4s
```

### 5. 开始使用

三种方式任选：

```bash
# 方式 1：Web 对话界面（推荐）
streamlit run src/ui/app.py
# 打开 http://localhost:8501

# 方式 2：API 服务
python -m src.main
# 打开 http://localhost:8000/docs 查看 Swagger

# 方式 3：命令行
python -m src.cli ask -q "CT伪影怎么处理？" -m hybrid     # 单次问答
python -m src.cli ask                                      # 交互模式
```

---

## 检索模式

| 模式 | 原理 | 适用场景 |
|------|------|---------|
| `semantic` | 1536 维稠密向量 + IVF_FLAT | 模糊描述，语义相近 |
| `keyword` | BM25 稀疏向量 + 倒排索引 | 精确术语、工单号、设备型号 |
| `hybrid` | 双路并行 + WeightedRanker 加权重排 | 默认推荐，兼顾语义和精确 |

默认权重 0.5:0.5，可在 `config.yml` 调整。

---

## API 接口

### 检索（不调 LLM）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/search/semantic` | POST | 语义检索 |
| `/api/search/keyword` | POST | 关键字检索 |
| `/api/search/hybrid` | POST | 混合检索 |

### RAG 问答（检索 + LLM 生成）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/ask` | POST | RAG 全链路问答 |
| `/api/ask/stream` | POST | RAG 流式问答（SSE） |

### 运维

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/` | GET | API 导航页 |
| `/docs` | GET | Swagger 交互文档 |
| `/redoc` | GET | ReDoc 文档 |

请求示例：

```json
POST /api/ask
{
  "question": "CT扫描图像伪影是什么原因？",
  "mode": "hybrid",
  "top_k": 5,
  "ticket_id": null,
  "device_type": "CT 机",
  "rerank": false,
  "query_expansion": false,
  "output_format": "text",
  "session_id": "my-session-001"
}
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `question` | str | 必填 | 用户问题（1-2000字符） |
| `mode` | str | `hybrid` | 检索模式：`semantic` / `keyword` / `hybrid` |
| `top_k` | int | 5 | 检索条数（1-20） |
| `ticket_id` | str | null | 按工单号过滤（如 `GD-2026-03001`） |
| `device_type` | str | null | 按设备类型过滤（10 种：CT 机、MRI 核磁共振等） |
| `rerank` | bool | false | 启用 qwen3-rerank 交叉编码器重排序 |
| `query_expansion` | bool | false | 启用 LLM 口语→关键词改写 |
| `output_format` | str | `text` | `text`（Few-shot+CoT）/ `json`（JSON Schema） |
| `session_id` | str | null | 会话 ID，传入则启用三级对话记忆 |

响应：

```json
{
  "question": "CT扫描图像伪影是什么原因？",
  "answer": "### 1. 故障分析\n根据工单记录，CT扫描图像伪影...",
  "sources": [
    {
      "ticket_id": "GD-2026-03001",
      "content": "...",
      "score": 0.9786,
      "source": "工单知识库.txt"
    }
  ],
  "mode": "hybrid",
  "has_answer": true,
  "output_format": "text",
  "session_id": "my-session-001",
  "from_cache": false,
  "latency_ms": 2340.5
}
```

| 响应字段 | 类型 | 说明 |
|----------|------|------|
| `answer` | str | LLM 生成的回答（json 模式为 JSON 字符串） |
| `sources` | list | 引用的工单来源 |
| `has_answer` | bool | 是否找到相关工单 |
| `output_format` | str | 实际输出格式 |
| `session_id` | str\|null | 会话 ID |
| `from_cache` | bool | 是否来自语义缓存（跳过 LLM） |
| `latency_ms` | float | 总耗时（毫秒） |

---

## RAG 问答流程

```
用户问题: "CT扫描图像伪影怎么排查？"  (session_id="abc123")
    │
    ▼
0a. 限流检查
     全局限流 (Token Bucket) → 用户限流 (Token Bucket) → LLM 预算 (Sliding Window)
    │
    ▼
0b. 语义缓存查找
     embedding(question) → Redis 余弦相似度 ≥0.95? → 命中 → 直接返回（跳过 LLM）
    │
    ▼
0c. 查询预处理 (可选)
     LLM 口语改写: "片子不清楚" → "CT 图像伪影 分辨率下降"
    │
    ▼
0c. 注入对话记忆 (有 session_id 时)
     Redis 读取最近 20 条消息 + 中期摘要 → 拼入 Prompt 的【对话背景】段
    │
    ▼
① 向量检索 (Milvus)
    ├─ 稠密向量 → IVF_FLAT → 语义匹配
    └─ 稀疏向量 → 倒排索引 → 关键字匹配
    ├─ 标量过滤: device_type == "CT 机"  (可选)
    │
    ▼
② 重排序 (可选)
    初检 top_k × 3 → qwen3-rerank 交叉编码器精排 → top_k
    │
    ▼
③ 质量过滤
    丢弃 score < threshold 的低相关 chunk
    │
    ▼
④ Prompt 构建
    system: "你是医疗设备运维专家...（安全守则 + CoT 推理链）"
    user:   【对话背景】... 【参考工单】... 【用户问题】... (含 Few-shot 示例)
    │
    ▼
⑤ LLM 生成 (Qwen-Max)
    ├─ 流式模式: 逐 token 返回（打字机效果）
    ├─ 格式: 故障分析 → 参考案例 → 处理步骤 → 注意事项
    └─ 或: JSON Schema 结构化输出
    │
    ▼
⑥ 记忆保存
    ├─ Redis: LPUSH 消息（短期）
    ├─ Kafka/MySQL: 持久化（长期）
    └─ 每 5 轮: LLM 生成中期摘要
    │
    ▼
⑦ 语义缓存
    新 Q&A → Redis (embedding + answer)，下次相同问题 9x 加速
    │
    ▼
⑧ 返回
    { "answer": "...", "sources": [...], "from_cache": false, "session_id": "..." }

---

## 前端界面

Streamlit 对话式 UI，支持：

- 🏠 **欢迎引导页**：首次打开展示能力卡片 + 快捷问题网格，降低使用门槛
- 💬 **对话式交互**：消息历史累积，流式打字机效果 + 闪烁光标
- 🔵🟡🟢 **三阶段状态指示器**：蓝色检索中 → 黄色生成中 → 绿色缓存命中，CSS 脉冲动画
- 🏷️ **元信息标签栏**：彩色圆角标签（命中/缓存/模式/延迟分级着色）
- 📚 **来源卡片**：工单编号 + 分数颜色渐变 + 可视进度条 + 内容预览
- ⚙️ **侧边栏分层**：基本设置 / 高级选项 / 过滤条件 三组折叠
- 🔄 **新建会话**：一键切换 session_id，不丢失历史

```bash
streamlit run src/ui/app.py
# → http://localhost:8501
```

---

## 三级对话记忆

```
用户消息 → Redis (短期, 2ms) → Kafka/MySQL (长期, 异步)
                                    ↑
                        每 5 轮 → LLM 摘要 (中期)
```

| 层级 | 存储 | 容量 | 生命周期 | 用途 |
|------|------|------|----------|------|
| 短期 | Redis List | 20 条/会话 | TTL 24h | 实时对话上下文注入 Prompt |
| 中期 | LLM 摘要 | 每 5 轮 1 条 | Redis TTL 24h + MySQL 永久 | 压缩历史，控制 token 消耗 |
| 长期 | MySQL | 不限 | 永久 | 历史回溯、数据分析 |

**使用方式**：API 传 `session_id` 即启用，不传即无记忆模式。

**降级策略**：Redis 不可用 → 跳过短期记忆；Kafka 不可用 → 直接写 MySQL；MySQL 不可用 → 只保留 Redis。**任一故障不影响核心问答**。

---

## 语义缓存

相同或高度相似的问题（余弦相似度 ≥ 0.95），直接返回缓存答案，跳过检索和 LLM 调用。

```
第 1 次: embedding("高压灭菌器灭菌程序中断...") → 缓存为空 → 完整 RAG → 存缓存 (~20s)
第 2 次: embedding("高压灭菌器灭菌程序中断...") → 余弦相似度 1.00 → 命中！ → 直接返回 (~2s)
```

**加速比**：9x（2721ms vs 24014ms实测）

---

## 限流与配额管理

三层防护架构，防止 API 滥用和 LLM 费用失控：

```
请求 → 全局限流 (Token Bucket) → 用户限流 (Token Bucket) → LLM 预算 (Sliding Window) → 放行
                ↓ 超限                  ↓ 超限                   ↓ 超限
              429 + Retry-After      429 + Retry-After       429 + 预算提示
```

| 层级 | 算法 | 默认值 | 后端 |
|------|------|--------|------|
| 全局限流 | Token Bucket | 60 req/min, burst=15 | Redis Lua → 内存兜底 |
| 用户限流 | Token Bucket | 20 req/min, burst=5 | Redis Lua → 内存兜底 |
| LLM 成本 | Sliding Window | 100K tokens/h, 50 次/h | Redis Sorted Set |

**用户识别**：`X-API-Key` > `X-User-ID` > `session_id` > 客户端 IP

**响应头**：每次请求返回 `X-RateLimit-Limit/Remaining/Reset`，超限时附带 `Retry-After`

**容错**：Redis 不可用时自动退回进程内存模式，成本上报失败不影响主流程

---

## 提示词工程

| 技术 | 位置 | 效果 |
|------|------|------|
| **Few-shot** | `rag_prompt_template` 末尾 | 1 个完整标注示例，稳定输出格式 |
| **Chain-of-Thought** | `system_prompt` 思维链章节 | 5 步推理框架（拆解→列举→对照→排除→形成） |
| **JSON Schema** | `json_prompt_template` | `output_format=json` 时强制结构化输出，含 `urgency` 评估 |

---

## 安全设计

System Prompt 内置 4 条安全守则：

1. 回答仅供参考，不能替代专业工程师现场诊断
2. 高风险操作必须标注"仅限授权人员操作"
3. 知识库无相关案例时诚实告知，不编造
4. 不越界给出治疗建议或临床诊断

无检索结果时不调 LLM，直接返回预设兜底回答，避免模型在没有依据的情况下生成不实信息。

---

## 测试

```bash
pytest tests/ -v
```

覆盖：配置加载、文档分块、工单编号提取、向量化、Milvus 增删查、标量过滤、三种检索模式、RAG 全链路集成、边界情况（空输入、无效模式、兜底、过滤）。

---

## 配置参考

```yaml
# config.yml 关键配置项

retrieval:
  default_mode: hybrid        # 默认检索模式
  hybrid:
    dense_weight: 0.5         # 语义权重（0~1）
    sparse_weight: 0.5        # 关键字权重（0~1）
  rerank:
    enabled: false            # 启用 qwen3-rerank 重排序
    oversample_factor: 3      # 初检倍数
  query_preprocess:
    enabled: false            # 启用 LLM 查询改写

llm:
  model: qwen-max
  temperature: 0.1
  output_format: text         # text | json
  system_prompt: |            # 人设 + CoT 推理链 + 安全守则
  rag_prompt_template: |      # {context} + {question} + Few-shot 示例
  json_prompt_template: |     # JSON Schema 约束输出模板

memory:
  redis:                      # 短期记忆
    ttl: 86400                # 24h 过期
    max_messages: 20          # 每会话最多保留
  mysql:                      # 长期记忆
    max_retries: 3            # 指数退避重试
  kafka:                      # 异步管道（不可用时直接写 MySQL）
  summary:
    trigger_turns: 5          # 每 5 轮生成中期摘要

cache:
  enabled: true               # 语义缓存
  similarity_threshold: 0.95  # 余弦相似度阈值
  max_entries: 100            # 最大缓存条目（LRU 淘汰）
  ttl: 3600                   # 1 小时过期

rate_limit:
  enabled: true               # 限流总开关
  global_rpm: 60              # 全局每分钟最大请求数
  global_burst: 15            # 全局突发容量
  user_rpm: 20                # 单用户每分钟最大请求数
  user_burst: 5               # 单用户突发容量
  llm_max_tokens_per_hour: 100000   # 单用户每小时 LLM 最大 token
  llm_max_requests_per_hour: 50     # 单用户每小时最大 LLM 调用次数
  redis_enabled: true         # Redis 分布式后端开关
```
