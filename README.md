# RAG 智能工单问答系统

基于 **Milvus + 千问大模型** 的医疗设备运维工单检索增强生成（RAG）系统。

---

## 项目概述

面向医院设备科的智能运维助手。通过向量检索从历史工单知识库中查找相关案例，结合千问大模型生成包含故障分析、参考案例、处理步骤和安全提醒的专业回答。

### 核心能力

| 能力 | 说明 |
|------|------|
| **语义检索** | 稠密向量（1536 维），理解同义词和上下文 |
| **关键字检索** | BM25 稀疏向量 + 倒排索引，精确匹配术语和型号 |
| **混合检索** | 语义 + 关键字加权重排，互补盲区 |
| **RAG 生成** | 检索→Prompt 拼接→LLM 生成，回答带工单引用来源 |
| **安全兜底** | 医疗专家 system prompt + 安全守则 + 无结果回退 |

### 技术栈

```
检索:  Milvus 2.x (IVF_FLAT + SPARSE_INVERTED_INDEX)
嵌入:  DashScope text-embedding-v1 (1536d)
分词:  jieba + BM25 (pymilvus.model.sparse)
LLM:   DashScope Qwen-Max
框架:  FastAPI + LangChain + pymilvus 3.0
```

---

## 项目结构

```
智能工单/
├── config.yml                 # 应用配置（Milvus/Embedding/LLM/Retrieval/Chunking）
├── .env                       # 环境变量（API Key 等，不入 git）
├── docker-compose.yml         # Docker 编排（etcd + minio + milvus + attu）
├── Dockerfile                 # 应用容器
├── requirements.txt           # Python 依赖
├── build_milvus.py            # ETL 构建脚本（加载→分块→向量化→入库）
│
├── data/
│   └── documents/
│       └── 工单知识库.txt      # 10 张模拟医疗设备运维工单
│
├── src/
│   ├── config.py              # 配置加载器（.env + config.yml → 类型化对象）
│   ├── main.py                # FastAPI 应用入口
│   ├── cli.py                 # 命令行工具（ingest / preview / stats / ask）
│   ├── interactive.py         # 交互式问答终端（启动即用）
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
│   │   └── milvus_client.py   #   Milvus 操作封装（增删查 + 双路检索）
│   │
│   ├── llm/                   # 大模型
│   │   ├── llm_client.py      #   千问 API 封装（重试 + 退避）
│   │   ├── prompts.py         #   提示词管理（格式化 + 模板填充）
│   │   └── rag_chain.py       #   RAG 核心链（检索→Prompt→生成）
│   │
│   └── api/                   # REST API
│       └── search.py          #   检索接口（semantic/keyword/hybrid）
│
└── tests/                     # 测试（43 tests, 0 failed）
    ├── test_config.py
    ├── test_chunker.py
    ├── test_loader.py
    ├── test_pipeline.py
    ├── test_milvus.py
    ├── test_milvus_manual.py
    └── test_embed_manual.py
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

### 3. 启动 Milvus

```bash
docker compose -p rag-ticket up -d
```

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
# 方式 1：交互式终端（推荐）
python -m src.interactive

# 方式 2：API 服务
uvicorn src.main:app --port 8000
# 打开 http://localhost:8000/docs 查看 Swagger

# 方式 3：命令行单条问答
python -m src.cli ask -q "CT伪影怎么处理？" -m hybrid
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

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/search/semantic` | POST | 语义检索 |
| `/api/search/keyword` | POST | 关键字检索 |
| `/api/search/hybrid` | POST | 混合检索 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger 文档 |

请求示例：

```json
POST /api/search/hybrid
{
  "query": "CT扫描图像伪影",
  "top_k": 5,
  "dense_weight": 0.5,
  "sparse_weight": 0.5
}
```

---

## RAG 问答流程

```
用户问题: "CT扫描图像伪影怎么排查？"
    │
    ▼
① 向量检索 (Milvus)
    ├─ 稠密向量 → IVF_FLAT → 语义匹配
    └─ 稀疏向量 → 倒排索引 → 关键字匹配
    │
    ▼
② 质量过滤
    丢弃 score < threshold 的低相关 chunk
    │
    ▼
③ Prompt 构建
    system: "你是医疗设备运维专家助手...（安全守则）"
    user:   【参考工单】chunk1, chunk2... 【用户问题】CT伪影
    │
    ▼
④ LLM 生成 (Qwen-Max)
    按格式输出: 故障分析 → 参考案例 → 处理步骤 → 注意事项
    │
    ▼
⑤ 返回
    { "answer": "...", "sources": [{ticket_id, score, content}, ...] }
```

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
# 43 tests passed, 0 failed
```

覆盖：配置加载、文档分块、工单编号提取、向量化、Milvus 增删查、标量过滤、语义/关键字/混合检索。

---

## 配置参考

```yaml
# config.yml 关键配置项
retrieval:
  default_mode: hybrid        # 默认检索模式
  hybrid:
    dense_weight: 0.5         # 语义权重（0~1，越大越偏语义）
    sparse_weight: 0.5        # 关键字权重（0~1，越大越偏精确匹配）

llm:
  model: qwen-max
  temperature: 0.1            # 创造性（低=稳定，高=多样）
  system_prompt: |            # 人设 + 安全守则
  rag_prompt_template: |      # {context} + {question} 拼接模板
  fallback_answer: "..."      # 无结果兜底回答
```
