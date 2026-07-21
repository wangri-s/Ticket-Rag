# 智能工单 RAG 系统 — 功能全景与场景化设计

> 一份文档看清：核心功能是什么、边界场景怎么处理、日志链路怎么追踪。

---

## 一、项目定位

面向医院设备科的**医疗工单智能问答系统**。工程师用自然语言描述故障现象，系统从历史工单库中检索相关案例，由 LLM 生成包含故障分析、参考案例、处理步骤和安全提醒的专业回答。

**一句话**：把沉淀在工单里的"老师傅经验"变成可检索、可生成的知识服务。

---

## 二、核心功能矩阵

| 功能 | 一句话 | 关键文件 | 技术实现 |
|------|--------|---------|---------|
| **文档摄入** | 工单 TXT → 智能分块 | `src/ingestion/` | 7级分隔符递归分割 + ticket_id继承 |
| **稠密向量** | 语义级相似度 | `src/embedding/embedding_client.py` | DashScope text-embedding-v1 (1536d) |
| **稀疏向量** | 关键词级精确匹配 | `src/embedding/sparse_embedder.py` | BM25 + jieba + 27个医疗术语 |
| **三模式检索** | semantic / keyword / hybrid | `src/retrieval/milvus_client.py` | IVF_FLAT + SPARSE_INVERTED_INDEX |
| **混合融合** | 语义+关键词加权 | 同上 | Milvus WeightedRanker |
| **重排序** | 粗召回 → 交叉编码精排 | `src/retrieval/reranker.py` | qwen3-rerank 两阶段 |
| **查询扩展** | 口语→专业关键词 | `src/retrieval/query_processor.py` | LLM Few-shot query2keywords |
| **元数据过滤** | 按设备/工单号筛选 | `src/retrieval/metadata_filter.py` | Milvus 标量索引 |
| **Prompt 工程** | Few-shot + CoT + 双格式 | `src/llm/prompts.py` | System Prompt + JSON Schema |
| **LLM 生成** | 检索→Prompt→生成 | `src/llm/llm_client.py` | Qwen-Max + 重试+退避 |
| **流式输出** | SSE 逐 token 返回 | `src/api/ask.py` | FastAPI StreamingResponse |
| **三级记忆** | 短期+中期+长期 | `src/memory/` | Redis → Kafka → MySQL + LLM摘要 |
| **语义缓存** | 相似问题跳过LLM | `src/memory/qa_cache.py` | embedding余弦匹配 + Redis LRU |
| **上下文压缩** | Prompt 不超 context window | `src/utils/context_compressor.py` | 四级递进策略 |
| **限流配额** | QPS + 用户 + LLM成本 | `src/utils/rate_limiter.py` | Token Bucket + Sliding Window |
| **安全兜底** | 无结果不调LLM | `src/llm/prompts.py` | 预设 fallback |

---

## 三、场景化设计

### 3.1 长上下文 — Prompt 超窗口怎么办？

**问题**：检索返回 10 条 chunk + System Prompt（~500 tokens）+ Few-shot 示例（~400 tokens）+ 对话记忆（~300 tokens），总 token 数可能超出 Qwen-Max 的 8K context window。

**方案**：`ContextCompressor` 四级递进压缩

```
┌─────────────────────────────────────────────────────────┐
│  检索结果 (10 chunks) + System Prompt + 记忆上下文       │
│                         │                               │
│              estimate_tokens() < 6000?                  │
│                    │         │                          │
│                   是        否                           │
│                    │         │                          │
│                    ▼         ▼                          │
│                 放行    Level 1: Score truncation       │
│                            丢弃低分 chunk（零成本）      │
│                                   │                    │
│                              fits? → 是 → 放行          │
│                                   │ 否                  │
│                                   ▼                     │
│                         Level 2: Content truncation     │
│                            截断到 300 字符/chunk         │
│                            （找句号边界，不硬切）         │
│                                   │                    │
│                              fits? → 是 → 放行          │
│                                   │ 否                  │
│                                   ▼                     │
│                         Level 3: Deduplication         │
│                            3-gram Jaccard 去重          │
│                            相邻重叠>70% → 合并          │
│                                   │                    │
│                              fits? → 是 → 放行          │
│                                   │ 否                  │
│                                   ▼                     │
│                         Level 4: LLM Summarization     │
│                            用 LLM 压缩上下文（+1调用）   │
│                                   │                    │
│                              失败/空 →                  │
│                                   ▼                     │
│                         Fallback: Aggressive truncate   │
│                            150 字符 + 3 条（极端兜底）   │
└─────────────────────────────────────────────────────────┘
```

**关键参数**：
- `max_context_tokens: 6000` — 输入上限
- `llm_response_reserve_tokens: 1500` — 留给 LLM 输出的空间
- `max_chunk_chars: 300` — Level 2 单块截断上限
- Token 估算：`chars / 1.2`（保守策略，高估 token 数）

**效果**：绝大多数场景 Level 1-2 就解决了，Level 4（LLM 压缩）作为最后手段极少触发。

---

### 3.2 多轮对话 — 用户说"上次那个问题"怎么办？

**问题**：用户连续追问，需要通过上下文理解"那个"、"它"等指代，否则检索和回答都会跑偏。

**当前策略**：对话记忆注入 Prompt

```
session_id="sess-001"

第 1 轮: "CT扫描图像伪影怎么排查？"
  → 检索 + LLM → 回答 → 写入 Redis 记忆

第 2 轮: "那探测器校准的具体步骤是什么？"
  → 从 Redis 读取第 1 轮对话
  → 拼入 Prompt 的【对话背景】段
  → LLM 看到完整上下文，理解"那"指 CT 伪影的探测器
  → 生成针对性的回答
```

**Prompt 注入格式**：
```
【历史对话摘要】
前 5 轮讨论了 CT 设备的图像伪影问题，已排查球管老化，确认为探测器校准偏移...

【最近对话】
用户: CT扫描图像伪影怎么排查？
助手: 根据工单 GD-2026-03001，建议先检查...
用户: 那探测器校准的具体步骤是什么？
```

**局限与改进方向**：
- 当前不做显式指代消解。如果用户连续跳过很多轮再追问，建议在查询预处理阶段加 LLM 指代消解改写（待实现）
- 消息上限 20 条 + LLM 摘要，防止记忆无限增长

---

### 3.3 三级记忆 — 对话历史怎么存？

```
add_message(session_id, "user", "CT伪影怎么处理？")
        │
        ├──► Redis (同步, ~2ms)
        │     LPUSH rag:session:sess001:messages
        │     LTRIM 保留最近 20 条
        │     TTL 24h
        │
        ├──► Kafka (异步, 不阻塞)
        │     send to rag.conversation.messages
        │     │
        │     ├─ Kafka 可用 → Consumer → MySQL
        │     └─ Kafka 挂了 → 直写 MySQL（fallback）
        │
        └──► LLM 摘要 (每 5 轮)
              摘要前 5 轮对话 → 写入 Redis + MySQL
              下次对话时注入 Prompt 的【历史对话摘要】
```

| 层级 | 存储 | 速度 | 容量 | 作用 |
|------|------|------|------|------|
| 短期 | Redis List | ~2ms | 20条/会话 | 实时上下文注入 |
| 中期 | LLM 摘要 | ~2000ms | 每5轮1条 | 压缩历史，省 token |
| 长期 | MySQL | ~10-50ms | 不限 | 永久存储，支持回溯 |

**降级链**：

```
Redis挂 → 跳过短期记忆，无上下文注入（单次问答模式）
Kafka挂 → 直写 MySQL
MySQL挂 → 只保留 Redis 短期（24h 后丢失）
全挂 → RAG 正常问答，只是无记忆
```

---

### 3.4 语义缓存 — 相同问题反复问怎么优化？

```
第 1 次: "CT伪影怎么处理？"
  → embedding(问题) → Redis 缓存为空 → 完整 RAG → 24s
  → store(问题, 答案, embedding) 存入 Redis

第 2 次: "CT扫描图像伪影怎么处理？"  (相似但措辞不同)
  → embedding(问题) → Redis 100 条全量拉取
  → 逐条余弦相似度计算
  → max_score = 0.97 > 0.95 → 命中！
  → 直接返回缓存答案 → 2.7s (9x 加速)
```

**存储结构**：
```
Key:  rag:cache:qa
Type: Redis List<JSON>
TTL:  3600s (1h)

[0] {"q":"CT伪影...", "a":"根据工单...", "embedding":[...1536维], "ts":...}
[1] {"q":"透析机...", "a":"首先检查...", "embedding":[...], "ts":...}
...
```

**LRU 淘汰**：RPUSH 新条目到右侧 → 超 100 条时 LTRIM 砍左侧旧条目

**高阈值（0.95）的合理性**：医疗场景下"CT伪影"和"MRI伪影"的 embedding 余弦可达 0.9+，但它们是不同设备——0.95 足够保守，宁可漏判不误判。

---

### 3.5 限流保护 — 如何防止 API 滥用和费用失控？

```
请求进来
   │
   ▼
Level 1: 全局限流 (Token Bucket)
   60 req/min, burst=15
   超限 → HTTP 429 + Retry-After
   │
   ▼
Level 2: 用户限流 (Token Bucket)
   20 req/min/user, burst=5
   用户标识: X-API-Key > User-ID > session_id > IP
   超限 → HTTP 429
   │
   ▼
Level 3: LLM 成本 (Sliding Window)
   100K tokens/h, 50 次/h
   超限 → HTTP 429 + 预算提示
   │
   ▼
放行 → RAG
```

**为什么 QPS 用 Token Bucket 而非固定窗口？**

固定窗口在 59s→00s 边界会产生 2x 瞬时流量。Token Bucket 以恒定速率补充令牌，天然平滑。

**为什么 LLM 预算用 Sliding Window？**

Token Bucket 适合秒/分钟级的速率控制。小时级的 token 预算需要精确统计过去 1 小时的实际消耗，Sliding Window（Redis Sorted Set + ZREMRANGEBYSCORE）更精确。

**Redis 挂了怎么办？** 自动退回进程内存模式（`threading.Lock`），损失分布式一致性但服务不中断。

---

### 3.6 容错降级 — 任何组件挂了系统还能用吗？

```
                     Embedding API ──→ 挂了 → "检索失败" ❌ (无法降级)
                     LLM API      ──→ 挂了 → "生成失败" ❌ (无法降级)
  RAG 依赖链 ──────── Milvus      ──→ 挂了 → "检索失败" ❌ (无法降级)
           │
           │          Redis(记忆) ──→ 挂了 → 跳过记忆 ✅ (无上下文模式)
           │          Redis(缓存) ──→ 挂了 → 走完整RAG ✅ (无加速)
           增强链 ──── Redis(限流) ──→ 挂了 → 退回内存 ✅
                     Kafka       ──→ 挂了 → 直写MySQL ✅
                     MySQL       ──→ 挂了 → 仅Redis  ✅ (24h后丢失)
                     Reranker    ──→ 挂了 → 原始排序 ✅
                     Query Rewrite→ 挂了 → 用原问题 ✅
                     成本上报    ──→ 挂了 → 静默忽略 ✅
```

**设计原则**：只有 Embedding 和 LLM API 是硬依赖（没有它们无法回答），其余全部可降级。

---

## 四、一次完整问答的数据流

```
POST /api/ask  {"question":"CT图像伪影怎么排查？", "session_id":"sess-001"}

 1. [限流]    全局限流check → 用户限流check → LLM预算check          ~2ms
 2. [记忆]    MemoryManager.add_message("sess-001", "user", ...)      ~2ms
 3. [缓存]    QACache.lookup(question) → embedding + 余弦匹配         ~100ms
              → 命中? 直接返回 ✅ (跳过后续)
              → 未命中? 继续 ↓
 4. [预处理]  QueryProcessor.rewrite("图像伪影") → "CT伪影 探测器校准" ~500ms
 5. [检索]    MilvusStore.hybrid_search(dense, sparse, top_k=5)       ~20ms
 6. [过滤]    丢弃 score < threshold 的 chunk                         ~0ms
 7. [压缩]    ContextCompressor.compress(chunks, ...)                 ~0ms
              → 检查是否 > 6000 tokens → 按需截断/去重
 8. [记忆]    MemoryManager.build_memory_prompt("sess-001")           ~2ms
              → 读取 Redis 历史 + LLM 摘要
 9. [Prompt]  build_full_prompt(question, chunks, memory, format)     ~0ms
              → 拼接 System Prompt + 记忆背景 + 检索结果 + 问题
10. [LLM]     LLMClient.generate(system_prompt, user_message)         ~2000ms
              → Qwen-Max 流式生成
11. [记忆]    MemoryManager.add_message("sess-001", "assistant", ...) ~2ms
12. [缓存]    QACache.store(question, answer)                         ~100ms
              → embedding + RPUSH 到 Redis

总延迟: ~2.7s（主要瓶颈在 LLM 生成，占 75%）
缓存命中: ~0.1s（跳过 5-11，仅 embedding + cosine）
```

---

## 五、日志与链路追踪

### 5.1 日志体系

```
配置: logging.basicConfig(level=INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

关键日志点:

[api]          POST /api/ask → 200 (2340ms)              ← 请求耗时
[RAGChain]     RAG问答: mode=hybrid rerank=true ...       ← 每次问答参数
[LLMClient]    LLM调用成功: input=1730 output=650 total=2380 tokens  ← token统计
[Reranker]     Reranker: query='CT伪影...' candidates=15 → top_n=5
[QACache]      QA缓存命中: similarity=0.9721              ← 缓存命中
[MemoryManager] 中期摘要已存储: session=sess-001... turns=1-5
[ContextCompressor] 上下文需压缩: 8230/6000 tokens (fixed=1200, chunks=10)
[RateLimiter]  分布式限流已启用 (Redis): global=60/min, user=20/min
```

### 5.2 请求链路追踪

```
每个请求的完整日志链路:

21:14:03 [INFO] api - POST /api/ask → 200 (2340ms)
  ├─ 21:14:03 [INFO] RateLimiter - global check OK, user=ip:192.168.1.1 remaining=14
  ├─ 21:14:03 [DEBUG] QACache - QA缓存未命中: best_similarity=0.8234 (threshold=0.95)
  ├─ 21:14:03 [INFO] QueryProcessor - Query rewrite: 'CT伪影...' → 'CT 图像伪影 探测器校准'
  ├─ 21:14:03 [INFO] Reranker - candidates=15 → top_n=5, top_score=0.9721
  ├─ 21:14:03 [DEBUG] ContextCompressor - 上下文无需压缩: 3210/6000 tokens (chunks=5)
  ├─ 21:14:04 [INFO] LLMClient - LLM调用成功: input=1850 output=720 total=2570
  ├─ 21:14:04 [INFO] MemoryManager - 中期摘要已存储: session=sess-001 turns=6-10
  └─ 21:14:04 [DEBUG] QACache - QA缓存已存储: 'CT图像伪影怎么排查？...' (total=42)
```

### 5.3 关键监控指标

| 指标 | 日志来源 | 采集方式 |
|------|---------|---------|
| API 延迟 | `api - POST /api/ask → 200 (2340ms)` | 中间件 `log_requests` |
| 检索耗时 | `RAGChain - 问答` 日志 | LLM 调用前后的 `time.perf_counter()` |
| LLM token | `LLMClient - input=X output=Y total=Z` | DashScope API `resp.usage` |
| 缓存命中率 | `QACache - 命中/未命中` 日志 | 计数统计 |
| 限流触发 | `RateLimitExceeded` 异常 | 异常处理器 |
| 预算超限 | `BudgetExceeded` 异常 | 异常处理器 |
| 记忆降级 | `MemoryManager - redis=... mysql=... kafka=...` | 初始化日志 |
| 压缩触发 | `ContextCompressor - 上下文需压缩/无需压缩` | 压缩器日志 |

### 5.4 生产环境增强建议

```
当前: Python logging → stdout

生产版:
  logging → structlog (结构化 JSON 日志)
          → Filebeat → Elasticsearch → Kibana 仪表盘
          → Prometheus metrics (/metrics 端点)
          → Grafana 面板: QPS / P50/P99延迟 / 缓存命中率 / token消耗 / 限流触发次数
          → 告警: LLM API 连续失败 3 次 → 钉钉/企微通知
```

### 5.5 分布式追踪（多服务场景）

```
当前是单进程，如果拆成微服务:

Request ID: X-Request-ID 头贯穿全链路

API Gateway → Rate Limiter Service → RAG Service → Milvus
                                                    → Embedding API
                                                    → LLM API
                                                    → Redis
                                                    → Kafka

每个服务打印: [request_id=abc123] [span=retrieve] ...
Elastic APM / Jaeger 收集 Span → 调用链可视化
```
