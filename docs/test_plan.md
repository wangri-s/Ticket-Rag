# 智能工单 RAG 系统 — 功能测试文档

> 一份文档看清：测了什么、怎么测的、边界在哪、怎么跑。

---

## 一、测试总览

### 1.1 测试金字塔

```
             ┌──────────┐
             │   E2E    │  人工 curl 验证（6 条链路）
            ┌┴──────────┴┐
            │  集成测试   │  RAGChain 全链路、API 端点、MemoryManager（Mock 全部依赖）
           ┌┴────────────┴┐
           │   单元测试    │  14 个模块，221 个用例，全部免外部依赖
          └───────────────┘
```

### 1.2 测试规模一览

| 指标 | 数值 |
|------|------|
| 测试文件总数 | 14 |
| 用例总数 | 221 |
| 全量通过时间 | ~52s |
| 外部依赖需求 | 0（全部 Mock） |
| 覆盖源码模块 | 16 |

### 1.3 模块 → 测试文件映射

| 模块 | 测试文件 | 用例数 | 类型 | 状态 |
|------|---------|--------|------|------|
| `config.py` | [test_config.py](../tests/test_config.py) | 11 | 单元 | ✅ |
| `ingestion/loader.py` | [test_loader.py](../tests/test_loader.py) | 6 | 单元 | ✅ |
| `ingestion/chunker.py` | [test_chunker.py](../tests/test_chunker.py) | 14 | 单元 | ✅ |
| `ingestion/pipeline.py` | [test_pipeline.py](../tests/test_pipeline.py) | 5 | 单元+集成 | ✅ |
| `retrieval/metadata_filter.py` | [test_metadata_filter.py](../tests/test_metadata_filter.py) | 15 | 单元 | ✅ |
| `llm/prompts.py` | [test_prompts.py](../tests/test_prompts.py) | 17 | 单元 | ✅ |
| `utils/context_compressor.py` | [test_context_compressor.py](../tests/test_context_compressor.py) | 20 | 单元 | ✅ |
| `utils/rate_limiter.py` | [test_rate_limiter.py](../tests/test_rate_limiter.py) | 20 | 单元 | ✅ |
| `memory/redis_client.py` | [test_redis_memory.py](../tests/test_redis_memory.py) | 16 | 单元+集成 | ✅ |
| `memory/qa_cache.py` | [test_qa_cache.py](../tests/test_qa_cache.py) | 19 | 单元 | ✅ |
| `memory/memory_manager.py` | [test_memory_manager.py](../tests/test_memory_manager.py) | 16 | 集成 | ✅ |
| `llm/llm_client.py` | [test_llm_client.py](../tests/test_llm_client.py) | 8 | 单元 | ✅ |
| `llm/rag_chain.py` | [test_rag_chain.py](../tests/test_rag_chain.py) | 11 | 集成 | ✅ |
| `api/ask.py` | [test_api.py](../tests/test_api.py) | 21 | 集成 | ✅ |
| `retrieval/milvus_client.py` | [test_milvus.py](../tests/test_milvus.py) | — | 手动 | 🔧 需 Docker |
| `embedding/` | [test_embed_manual.py](../tests/test_embed_manual.py) | — | 手动 | 🔧 需 API Key |

---

## 二、共享测试基础设施 — conftest.py

[conftest.py](../tests/conftest.py) 提供 13 个共享 fixtures，所有 Mock 测试文件共用：

| Fixture | 用途 |
|---------|------|
| `sample_chunks` | 4 个标准检索结果（分数递减: 0.95→0.82→0.68→0.55） |
| `single_chunk` | 单个 chunk |
| `fixed_embedding` / `another_embedding` | 固定 1536d 向量（用于 cosine 计算验证） |
| `mock_embedding_client` | Mock `EmbeddingClient`，返回固定向量 |
| `mock_redis` | Mock Redis — `ping()` 返回 `True` |
| `mock_redis_unavailable` | Mock Redis — `ping()` 抛 `ConnectionError` |
| `mock_llm_client` | Mock `LLMClient` — `generate()` 返回预设答案 |
| `mock_milvus_store` | Mock `MilvusStore` — 返回 sample_chunks |
| `mock_memory_manager` | Mock `MemoryManager` — 空历史 |
| `mock_qa_cache` | Mock `QACache` — 默认未命中 |
| `mock_compressor` | Mock `ContextCompressor` — 透传不压缩 |
| `mock_reranker` | Mock `Reranker` — 返回 top_n |
| `mock_query_processor` | Mock `QueryProcessor` — 返回改写关键词 |

---

## 三、各测试模块详解

### 3.1 test_config.py — 配置加载（11 用例）

**覆盖文件**: [src/config.py](../src/config.py)

| 用例类 | 用例数 | 测试重点 |
|--------|--------|---------|
| `TestEnvResolution` | 5 | `${VAR:-default}` 解析、嵌套变量、环境变量覆盖、非字符串透传 |
| `TestConfigLoading` | 6 | config.yml 全量解析、URI 格式、检索/chunking/LLM 默认值、单例模式、validate 检查 |

**运行**: `pytest tests/test_config.py -v` → 11 passed ✅

---

### 3.2 test_loader.py — 文档加载器（6 用例）

**覆盖文件**: [src/ingestion/loader.py](../src/ingestion/loader.py)

| 用例 | 场景 |
|------|------|
| `test_load_txt_file` | 正常加载 .txt → 1 Document，file_type="txt" |
| `test_load_nonexistent_file` | 文件不存在 → `FileNotFoundError` |
| `test_load_unsupported_format` | .xyz 扩展名 → `ValueError` |
| `test_load_directory` | 目录 2TXT+1 非文档 → 只加载 2 个 |
| `test_load_directory_skips_unsupported` | .png 混合 → 自动跳过 |
| `test_metadata_present` | source + file_type 元数据完整性 |

**运行**: `pytest tests/test_loader.py -v` → 6 passed ✅

---

### 3.3 test_chunker.py — 文本分割器（14 用例）

**覆盖文件**: [src/ingestion/chunker.py](../src/ingestion/chunker.py)

| 用例类 | 用例数 | 测试重点 |
|--------|--------|---------|
| `TestTicketIdExtraction` | 4 | GD-YYYY-NNNNN 正则匹配、无编号返回 None、多编号取首个 |
| `TestChunker` | 8 | 空文本/短文本不切分、超 500 切分、min_chunk_length 过滤、分隔符优先级、元数据继承、工单编号注入 |
| `TestSplitWithStats` | 2 | 统计字段完整性、max_size 不超标 |
| `TestGroupByTicket` | 1 | 按工单编号分组 + __unmatched__ 兜底 |

**运行**: `pytest tests/test_chunker.py -v` → 14 passed ✅

---

### 3.4 test_pipeline.py — 摄入流水线（5 用例）

**覆盖文件**: [src/ingestion/pipeline.py](../src/ingestion/pipeline.py)

| 用例 | 场景 |
|------|------|
| `test_run_with_txt_directory` | tempfile 目录 → ≥2 chunks |
| `test_run_with_single_file` | 单文件路径 → ≥2 chunks |
| `test_run_with_stats` | 统计模式 → total_chunks/total_tickets/ticket_ids |
| `test_run_empty_directory` | 空目录 → 返回 `[]` |
| `test_group_by_ticket_smoke` | 分组冒烟测试 |

**运行**: `pytest tests/test_pipeline.py -v` → 5 passed ✅

---

### 3.5 test_metadata_filter.py — 标量过滤（15 用例）

**覆盖文件**: [src/retrieval/metadata_filter.py](../src/retrieval/metadata_filter.py)

| 用例类 | 用例数 | 核心测试 |
|--------|--------|---------|
| `TestBuildFilterExpr` | 11 | 单条件 ticket_id/device_type、双条件 AND、无参/None/空字符串返回 None、部分参数、连字符/空格特殊字符 |
| `TestKnownDeviceTypes` | 4 | ≥10 个设备类型、含 CT/MRI/血液透析机 |

**关键设计验证**:

- `build_filter_expr()` 返回 → `ticket_id == "GD-2026-03001" and device_type == "CT 机"`
- 空字符串 `""` 视同未提供（Python falsy 语义）
- 已知设备类型列表用于 Streamlit UI 下拉菜单

**运行**: `pytest tests/test_metadata_filter.py -v` → 15 passed ✅

---

### 3.6 test_prompts.py — Prompt 构建（17 用例）

**覆盖文件**: [src/llm/prompts.py](../src/llm/prompts.py)

| 用例类 | 用例数 | 关键覆盖 |
|--------|--------|---------|
| `TestFormatChunks` | 5 | 空列表 → "无参考工单"、单/多 chunk 格式化、分隔线 `═`/`─` 清除、缺失 ticket_id 兜底 |
| `TestBuildRagPrompt` | 3 | text 格式构建、空 chunks→None、ticket_id_filter 注入 |
| `TestBuildJsonPrompt` | 2 | JSON 格式构建（含 JSON Schema）、空 chunks→None |
| `TestBuildFullPrompt` | 4 | text/json 双格式、空 chunks 返回 (None, None) |
| `TestFallback` | 3 | 兜底含"未找到"、JSON 兜底可解析、recommendations 含 steps |
| `TestSystemPrompt` | 2 | 非空、含"医疗"/"设备"关键词 |

**设计细节**:

- `format_chunks` 从 chunk content 中清除 `═`（U+2550）和 `─`（U+2500），但**格式化头** `─── 工单 N ───` 自身含 `─`，测试需区分这两种情况
- JSON fallback 返回合法 JSON（`has_reference: false`），下游可统一解析

**运行**: `pytest tests/test_prompts.py -v` → 17 passed ✅

---

### 3.7 test_context_compressor.py — 上下文压缩（20 用例）

**覆盖文件**: [src/utils/context_compressor.py](../src/utils/context_compressor.py)

| 用例类 | 用例数 | 测试内容 |
|--------|--------|---------|
| `TestEstimateTokens` | 7 | 空文本/None→0、中文/英文/混排估算 (`chars/1.2`)、极短文本最小为 1 |
| `TestEstimateChunkTokens` | 2 | 含 40 chars 格式化 overhead、空 content 仍有 overhead |
| `TestDeduplicateChunks` | 7 | 空列表/单条透传、相邻高重叠合并（保留高分）、低重叠保留两条、极高阈值不合并 |
| `TestContextCompressorBasics` | 2 | 空 chunks→`[]`、少量 chunk 无需压缩 |
| `TestCompressWithLongChunks` | 8 | L1 分数截断（高分保留）、L2 内容截断（句号边界/`…` 结尾）、fallback 固定开销超限→`chunks[:1]`、记忆上下文消耗预算、reserve_tokens 计入 |
| `TestCompressOrderPreservation` | 1 | 返回的 chunk 分数递减顺序不变 |
| `TestDedupIntegration` | 1 | 高重叠相邻 chunk 在压缩链路中被合并 |

**四级压缩策略验证**:

```
Level 1: Score truncation     → 按相关性分数丢弃低分 chunk
Level 2: Content truncation   → 截断到 max_chunk_chars (300)，找句号边界
Level 3: Deduplication        → 3-gram Jaccard > dedup_threshold (0.7) 合并
Level 4: LLM Summarization    → (Mock 未测，需真实 LLM)
Fallback: Aggressive          → chunks[:3], 每块 150 chars
```

**Token 估算公式**: `max(1, int(len(text) / 1.2))` — 保守策略，宁可多截不超限

**运行**: `pytest tests/test_context_compressor.py -v` → 20 passed ✅

---

### 3.8 test_rate_limiter.py — 限流配额（20 用例）

**覆盖文件**: [src/utils/rate_limiter.py](../src/utils/rate_limiter.py)

| 用例类 | 用例数 | 核心测试 |
|--------|--------|---------|
| `TestInMemoryTokenBucket` | 8 | 首次消费成功、连续耗尽、sleep 后恢复、capacity cap、不同 key 独立、retry_after 计算、零 rate 兜底 60s |
| `TestInMemoryTokenBucketConcurrency` | 1 | 15 线程并发 → 总消耗 ≤ capacity（`threading.Lock` 有效性） |
| `TestGetUserId` | 7 | API Key > User ID > Session ID > IP 优先级链、X-Forwarded-For 取首个、无 client→`"ip:unknown"` |
| `TestContextVars` | 2 | `set_current_user`/`get_current_user` 传播 |
| `TestRateLimiterDisabled` | 1 | `enabled=false` → check 全部放行 |
| `TestRateLimiterMemoryMode` | 2 | 全局 burst=3 → 第 4 次拒绝、用户 burst=5 独立 |
| `TestCostTracker` | 1 | Redis 不可用 → 预算检查放行（不阻塞核心流程） |
| `TestRateLimitExceeded` | 1 | Exception 含 retry_after/limit/remaining |
| `TestBudgetExceeded` | 1 | Exception 含 retry_after=3600、含"token"信息 |

**核心算法验证**:

- Token Bucket: `tokens = min(capacity, tokens + elapsed * refill_rate)`
- retry_after = `deficit / refill_rate`（需要等待的时间）
- 并发安全: `threading.Lock()` 保护 `_state` 字典

**运行**: `pytest tests/test_rate_limiter.py -v` → 20 passed ✅

---

### 3.9 test_redis_memory.py — Redis 短期记忆（16 用例）

**覆盖文件**: [src/memory/redis_client.py](../src/memory/redis_client.py)

| 用例类 | 用例数 | 测试内容 |
|--------|--------|---------|
| `TestRedisMemoryStoreWithMock` | 15 | add_message → rpush+ltrim+expire、get_messages JSON 解析、空列表、损坏 JSON 跳过、turn_count 统计、update_summary/get_summary/clear、available 属性、key 前缀格式 |
| `TestRedisMemoryStoreConnectionFailure` | 1 | ping 失败 → `_client=None` → available=False |

**存储结构验证**:

```
Key: rag:session:{sid}:messages  → List<JSON>  (LTRIM 到 max_messages=20)
Key: rag:session:{sid}:summary   → String       (TTL=86400s)
```

**降级行为**: `_client=None` 时 — add_message→False, get_messages→[], get_summary→None

**运行**: `pytest tests/test_redis_memory.py -v` → 16 passed ✅

---

### 3.10 test_qa_cache.py — 语义缓存（19 用例）

**覆盖文件**: [src/memory/qa_cache.py](../src/memory/qa_cache.py)

| 用例类 | 用例数 | 核心测试 |
|--------|--------|---------|
| `TestCosineSimilarity` | 6 | 相同向量→1.0、正交→0.0、相反→-1.0、零向量保护→0.0、近似但不相同 0.99~1.0 |
| `TestQACacheWithMock` | 10 | 空缓存→miss、相同向量命中、低于阈值 miss、多条取最高、损坏 JSON 跳过、空 embedding 跳过、store LRU 淘汰、TTL 续期、clear、available/size 属性 |
| `TestQACacheDisabled` | 3 | enabled=False → lookup 永远 miss、store 永远 False、available=False |
| `TestQACacheRedisUnavailable` | 1 | `_redis=None` → 所有操作安全返回 False/null |

**存储结构**: `rag:cache:qa` → Redis List\<JSON\> — `{q, a, embedding[...], ts}`，LRU: RPUSH + LTRIM

**余弦相似度**: `QACache._cosine(a, b)` = `dot(a,b) / (|a| * |b|)`，零向量安全

**运行**: `pytest tests/test_qa_cache.py -v` → 19 passed ✅

---

### 3.11 test_memory_manager.py — 三级记忆编排（16 用例）

**覆盖文件**: [src/memory/memory_manager.py](../src/memory/memory_manager.py)

| 用例类 | 用例数 | 测试内容 |
|--------|--------|---------|
| `TestMemoryManagerWithMocks` | 14 | add_message 写 Redis+Kafka、turn 计数（user 递增/assistant 不增）、get_context 结构、build_memory_prompt 空/有消息/有摘要/长内容截断 300、摘要第 5 轮触发/第 3 轮不触发、clear_session |
| `TestMemoryManagerDegradation` | 2 | Kafka 不可用→直写 MySQL；Kafka+MySQL 都挂→仅写 Redis（不崩溃） |

**降级链验证**:

```
Redis 挂 → 跳过短期记忆（单次问答模式）
Kafka 挂 → 直写 MySQL
MySQL 挂 → 只保留 Redis 短期
全挂   → RAG 正常问答，只是无记忆
```

**build_memory_prompt 格式**:

```
【历史对话摘要】
前 N 轮讨论了...

【最近对话】
用户: 上次的问题
助手: 上次的回答
```

**运行**: `pytest tests/test_memory_manager.py -v` → 16 passed ✅

---

### 3.12 test_llm_client.py — LLM 调用（8 用例）

**覆盖文件**: [src/llm/llm_client.py](../src/llm/llm_client.py)

| 用例类 | 用例数 | 测试内容 |
|--------|--------|---------|
| `TestLLMClientGenerate` | 3 | DashScope API 正常返回、system_prompt 传参、返回 str 非空 |
| `TestLLMClientRetry` | 2 | 前 2 次 500→第 3 次 200 成功、3 次全失败→RuntimeError |
| `TestLLMClientStream` | 1 | 流式 yield 多个 chunk（"根据"+"工单"） |
| `TestLLMClientCostReporting` | 1 | 成功后 `CostTracker.record_usage()` 被调用 |

**关键 Mock 细节**:

- DashScope `Generation.call()` 返回格式: `resp.output.text` (str) + `resp.usage` (dict)
- `get_current_user`/`get_cost_tracker` 在 `_call_api()` 内惰性导入 → patch 目标必须是 `src.utils.rate_limiter`
- 流式 chunk 格式: `chunk.output.text` + `chunk.status_code`

**运行**: `pytest tests/test_llm_client.py -v` → 8 passed ✅

---

### 3.13 test_rag_chain.py — RAG 全链路（11 用例）

**覆盖文件**: [src/llm/rag_chain.py](../src/llm/rag_chain.py)

| 用例类 | 用例数 | 场景 |
|--------|--------|------|
| `TestRAGChainCacheHit` | 2 | 缓存命中 → 跳过 LLM、流式缓存命中 |
| `TestRAGChainFullFlow` | 8 | 完整检索+生成、无结果→fallback、semantic/keyword 模式分发、JSON 输出格式、session 记忆保存（2次 add_message）、ticket_id_filter 传递 |
| `TestRAGChainScoreFilter` | 1 | score_threshold=0.5 → 低分 chunk 被过滤 |

**全链路 Mock 矩阵**: QACache + MemoryManager + LLMClient + MilvusStore + EmbeddingClient + BM25SparseEmbedder + ContextCompressor

**关键验证点**:

- 缓存命中时 `mock_llm.generate.assert_not_called()` — 确保不浪费 LLM 调用
- 无检索结果时 `has_answer=False`，LLM 也不应被调用
- `semantic` 模式 → `search()` 被调，`hybrid_search()` 不被调
- `keyword` 模式 → `sparse_search()` 被调
- session_id 传入 → `add_message` 被调用 2 次（user + assistant）

**运行**: `pytest tests/test_rag_chain.py -v` → 11 passed ✅

---

### 3.14 test_api.py — API 端点（21 用例）

**覆盖文件**: [src/api/ask.py](../src/api/ask.py)

| 用例类 | 用例数 | 测试内容 |
|--------|--------|---------|
| `TestAskEndpoint` | 11 | 200 正常返回、latency_ms、限流头注入、keyword/semantic mode、session_id、JSON 输出、ticket_id_filter、device_type_filter、rerank、query_expansion |
| `TestAskValidation` | 7 | 空 question→422、缺失字段→422、超长 2001→422、无效 mode→422、无效 output_format→422、top_k>20→422、top_k=0→422 |
| `TestAskStreamEndpoint` | 5 | 200 + `text/event-stream`、Cache-Control 头、sources/delta/done 三种 event、空问题→422 |
| `TestHealthEndpoint` | 1 | chain 已注入 → `"ready"` |
| `TestRateLimitIntegration` | 2 | `RateLimitExceeded`→429、`BudgetExceeded`→429 |

**Mock 策略**: 直接注入 `ask_mod._chain = MagicMock()` 而非 patch RAGChain 类 — 避免 `@lru_cache` 缓存的真实组件被初始化

**Pydantic 校验覆盖**:

| 字段 | 约束 | 测试 |
|------|------|------|
| `question` | min_length=1, max_length=2000 | 空字符串 + 超长 |
| `mode` | pattern=`semantic\|keyword\|hybrid` | 非法值 |
| `top_k` | ge=1, le=20 | 0 和 100 |
| `output_format` | pattern=`text\|json` | 非法值 |

**运行**: `pytest tests/test_api.py -v` → 21 passed ✅

---

## 四、Mock 依赖矩阵

| 模块 | 外部依赖 | Mock 方案 |
|------|---------|----------|
| `config.py` | 无 | 直接测 |
| `loader.py` | 文件系统 | `tempfile` |
| `chunker.py` | 无 | 直接测 |
| `pipeline.py` | loader + chunker | `tempfile` |
| `metadata_filter.py` | 无 | 直接测 |
| `prompts.py` | `config.yml` | 项目内已有 |
| `context_compressor.py` | `LLMClient` (L4) | 未达 L4，不触发 |
| `rate_limiter.py` | Redis, `time` | Mock Redis / `time.sleep` |
| `redis_client.py` | Redis | `unittest.mock.MagicMock` |
| `qa_cache.py` | Redis, EmbeddingClient | Mock 两者 |
| `memory_manager.py` | Redis, MySQL, Kafka, LLM | Mock 全部 4 个 |
| `llm_client.py` | `dashscope.Generation` | Mock `call()` |
| `rag_chain.py` | 全部 8 个依赖 | Mock 全部 |
| `api/ask.py` | RAGChain, RateLimiter, CostTracker | 直接注入 `_chain` + Mock 限流 |

---

## 五、E2E 验证清单

### 5.1 手动 curl 测试

```bash
# 1. 基础问答
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"CT扫描图像伪影怎么排查？","mode":"hybrid","top_k":3}' | python -m json.tool

# 2. 关键词检索
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"透析液电导率偏高","mode":"keyword","top_k":5}' | python -m json.tool

# 3. JSON 结构化输出
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"血液透析机电导率报警","output_format":"json"}' | python -m json.tool

# 4. 流式 SSE 输出
curl -N -X POST http://localhost:8000/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"MRI梯度线圈啸叫","mode":"hybrid","top_k":3}'

# 5. 多轮对话
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"CT伪影怎么处理？","session_id":"test-session-01"}'

curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"那探测器校准的具体步骤呢？","session_id":"test-session-01"}'

# 6. 限流触发（连续快速发送 >15 次）
for i in $(seq 1 20); do
  curl -s -o /dev/null -w "req $i: %{http_code}\n" \
    -X POST http://localhost:8000/api/ask \
    -H "Content-Type: application/json" \
    -d '{"question":"test","mode":"semantic"}'
done
```

### 5.2 手动验证点

| # | 验证项 | 检查方式 |
|---|--------|---------|
| 1 | 首次问答返回非空 answer | curl → `jq .answer` |
| 2 | sources 含 ticket_id | curl → `jq .sources[].ticket_id` |
| 3 | `from_cache=false`（首次） | curl → `jq .from_cache` |
| 4 | 重复问题返回 `from_cache=true` | 同一问题再次 curl |
| 5 | stream 模式有 3 种 event | curl -N → 看到 sources/delta/done |
| 6 | session 多轮对话能关联上下文 | 两轮问答 → 第二轮回答关联第一轮 |
| 7 | 无匹配问题返回 fallback | 问"今天天气怎么样" → has_answer=false |
| 8 | 429 限流触发 | 快速刷 20 次 |
| 9 | Streamlit UI 正常加载 | 浏览器 localhost:8501 |
| 10 | 三模式切换正常 | UI 下拉切换 semantic/keyword/hybrid |

---

## 六、测试数据

### 6.1 现有测试文档

项目已有 `tests/fixtures/test_tickets.txt`（用于 `test_pipeline.py`），内含 2 张工单（CT 伪影 + MRI 啸叫）。

### 6.2 测试用环境变量

```bash
# 单元测试无需任何环境变量（全部 Mock）
# 手动测试需要:
DASHSCOPE_API_KEY=sk-xxx    # Embedding + LLM
MILVUS_HOST=localhost        # Milvus 向量库
MILVUS_PORT=19530
REDIS_HOST=localhost         # 缓存 + 记忆
REDIS_PORT=6379
```

---

## 七、运行指南

### 7.1 日常开发（免外部依赖，~52s）

```bash
# 全量单元测试
pytest tests/ -v \
  --ignore=tests/test_milvus.py \
  --ignore=tests/test_milvus_manual.py \
  --ignore=tests/test_rag.py \
  --ignore=tests/test_embed_manual.py
```

### 7.2 单文件快速验证（~2s）

```bash
pytest tests/test_config.py tests/test_metadata_filter.py tests/test_rate_limiter.py -v
```

### 7.3 集成测试（需 Docker）

```bash
docker compose -p rag up -d redis
pytest tests/test_milvus.py tests/test_milvus_manual.py -v
```

### 7.4 覆盖率

```bash
pytest tests/ -v --cov=src --cov-report=term-missing \
  --ignore=tests/test_milvus.py \
  --ignore=tests/test_milvus_manual.py \
  --ignore=tests/test_rag.py \
  --ignore=tests/test_embed_manual.py
```

---

## 八、已知测试盲区

| 盲区 | 原因 | 缓解措施 |
|------|------|---------|
| Embedding API 真实调用 | 依赖 DashScope API Key | 手动 `test_embed_manual.py` |
| LLM 真实生成质量 | 依赖 Qwen-Max API | curl 手动验证 + Mock 覆盖调用链路 |
| Milvus 混合检索精度 | 需真实向量 + 完整语料 | Recall@K 人工评估 |
| Reranker 效果 | 依赖 qwen3-rerank API | NDCG 人工评估 |
| Kafka 消息可靠性 | 需消费者 + MySQL 对账 | 集成环境验证 |
| Redis 故障注入 | 需模拟网络分区 | Mock 覆盖降级路径 |
| ContextCompressor L4 | 需真实 LLM 压缩上下文 | L1-L3 全覆盖，L4 有代码 + 异常兜底 |
| 并发真实竞态 | Token Bucket 高并发 | `TestInMemoryTokenBucketConcurrency` 覆盖 15 线程 |
| 内存泄漏 | 长时间运行 | 24h 浸泡测试（暂未做） |

---

## 九、测试统计

### 9.1 按测试类型

| 类型 | 文件数 | 用例数 | 耗时 |
|------|--------|--------|------|
| 纯单元（无 Mock 外部服务） | 6 | 66 | ~5s |
| Mock 单元（Mock Redis/API） | 4 | 63 | ~10s |
| Mock 集成（Mock 全部依赖） | 3 | 48 | ~15s |
| 手动（需 Docker/API Key） | 4 | — | 按需 |

### 9.2 按模块

| 层级 | 模块 | 测试用例数 |
|------|------|-----------|
| 基础设施 | config, loader, chunker, pipeline | 36 |
| 检索 | metadata_filter, prompts | 32 |
| LLM | llm_client, rag_chain | 19 |
| 记忆/缓存 | redis_client, qa_cache, memory_manager | 51 |
| 工具 | context_compressor, rate_limiter | 40 |
| API | ask.py | 21 |
| **合计** | **14 文件** | **221** |

---

## 十、CI/CD 建议

```yaml
# .github/workflows/test.yml
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.11" }
    - run: pip install -r requirements.txt
    - run: |
        pytest tests/ -v \
          --ignore=tests/test_milvus.py \
          --ignore=tests/test_milvus_manual.py \
          --ignore=tests/test_rag.py \
          --ignore=tests/test_embed_manual.py
```

> **原则**: CI 只跑免外部依赖的测试。Milvus/LLM/Embedding API 相关测试在本地手动验证。
