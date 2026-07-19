# RAG 智能工单 — 三级记忆系统架构图

```
+==========================================================================================+
|                          RAG 智能工单 — 三级记忆系统架构                                    |
+==========================================================================================+

                              +---------------------------+
                              |       用户 / API 请求       |
                              |   POST /api/ask            |
                              |   session_id="abc123"      |
                              +-------------+-------------+
                                            |
                                            v
+------------------------------------------------------------------------------------------+
|                                  RAGChain.ask()                                          |
|                                                                                          |
|  1. add_message("user", question)                                                        |
|  2. 检索 (Milvus) -> 构建 Prompt                                                         |
|  3. build_memory_prompt() -> 注入对话历史上下文到 Prompt                                   |
|  4. LLM 生成回答                                                                          |
|  5. add_message("assistant", answer)                                                     |
|                                                                                          |
+----+-----------------------------------------+-------------------------------------------+
     |                                         |
     |  1. 保存用户消息                         |  5. 保存助手回答
     |  (同步)                                 |  + 触发摘要检查
     v                                         v
+------------------------------------------------------------------------------------------+
|                            MEMORY  MemoryManager (编排层)                                 |
|                                                                                          |
|  add_message(session_id, role, content)                                                  |
|    |                                                                                     |
|    +--> [1] Redis 短期记忆   写入 (同步，毫秒级)                                          |
|    |        +-- session:abc123:messages  (LPUSH + LTRIM 最近20条 + EXPIRE 24h)           |
|    |        +-- session:abc123:summary   (最新中期摘要)                                   |
|    |                                                                                     |
|    +--> [2] Kafka Producer   异步发送 (不阻塞，发后即忘)                                   |
|    |        +-- topic: rag.conversation.messages                                         |
|    |        +-- topic: rag.conversation.summaries                                        |
|    |        +-- 重试: _ensure_producer() 自动重连                                         |
|    |        +-- 退避: 0.5s -> 1s -> 2s -> 4s (cap 5s)                                   |
|    |                                                                                     |
|    +--> [3] LLM 摘要生成   每5轮触发                                                      |
|             +-- 成功 -> 写入 Redis + Kafka                                               |
|             +-- 失败 -> 返回 None，不阻塞                                                 |
|                                                                                          |
|  get_context(session_id) -> {messages, summary, turn_count, has_memory}                  |
|  build_memory_prompt(session_id) -> "[历史对话摘要]\n...\n[最近对话]\n..."                 |
|  clear_session(session_id) -> 删除 Redis key                                              |
+-------------------+----------------------------------+-----------------------------------+
                    |                                  |
                    |  [1] 同步读写                     |  [2] 异步发送
                    v                                  v
+------------------------------+        +----------------------------------------------+
|  [REDIS]                     |        |  [KAFKA] 消息队列                            |
|  localhost:6379              |        |  localhost:9092                              |
|                              |        |                                              |
|  +-- 短期记忆 (TTL 24h) ---+ |  异步  |  +-- Topic -------------------------------+  |
|  |                         | |  管道  |  |  rag.conversation.messages             |  |
|  |  messages: List<JSON>   | |<------>|  |  rag.conversation.summaries            |  |
|  |  summary:  String       | | Producr|  +----------------------------------------+  |
|  |                         | |  ----> |                                              |
|  |  过期: 24小时           | | Consumr|  Consumer Group: rag-memory-consumer          |
|  |  上限: 20条/会话        | |  <---- |  降级: NoBrokersAvailable -> 暂停写入         |
|  +-------------------------+ |        |                                              |
|                              |        +----------------------+-----------------------+
|  降级: Error 10061          |                               |
|  -> available=False         |                               |  [3] 后台消费
|  -> 不影响问答              |                               |  Consumer Thread (daemon)
+------------------------------+                               v
                                            +------------------------------------------+
                                            |  [MYSQL] 长期记忆                          |
                                            |  localhost:3306                            |
                                            |  database: rag_memory                      |
                                            |                                            |
                                            |  +-- rag_conversations -----------------+  |
                                            |  |  id          BIGINT PK              |  |
                                            |  |  session_id  VARCHAR(64)            |  |
                                            |  |  role        VARCHAR(16)            |  |
                                            |  |  content     TEXT                   |  |
                                            |  |  turn_number INT                    |  |
                                            |  |  created_at  TIMESTAMP              |  |
                                            |  |  INDEX(session_id, turn_number)     |  |
                                            |  +--------------------------------------+  |
                                            |                                            |
                                            |  +-- rag_summaries --------------------+  |
                                            |  |  id          BIGINT PK              |  |
                                            |  |  session_id  VARCHAR(64)            |  |
                                            |  |  summary     TEXT                   |  |
                                            |  |  turn_range  VARCHAR(32)            |  |
                                            |  |  created_at  TIMESTAMP              |  |
                                            |  |  INDEX(session_id)                  |  |
                                            |  +--------------------------------------+  |
                                            |                                            |
                                            |  重试: 指数退避 3次                        |
                                            |  降级: pymysql.err -> 跳过                |
                                            +--------------------------------------------+

                                            +------------------------------------------+
                                            |  [LLM] 中期摘要 (每5轮)                    |
                                            |  DashScope qwen-max                        |
                                            |                                            |
                                            |  触发: turn % 5 == 0 且 role=="assistant" |
                                            |  输入: 最近 5 轮对话消息                    |
                                            |  输出: <300字 结构化摘要                    |
                                            |                                            |
                                            |  摘要内容:                                  |
                                            |  +-- 用户核心问题 + 设备类型                |
                                            |  +-- 诊断结论 + 处理方案                    |
                                            |  +-- 工单编号引用                           |
                                            |  +-- 未解决的问题                           |
                                            |                                            |
                                            |  失败: 返回 None -> 不阻塞                 |
                                            +--------------------------------------------+
```

---

## 数据流时序

```
时间轴 ------------------------------------------------------------------------------->

User: "CT伪影怎么处理？"
       |
t=0    |  RAGChain.ask(session_id="abc123")
       +---> MemoryManager.add_message("user", "CT伪影...")
       |        +-- Redis: LPUSH + LTRIM              [~2ms, 同步]
       |        +-- Kafka: _send_with_retry()          [异步, 不等待]
       |
       +---> Milvus search()                           [~50ms]
       |
       +---> build_memory_prompt("abc123")
       |        +-- Redis: LRANGE messages              [~1ms]
       |        +-- Redis: GET summary                  [~1ms]
       |        --> 注入到 Prompt 的 [对话背景] 段
       |
       +---> LLM generate(prompt + memory_context)     [~2000ms]
       |
       +---> MemoryManager.add_message("assistant", answer)
                +-- Redis: LPUSH + LTRIM               [~2ms]
                +-- Kafka: _send_with_retry()           [异步]
                +-- turn==5? --> LLM 生成摘要
                     +-- Redis: SET summary            [~2ms]
                     +-- Kafka: send_summary()          [异步]
       |
       +---> 返回 answer + sources + session_id


异步路径 (并行进行，不阻塞用户响应):

  Kafka Producer (t=2ms)                      Kafka Consumer (t~100ms)
  +--------------------------+                +-----------------------------+
  | send(topic, payload)     |    Broker      | poll() -> msg               |
  |   +-- future.get(10s)    |<=============> |   +-- process(msg)          |
  |      +-- OK -> return    |                |      +-- MySQL.insert_*()   |
  |      +-- Fail -> retry   |                |         +-- OK -> commit    |
  |         +-- 0.5s 延迟    |                |         +-- Fail -> retry   |
  |         +-- 1.0s 延迟    |                |            +-- 0.5s 延迟    |
  |         +-- 2.0s 延迟    |                |            +-- 1.0s 延迟    |
  +--------------------------+                |            +-- 2.0s 延迟    |
                                              +-----------------------------+
```

---

## 降级策略矩阵

```
  组件故障              | 系统行为
  ----------------------+------------------------------------------------
  Redis 不可用          | available=False -> 跳过短期记忆
                        | -> 问答正常，无记忆上下文，无摘要
                        | -> 不影响 API 响应
  ----------------------+------------------------------------------------
  Kafka 不可用          | available=False -> 跳过消息队列
                        | -> Redis 短期记忆正常工作
                        | -> MySQL 长期记忆暂停写入
                        | -> 不影响 API 响应
  ----------------------+------------------------------------------------
  MySQL 不可用          | _get_conn() 重试3次后返回 None
                        | -> Redis + Kafka 正常工作
                        | -> Kafka 消息堆积在 Broker (等 Consumer 恢复)
                        | -> Consumer 连续5次异常后退出
  ----------------------+------------------------------------------------
  LLM 摘要生成失败      | generate() 返回 None
                        | -> 跳过本次摘要
                        | -> 5轮后再尝试
  ----------------------+------------------------------------------------
  全部不可用            | RAG 问答正常 (检索 + LLM 生成)
                        | -> session_id 不传 = 无记忆模式
  ----------------------+------------------------------------------------
  不传 session_id       | 完全跳过记忆系统，行为与优化前一致
```

---

## 重试参数总览

```
  组件     | 操作         | 最大重试 | 退避序列              | 失败返回
  ---------+--------------+----------+-----------------------+--------
  MySQL    | 连接/写入    | 3        | 0.5-1-2-4s (cap 5s)  | False
  MySQL    | 查询 (读)    | 2        | 0.5-1s   (cap 5s)    | []
  Kafka    | Producer发送 | 3        | 0.5-1-2-4s (cap 5s)  | False
  Kafka    | Producer重连 | 3        | 0.5-1-2-4s (cap 5s)  | False
  Kafka    | Consumer异常 | 连续5次  | 2-4-8-16-30s          | 退出消费
  LLM摘要  | generate()   | 1        | -                    | None
```

---

## Redis Key 设计

```
  Key Pattern                            | Type  | TTL   | 说明
  ---------------------------------------+-------+-------+------------------
  rag:session:{sid}:messages             | List  | 24h   | 最近20条消息 (JSON)
  rag:session:{sid}:summary              | String| 24h   | 最新中期摘要文本

  消息格式:
  {
    "role": "user|assistant",
    "content": "消息文本...",
    "turn": 3,
    "timestamp": "2026-07-20T15:30:00+08:00"
  }
```

## MySQL 表设计

```sql
-- 对话消息表
CREATE TABLE rag_conversations (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id  VARCHAR(64)    NOT NULL,
    role        VARCHAR(16)    NOT NULL,       -- user | assistant
    content     TEXT           NOT NULL,
    turn_number INT            NOT NULL DEFAULT 0,
    created_at  TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_session_turn (session_id, turn_number)
);

-- 中期摘要表
CREATE TABLE rag_summaries (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id  VARCHAR(64)    NOT NULL,
    summary     TEXT           NOT NULL,
    turn_range  VARCHAR(32)    NOT NULL DEFAULT '', -- e.g. "1-5", "6-10"
    created_at  TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id)
);
```

## Kafka Topic 设计

```
  Topic                        | 分区 | 副本 | 消息内容
  -----------------------------+------+------+----------------------------------
  rag.conversation.messages    | 1    | 1    | {session_id, type:"message", message:{...}}
  rag.conversation.summaries   | 1    | 1    | {session_id, type:"summary", summary:"...", turn_range:"1-5"}
```
