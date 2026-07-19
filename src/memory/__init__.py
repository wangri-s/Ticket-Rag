"""
三级记忆系统 — 对话历史管理

架构:
  Redis   — 短期记忆（当前会话最近 N 轮对话，TTL 24h）
  Kafka   — 消息队列（Redis → MySQL 异步持久化管道）
  MySQL   — 长期记忆（全量会话历史，永久存储）
  LLM摘要 — 中期记忆（每 5 轮生成一次摘要，桥接短/长期记忆）

用法:
  from src.memory.memory_manager import MemoryManager

  manager = MemoryManager()
  manager.add_message(session_id, "user", "CT伪影怎么处理？")
  manager.add_message(session_id, "assistant", "根据工单...")
  context = manager.get_context(session_id)  # → 历史消息 + 摘要
"""
