"""
Kafka 消息队列客户端 — Redis → MySQL 异步持久化管道（含重试机制）

Producer: 发送失败自动重试（指数退避），不可用时尝试重连
Consumer: 消费 Kafka 消息写入 MySQL（MySQL 内部已有重试）

用法:
  # Producer
  kafka_producer = KafkaProducerClient()
  kafka_producer.send_message(session_id, message_dict)
  kafka_producer.send_summary(session_id, summary, turn_range)

  # Consumer（通常独立进程运行）
  kafka_consumer = KafkaConsumerClient(mysql_store)
  kafka_consumer.start()
"""

import json
import logging
import threading
import time
from typing import Optional

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

from src.config import KafkaConfig, get_config

logger = logging.getLogger(__name__)


class KafkaProducerClient:
    """
    Kafka 生产者 — 将消息异步发布到 topic，发送失败自动重试。

    重试策略:
      - Producer 不可用时 → 尝试重连（指数退避）
      - send() 抛异常 → 指数退避重试（0.5s→1s→2s...）
      - KafkaError → 区分可重试/不可重试
      - 所有重试耗尽 → 返回 False（不阻塞主流程）
    """

    def __init__(self, config: Optional[KafkaConfig] = None):
        cfg = config or get_config().memory.kafka
        self.topic_messages = cfg.topic_messages
        self.topic_summaries = cfg.topic_summaries
        self._bootstrap_servers = cfg.bootstrap_servers
        self._max_retries = cfg.max_retries
        self._base_delay = cfg.retry_base_delay
        self._max_delay = cfg.retry_max_delay

        self._producer: Optional[KafkaProducer] = None
        self._connect()

    # ── 连接管理 ──────────────────────────────

    def _connect(self) -> bool:
        """尝试建立 Kafka 连接"""
        try:
            self._producer = KafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                value_serializer=lambda v: json.dumps(
                    v, ensure_ascii=False,
                ).encode("utf-8"),
                max_block_ms=5000,
                request_timeout_ms=5000,
                retries=2,  # Kafka 内置重试
                acks=1,     # leader ack 即可
            )
            logger.info(f"Kafka Producer 已连接: {self._bootstrap_servers}")
            return True
        except Exception as e:
            logger.warning(f"Kafka Producer 连接失败: {e}")
            self._producer = None
            return False

    def _ensure_producer(self) -> bool:
        """确保 producer 可用，不可用时尝试重连"""
        if self._producer is not None:
            try:
                # 轻量检查：尝试获取集群元数据
                self._producer.partitions_for(self.topic_messages)
                return True
            except Exception:
                logger.warning("Kafka Producer 连接失效，尝试重连...")
                try:
                    self._producer.close(timeout=5)
                except Exception:
                    pass
                self._producer = None

        # 重试重连
        for attempt in range(self._max_retries):
            if self._connect():
                return True
            delay = min(self._base_delay * (2 ** attempt), self._max_delay)
            logger.warning(
                f"Kafka 重连失败，{delay:.1f}s 后第 {attempt+2}/{self._max_retries} 次尝试..."
            )
            time.sleep(delay)

        logger.error(f"Kafka Producer 重连失败（{self._max_retries} 次重试耗尽）")
        return False

    def _sleep_backoff(self, attempt: int):
        """指数退避等待"""
        delay = min(self._base_delay * (2 ** attempt), self._max_delay)
        time.sleep(delay)

    # ── 发送接口 ──────────────────────────────

    def send_message(self, session_id: str, message: dict) -> bool:
        """
        发送消息到持久化 topic（含重试）。

        参数:
          session_id: 会话 ID
          message:    {"role", "content", "turn", "timestamp"}

        返回: True=发送成功，False=重试耗尽（不阻塞主流程）
        """
        payload = {"session_id": session_id, "type": "message", "message": message}
        return self._send_with_retry(self.topic_messages, payload, "消息", session_id)

    def send_summary(self, session_id: str, summary: str, turn_range: str) -> bool:
        """发送摘要到持久化 topic（含重试）"""
        payload = {
            "session_id": session_id,
            "type": "summary",
            "summary": summary,
            "turn_range": turn_range,
        }
        return self._send_with_retry(self.topic_summaries, payload, "摘要", session_id)

    def _send_with_retry(
        self, topic: str, payload: dict, label: str, session_id: str,
    ) -> bool:
        """
        指数退避重试发送。

        流程:
          1. 确保 producer 可用（不可用则重连）
          2. 发送消息（Future 模式，等待结果）
          3. 失败 → 区分可重试 vs 不可重试 → 退避重试
        """
        for attempt in range(self._max_retries + 1):  # 1 次初始 + N 次重试
            if not self._ensure_producer():
                # 重连失败，退避后继续
                if attempt < self._max_retries:
                    self._sleep_backoff(attempt)
                    continue
                return False

            try:
                future = self._producer.send(topic, value=payload)
                # 等待结果（带超时），确保捕获发送异常
                record_metadata = future.get(timeout=10)
                logger.debug(
                    f"Kafka {label}已发送: session={session_id[:12]}... "
                    f"topic={record_metadata.topic} offset={record_metadata.offset}"
                )
                return True
            except KafkaError as e:
                # 区分可重试 vs 不可重试
                retriable = getattr(e, "retriable", False)
                if retriable and attempt < self._max_retries:
                    logger.warning(
                        f"Kafka {label}发送失败（可重试），"
                        f"第 {attempt+1}/{self._max_retries} 次重试: {e}"
                    )
                    # 可重试错误后重置连接
                    try:
                        self._producer.close(timeout=5)
                    except Exception:
                        pass
                    self._producer = None
                    self._sleep_backoff(attempt)
                elif not retriable:
                    logger.error(f"Kafka {label}发送失败（不可重试）: {e}")
                    return False
                else:
                    logger.error(
                        f"Kafka {label}发送失败，重试耗尽（{self._max_retries} 次）: {e}"
                    )
                    return False
            except Exception as e:
                if attempt < self._max_retries:
                    logger.warning(
                        f"Kafka {label}发送异常，第 {attempt+1}/{self._max_retries} 次重试: {e}"
                    )
                    try:
                        self._producer.close(timeout=5)
                    except Exception:
                        pass
                    self._producer = None
                    self._sleep_backoff(attempt)
                else:
                    logger.error(
                        f"Kafka {label}发送失败，重试耗尽（{self._max_retries} 次）: {e}"
                    )
                    return False

        return False

    # ── 生命周期 ──────────────────────────────

    @property
    def available(self) -> bool:
        return self._producer is not None

    def close(self):
        if self._producer:
            self._producer.close(timeout=5)


class KafkaConsumerClient:
    """
    Kafka 消费者 — 消费 topic 消息写入 MySQL。

    MySQL 写入已有独立重试机制，Consumer 只需关注消费位移和异常恢复。
    """

    def __init__(
        self,
        mysql_store,  # MySQLMemoryStore 实例
        config: Optional[KafkaConfig] = None,
    ):
        cfg = config or get_config().memory.kafka
        self._mysql = mysql_store
        self.topic_messages = cfg.topic_messages
        self.topic_summaries = cfg.topic_summaries
        self.bootstrap_servers = cfg.bootstrap_servers
        self.consumer_group = cfg.consumer_group
        self._max_retries = cfg.max_retries

        self._consumer: Optional[KafkaConsumer] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, background: bool = True):
        """
        启动消费者。

        参数:
          background: True → 后台线程运行；False → 阻塞当前线程
        """
        try:
            self._consumer = KafkaConsumer(
                self.topic_messages,
                self.topic_summaries,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.consumer_group,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                max_poll_interval_ms=300000,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
            )
            logger.info(f"Kafka Consumer 已连接: {self.bootstrap_servers}")
        except Exception as e:
            logger.warning(f"Kafka Consumer 连接失败（长期记忆异步写入不可用）: {e}")
            return

        self._running = True

        if background:
            self._thread = threading.Thread(target=self._consume_loop, daemon=True)
            self._thread.start()
            logger.info("Kafka Consumer 已启动（后台线程）")
        else:
            self._consume_loop()

    def _consume_loop(self):
        """消费循环 — 含异常恢复"""
        logger.info("Kafka Consumer 开始消费...")
        consecutive_errors = 0

        while self._running:
            try:
                for msg in self._consumer:
                    if not self._running:
                        break
                    self._process(msg)
                    consecutive_errors = 0  # 成功后重置
            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    f"Kafka Consumer 消费异常（连续 {consecutive_errors} 次）: {e}"
                )
                if consecutive_errors >= 5:
                    logger.critical("Kafka Consumer 连续 5 次异常，停止消费")
                    break
                # 退避后重试
                time.sleep(min(2 ** min(consecutive_errors, 5), 30))
        else:
            pass  # self._running 被设置为 False

        if self._consumer:
            self._consumer.close()
        logger.info("Kafka Consumer 已退出消费循环")

    def _process(self, msg):
        """
        处理单条 Kafka 消息。

        MySQL insert 内部已有重试，此处只记录处理失败。
        """
        try:
            data = msg.value
            msg_type = data.get("type", "")
            session_id = data.get("session_id", "")

            if msg_type == "message":
                if not self._mysql.insert_message(session_id, data["message"]):
                    logger.warning(
                        f"Kafka Consumer: 消息写入 MySQL 失败（重试已耗尽）: "
                        f"session={session_id[:12]}..."
                    )
            elif msg_type == "summary":
                if not self._mysql.insert_summary(
                    session_id, data["summary"], data.get("turn_range", ""),
                ):
                    logger.warning(
                        f"Kafka Consumer: 摘要写入 MySQL 失败（重试已耗尽）: "
                        f"session={session_id[:12]}..."
                    )
        except Exception as e:
            # 不抛异常，避免消费者崩溃。消息 offset 已 commit，
            # 该条消息可能丢失但保证消费者继续运行
            logger.error(f"Kafka Consumer 消息处理异常（消息可能丢失）: {e}")

    def stop(self):
        """停止消费者"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        if self._consumer:
            self._consumer.close()
        logger.info("Kafka Consumer 已停止")

    @property
    def available(self) -> bool:
        return self._consumer is not None
