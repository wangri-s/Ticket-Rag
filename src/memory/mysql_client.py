"""
MySQL 长期记忆客户端 — 含指数退避重试机制

表结构:
  rag_conversations — 对话消息表
  rag_summaries     — 中期摘要表

重试策略:
  - 连接失败: 指数退避重试 (0.5s→1s→2s→4s, 上限 5s)
  - 写入失败: 自动重连 + 重试，区分可重试错误
  - 查询失败: 2 次轻量重试

用法:
  mysql_mem = MySQLMemoryStore()
  mysql_mem.init_tables()
  mysql_mem.insert_message(session_id, {...})
  mysql_mem.insert_summary(session_id, "摘要", "1-5")
  history = mysql_mem.get_history(session_id)
"""

import logging
import time
from typing import Callable, Optional

import pymysql
from pymysql.cursors import DictCursor

from src.config import MySQLConfig, get_config

logger = logging.getLogger(__name__)

# ── 建表 SQL ──────────────────────────────────

CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS rag_conversations (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id  VARCHAR(64)    NOT NULL COMMENT '会话ID',
    role        VARCHAR(16)    NOT NULL COMMENT '角色: user|assistant',
    content     TEXT           NOT NULL COMMENT '消息内容',
    turn_number INT            NOT NULL DEFAULT 0 COMMENT '对话轮数',
    created_at  TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_session     (session_id),
    INDEX idx_session_turn (session_id, turn_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='RAG 对话消息 — 长期记忆';
"""

CREATE_SUMMARIES_TABLE = """
CREATE TABLE IF NOT EXISTS rag_summaries (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id  VARCHAR(64)    NOT NULL COMMENT '会话ID',
    summary     TEXT           NOT NULL COMMENT '摘要内容',
    turn_range  VARCHAR(32)    NOT NULL DEFAULT '' COMMENT '覆盖的轮次范围，如 1-5',
    created_at  TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='RAG 对话摘要 — 中期记忆';
"""

# 可重试的 MySQL 错误码
_RETRYABLE_ERROR_CODES = {
    2002,  # CR_CONNECTION_ERROR — 无法连接
    2003,  # CR_CONN_HOST_ERROR
    2006,  # CR_SERVER_GONE_ERROR — 连接断开
    2013,  # CR_SERVER_LOST — 查询过程中断连
    1205,  # ER_LOCK_WAIT_TIMEOUT — 锁等待超时
    1213,  # ER_LOCK_DEADLOCK — 死锁
    2001,  # CR_SOCKET_CREATE_ERROR
}


def _is_retryable(error: Exception) -> bool:
    """判断异常是否可重试"""
    if isinstance(error, pymysql.err.OperationalError):
        code = getattr(error, "args", [None])[0] if error.args else None
        return code in _RETRYABLE_ERROR_CODES
    if isinstance(error, pymysql.err.InterfaceError):
        return True  # 连接已关闭等接口错误
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


class MySQLMemoryStore:
    """
    MySQL 长期记忆存储 — 含指数退避重试机制。

    职责:
      - 持久化存储所有对话消息
      - 持久化存储中期摘要
      - 支持按会话查询历史
      - 连接断开/写入失败自动重试
    """

    def __init__(self, config: Optional[MySQLConfig] = None):
        cfg = config or get_config().memory.mysql
        self._cfg = cfg
        self._conn: Optional[pymysql.Connection] = None
        self._max_retries = cfg.max_retries
        self._base_delay = cfg.retry_base_delay
        self._max_delay = cfg.retry_max_delay

    # ── 重试引擎 ──────────────────────────────

    def _retry_with_backoff(
        self,
        operation: Callable[[], any],
        op_name: str,
        max_retries: int = None,
        is_read: bool = False,
    ) -> any:
        """
        指数退避重试包装器。

        参数:
          operation:   要执行的函数
          op_name:     操作名称（用于日志）
          max_retries: 最大重试次数（默认取 config）
          is_read:     True=读操作，失败不重连；False=写操作，失败先重连

        返回: operation 的返回值，全部尝试失败返回 None（写）/ []（读）
        """
        if max_retries is None:
            max_retries = self._max_retries

        last_error = None
        for attempt in range(max_retries + 1):  # 1 次初始 + N 次重试
            try:
                return operation()
            except pymysql.err.InterfaceError as e:
                # 连接已关闭 → 重连后重试
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"MySQL {op_name}: 连接断开，正在重连 "
                        f"(第 {attempt+1}/{max_retries} 次重试)"
                    )
                    self._reset_connection()
                    self._sleep_backoff(attempt)
            except pymysql.err.OperationalError as e:
                last_error = e
                if _is_retryable(e) and attempt < max_retries:
                    logger.warning(
                        f"MySQL {op_name}: 可重试错误 (code={e.args[0] if e.args else '?'}), "
                        f"第 {attempt+1}/{max_retries} 次重试: {e}"
                    )
                    if not is_read:
                        self._reset_connection()
                    self._sleep_backoff(attempt)
                else:
                    logger.error(f"MySQL {op_name}: 不可重试错误: {e}")
                    break
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        f"MySQL {op_name}: 未知错误，第 {attempt+1}/{max_retries} 次重试: {e}"
                    )
                    self._sleep_backoff(attempt)
                else:
                    logger.error(f"MySQL {op_name}: 重试耗尽: {e}")
                    break

        logger.error(
            f"MySQL {op_name}: 所有重试（{max_retries} 次）已耗尽。"
            f"最后错误: {last_error}"
        )
        return [] if is_read else False

    def _sleep_backoff(self, attempt: int):
        """指数退避等待: 0.5s → 1s → 2s → 4s (上限 5s)"""
        delay = min(self._base_delay * (2 ** attempt), self._max_delay)
        time.sleep(delay)

    # ── 连接管理 ──────────────────────────────

    def _connect(self) -> pymysql.Connection:
        """建立新连接（含自动建库）"""
        # 先确保数据库存在
        tmp_conn = pymysql.connect(
            host=self._cfg.host,
            port=self._cfg.port,
            user=self._cfg.user,
            password=self._cfg.password,
            charset=self._cfg.charset,
            connect_timeout=5,
        )
        with tmp_conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{self._cfg.database}` "
                f"DEFAULT CHARACTER SET {self._cfg.charset}"
            )
        tmp_conn.close()

        conn = pymysql.connect(
            host=self._cfg.host,
            port=self._cfg.port,
            user=self._cfg.user,
            password=self._cfg.password,
            database=self._cfg.database,
            charset=self._cfg.charset,
            cursorclass=DictCursor,
            autocommit=True,
            connect_timeout=5,
            read_timeout=10,
            write_timeout=10,
        )
        logger.info(f"MySQL 已连接: {self._cfg.host}:{self._cfg.port}/{self._cfg.database}")
        return conn

    def _reset_connection(self):
        """强制关闭并重建连接"""
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    def _get_conn(self) -> Optional[pymysql.Connection]:
        """获取连接（自动重连），不可用返回 None"""
        # 检查现有连接
        try:
            if self._conn is not None and self._conn.open:
                # ping 检查连接是否真正存活
                self._conn.ping(reconnect=False)
                return self._conn
        except Exception:
            self._reset_connection()

        # 重试建立新连接
        return self._retry_with_backoff(
            lambda: self._connect(),
            "连接",
            max_retries=self._max_retries,
            is_read=True,
        )

    # ── 表初始化 ──────────────────────────────

    def init_tables(self) -> bool:
        """创建表（幂等，首次启动调用）"""
        def _do_init():
            conn = self._get_conn()
            if conn is None:
                raise pymysql.err.OperationalError(2003, "无法获取 MySQL 连接")
            with conn.cursor() as cur:
                cur.execute(CREATE_CONVERSATIONS_TABLE)
                cur.execute(CREATE_SUMMARIES_TABLE)
            logger.info("MySQL 记忆表初始化完成")
            return True

        return self._retry_with_backoff(
            _do_init, "建表", max_retries=2, is_read=False,
        ) or False

    # ── 消息操作 ──────────────────────────────

    def insert_message(self, session_id: str, message: dict) -> bool:
        """
        插入一条消息到长期记忆（含重试）。

        参数:
          session_id: 会话 ID
          message:    {"role": "user|assistant", "content": "...",
                       "turn": int, "timestamp": "ISO8601"}
        """
        def _do_insert():
            conn = self._get_conn()
            if conn is None:
                raise pymysql.err.OperationalError(2003, "无法获取 MySQL 连接")
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO rag_conversations (session_id, role, content, turn_number) "
                    "VALUES (%s, %s, %s, %s)",
                    (
                        session_id,
                        message.get("role", "user"),
                        message.get("content", ""),
                        message.get("turn", 0),
                    ),
                )
            return True

        return self._retry_with_backoff(
            _do_insert, "插入消息", is_read=False,
        ) or False

    def insert_summary(self, session_id: str, summary: str, turn_range: str) -> bool:
        """插入一条中期摘要（含重试）"""
        def _do_insert():
            conn = self._get_conn()
            if conn is None:
                raise pymysql.err.OperationalError(2003, "无法获取 MySQL 连接")
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO rag_summaries (session_id, summary, turn_range) "
                    "VALUES (%s, %s, %s)",
                    (session_id, summary, turn_range),
                )
            return True

        return self._retry_with_backoff(
            _do_insert, "插入摘要", is_read=False,
        ) or False

    def get_history(self, session_id: str, limit: int = 50) -> list[dict]:
        """
        获取某个会话的历史消息（最近 limit 条）。
        读操作，2 次轻量重试。
        """
        def _do_query():
            conn = self._get_conn()
            if conn is None:
                return []
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content, turn_number, created_at "
                    "FROM rag_conversations "
                    "WHERE session_id = %s "
                    "ORDER BY id DESC LIMIT %s",
                    (session_id, limit),
                )
                rows = cur.fetchall()
            rows.reverse()
            return [
                {
                    "role": r["role"],
                    "content": r["content"],
                    "turn": r["turn_number"],
                    "timestamp": r["created_at"].isoformat() if r["created_at"] else "",
                }
                for r in rows
            ]

        return self._retry_with_backoff(
            _do_query, "查询历史", max_retries=2, is_read=True,
        ) or []

    def get_summaries(self, session_id: str, limit: int = 5) -> list[dict]:
        """获取某个会话的最近摘要。读操作，2 次轻量重试。"""
        def _do_query():
            conn = self._get_conn()
            if conn is None:
                return []
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT summary, turn_range, created_at "
                    "FROM rag_summaries "
                    "WHERE session_id = %s "
                    "ORDER BY id DESC LIMIT %s",
                    (session_id, limit),
                )
                rows = cur.fetchall()
            return [
                {
                    "summary": r["summary"],
                    "turn_range": r["turn_range"],
                    "timestamp": r["created_at"].isoformat() if r["created_at"] else "",
                }
                for r in rows
            ]

        return self._retry_with_backoff(
            _do_query, "查询摘要", max_retries=2, is_read=True,
        ) or []

    # ── 生命周期 ──────────────────────────────

    @property
    def available(self) -> bool:
        return self._get_conn() is not None

    def close(self):
        if self._conn and self._conn.open:
            self._conn.close()
