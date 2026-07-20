"""
限流与配额管理 — Token Bucket + LLM 成本追踪

三层防护:
  1. 全局限流 — API 整体 QPS 上限（每进程独立 or Redis 分布式）
  2. 用户级限流 — 按用户维度的请求频率限制
  3. LLM 成本控制 — 按用户维度的 token 消耗预算

Redis 用作分布式协调后端，不可用时自动退回进程内存模式。

用法:
  from src.utils.rate_limiter import RateLimiter, CostTracker, get_user_id

  limiter = RateLimiter()
  limiter.check_global()           # 全局限流 → 超限抛 RateLimitExceeded
  limiter.check_user("user-001")   # 用户限流 → 超限抛 RateLimitExceeded

  tracker = CostTracker()
  tracker.check_budget("user-001")    # 预算检查 → 超限抛 BudgetExceeded
  tracker.record_usage("user-001", input_tokens=500, output_tokens=300)
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import redis

from src.config import RateLimitConfig, get_config

# ═══════════════════════════════════════════════════════════════
# 上下文传播 — 跨调用链传递当前用户标识
# ═══════════════════════════════════════════════════════════════

_current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user_id", default=""
)


def set_current_user(user_id: str) -> None:
    """设置当前请求的用户 ID（API 入口调用）"""
    _current_user_id.set(user_id)


def get_current_user() -> str:
    """获取当前请求的用户 ID（LLMClient 等下游调用）"""
    return _current_user_id.get()

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 异常
# ═══════════════════════════════════════════════════════════════

class RateLimitExceeded(Exception):
    """请求频率超限"""

    def __init__(self, retry_after: float, limit: int, remaining: int = 0):
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        super().__init__(f"请求过于频繁，请在 {retry_after:.1f}s 后重试")


class BudgetExceeded(Exception):
    """LLM 使用预算超限"""

    def __init__(self, message: str, retry_after: float = 3600):
        self.retry_after = retry_after
        super().__init__(message)


# ═══════════════════════════════════════════════════════════════
# 用户标识提取
# ═══════════════════════════════════════════════════════════════

def get_user_id(request, session_id: str = None) -> str:
    """
    从请求中提取用户标识，优先级:
      X-API-Key > X-User-ID > session_id > 客户端 IP

    参数:
      request:     Starlette/FastAPI Request 对象
      session_id:  可选的会话 ID（来自请求体）
    """
    # 1. API Key（最可靠）
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key[:32]}"

    # 2. 显式用户 ID
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return f"user:{user_id[:64]}"

    # 3. 会话 ID
    if session_id:
        return f"session:{session_id[:64]}"

    # 4. 客户端 IP（兜底）
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"

    return f"ip:{ip}"


# ═══════════════════════════════════════════════════════════════
# Token Bucket — 进程内存实现
# ═══════════════════════════════════════════════════════════════

@dataclass
class _BucketState:
    tokens: float
    last_refill: float


class InMemoryTokenBucket:
    """
    线程安全的 Token Bucket。

    算法:
      - 以固定速率 refill_rate 补充 token（token/s）
      - 最大容量 capacity，防止突发无限堆积
      - 每次请求消耗 1 token
      - token 不足时拒绝，返回重试等待时间
    """

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._state: dict[str, _BucketState] = {}
        self._lock = threading.Lock()

    def consume(self, key: str, tokens: float = 1.0) -> tuple[bool, float, float, float]:
        """
        尝试消耗 token。

        参数:
          key:    桶标识（如 "global"、"user:xxx"）
          tokens: 消耗数量（默认 1）

        返回:
          (allowed, remaining, retry_after_seconds, capacity)
        """
        now = time.monotonic()

        with self._lock:
            state = self._state.get(key)
            if state is None:
                state = _BucketState(tokens=self.capacity, last_refill=now)
                self._state[key] = state

            # 补充 token
            elapsed = now - state.last_refill
            refill = elapsed * self.refill_rate
            state.tokens = min(self.capacity, state.tokens + refill)
            state.last_refill = now

            # 尝试消费
            if state.tokens >= tokens:
                state.tokens -= tokens
                remaining = state.tokens
                retry_after = 0.0
                allowed = True
            else:
                # 计算需要等待多久才能凑够 token
                deficit = tokens - state.tokens
                retry_after = deficit / self.refill_rate if self.refill_rate > 0 else 60.0
                remaining = state.tokens
                allowed = False

        return allowed, remaining, retry_after, self.capacity


# ═══════════════════════════════════════════════════════════════
# Token Bucket — Redis 分布式实现
# ═══════════════════════════════════════════════════════════════

# Lua 脚本：原子化的 token bucket consume
_REDIS_TOKEN_BUCKET_SCRIPT = """
local key        = KEYS[1]
local capacity   = tonumber(ARGV[1])
local rate       = tonumber(ARGV[2])
local requested  = tonumber(ARGV[3])
local now        = tonumber(ARGV[4])

local tokens = tonumber(redis.call('HGET', key, 'tokens'))
if tokens == nil then tokens = capacity end

local last_refill = tonumber(redis.call('HGET', key, 'last_refill'))
if last_refill == nil then last_refill = now end

-- 补充
local elapsed = math.max(0, now - last_refill)
local refill = elapsed * rate
tokens = math.min(capacity, tokens + refill)

-- 判断
local allowed = 0
local remaining = tokens
local retry_after = 0
if tokens >= requested then
    allowed = 1
    tokens = tokens - requested
    remaining = tokens
else
    retry_after = math.ceil((requested - tokens) / rate)
end

redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
redis.call('EXPIRE', key, math.ceil(capacity / rate) + 120)

return {allowed, remaining, retry_after, capacity}
"""


class RedisTokenBucket:
    """
    Redis 分布式 Token Bucket。

    通过 Lua 脚本保证原子性，适合多进程/多容器部署。
    Redis 不可用时 RateLimiter 自动退回 InMemoryTokenBucket。
    """

    def __init__(self, redis_client: redis.Redis, key_prefix: str = "rag:ratelimit:bucket"):
        self._redis = redis_client
        self._prefix = key_prefix
        self._script: Optional[object] = None  # 注册的 Lua script

    @property
    def available(self) -> bool:
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

    def _ensure_script(self):
        """懒注册 Lua 脚本"""
        if self._script is None:
            self._script = self._redis.register_script(_REDIS_TOKEN_BUCKET_SCRIPT)

    def consume(
        self,
        key: str,
        capacity: float,
        refill_rate: float,
        tokens: float = 1.0,
    ) -> tuple[bool, float, float, float]:
        """
        尝试消耗 token（Redis 原子操作）。

        参数同 InMemoryTokenBucket.consume。
        """
        self._ensure_script()
        redis_key = f"{self._prefix}:{key}"
        now = time.time()

        try:
            result = self._script(
                keys=[redis_key],
                args=[capacity, refill_rate, tokens, now],
            )
            allowed, remaining, retry_after, cap = result
            return bool(allowed), float(remaining), float(retry_after), float(cap)
        except Exception as e:
            logger.error(f"Redis TokenBucket 操作失败: {e}")
            raise


# ═══════════════════════════════════════════════════════════════
# 统一限流器
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    """
    统一限流器：全局限流 + 用户级限流。

    优先使用 Redis 分布式后端，不可用时自动退回进程内存模式。
    退回内存模式时打印一条 Warning（仅一次）。

    用法:
      limiter = RateLimiter()

      # 应用到 FastAPI 端点
      limiter.check_global()
      user_id = get_user_id(request)
      limiter.check_user(user_id)
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        cfg = config or get_config().rate_limit
        self._cfg = cfg
        self._redis: Optional[redis.Redis] = None
        self._redis_bucket: Optional[RedisTokenBucket] = None
        self._memory_bucket: Optional[InMemoryTokenBucket] = None
        self._use_redis = False
        self._warned_fallback = False

        if not cfg.enabled:
            logger.info("限流已禁用 (rate_limit.enabled=false)")
            return

        # 尝试 Redis
        if cfg.redis_enabled:
            try:
                from src.config import get_config as _gc
                redis_cfg = _gc().memory.redis
                self._redis = redis.Redis(
                    host=redis_cfg.host,
                    port=redis_cfg.port,
                    db=redis_cfg.db,
                    password=redis_cfg.password,
                    decode_responses=False,  # Lua 脚本需要原始字节
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                self._redis.ping()
                self._redis_bucket = RedisTokenBucket(self._redis, key_prefix="rag:ratelimit:bucket")
                self._use_redis = True
                logger.info(
                    f"分布式限流已启用 (Redis): "
                    f"global={cfg.global_rate}/min, user={cfg.user_rate}/min, "
                    f"budget={cfg.llm_max_tokens_per_hour} tok/h"
                )
            except Exception as e:
                logger.warning(f"Redis 不可用，退回进程内存限流: {e}")

        # 内存兜底
        if not self._use_redis:
            # 全局桶: capacity=burst_size, refill_rate=requests_per_minute/60
            global_refill = cfg.global_rpm / 60.0
            global_cap = max(cfg.global_burst, global_refill)
            self._global_bucket = InMemoryTokenBucket(global_cap, global_refill)

            user_refill = cfg.user_rpm / 60.0
            user_cap = max(cfg.user_burst, user_refill)
            self._user_bucket_pattern = InMemoryTokenBucket(user_cap, user_refill)

            logger.info(
                f"进程内存限流已启用: "
                f"global={cfg.global_rpm}/min burst={cfg.global_burst}, "
                f"user={cfg.user_rpm}/min burst={cfg.user_burst}"
            )

    # ── 公开方法 ──────────────────────────────

    def check_global(self) -> None:
        """
        全局限流检查。

        抛出:
          RateLimitExceeded — 超限时
        """
        if not self._cfg.enabled:
            return

        allowed, remaining, retry, cap = self._consume_global()
        if not allowed:
            headers = {
                "X-RateLimit-Limit": str(int(cap)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time() + retry)),
                "Retry-After": str(int(retry + 0.999)),
            }
            raise RateLimitExceeded(
                retry_after=retry,
                limit=int(cap),
                remaining=0,
            )

        # 存储用于后续响应头（调用 get_headers 可获取）
        self._last_global_headers = {
            "X-RateLimit-Limit": str(int(cap)),
            "X-RateLimit-Remaining": str(int(remaining)),
            "X-RateLimit-Reset": str(int(time.time() + (remaining / (self._cfg.global_rpm / 60.0)) if self._cfg.global_rpm > 0 else 0)),
        }

    def check_user(self, user_id: str) -> None:
        """
        用户级限流检查。

        参数:
          user_id: 用户标识（来自 get_user_id()）

        抛出:
          RateLimitExceeded — 超限时
        """
        if not self._cfg.enabled:
            return

        allowed, remaining, retry, cap = self._consume_user(user_id)
        if not allowed:
            raise RateLimitExceeded(
                retry_after=retry,
                limit=int(cap),
                remaining=0,
            )

        self._last_user_headers = {
            "X-RateLimit-User-Limit": str(int(cap)),
            "X-RateLimit-User-Remaining": str(int(remaining)),
            "X-RateLimit-User-Reset": str(int(time.time() + retry)),
        }

    @property
    def rate_limit_headers(self) -> dict[str, str]:
        """返回应在响应中附加的限流头"""
        h = {}
        if hasattr(self, "_last_global_headers"):
            h.update(self._last_global_headers)
        if hasattr(self, "_last_user_headers"):
            h.update(self._last_user_headers)
        return h

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    # ── 内部 ──────────────────────────────────

    def _consume_global(self) -> tuple[bool, float, float, float]:
        if self._use_redis:
            try:
                return self._redis_bucket.consume(
                    "global",
                    capacity=float(self._cfg.global_burst),
                    refill_rate=self._cfg.global_rpm / 60.0,
                )
            except Exception:
                if not self._warned_fallback:
                    logger.warning("Redis 限流失效，本次放行")
                return True, 1.0, 0.0, float(self._cfg.global_burst)

        return self._global_bucket.consume("global")

    def _consume_user(self, user_id: str) -> tuple[bool, float, float, float]:
        if self._use_redis:
            try:
                return self._redis_bucket.consume(
                    f"user:{user_id}",
                    capacity=float(self._cfg.user_burst),
                    refill_rate=self._cfg.user_rpm / 60.0,
                )
            except Exception:
                return True, 1.0, 0.0, float(self._cfg.user_burst)

        return self._user_bucket_pattern.consume(f"user:{user_id}")


# ═══════════════════════════════════════════════════════════════
# LLM 成本追踪
# ═══════════════════════════════════════════════════════════════

# Lua: 滑动窗口计数（小时级）
_COST_CHECK_SCRIPT = """
local key_tokens = KEYS[1]
local key_count  = KEYS[2]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_tokens = tonumber(ARGV[3])
local max_count  = tonumber(ARGV[4])

-- 清理过期
redis.call('ZREMRANGEBYSCORE', key_tokens, 0, now - window)
redis.call('ZREMRANGEBYSCORE', key_count,  0, now - window)

-- 当前用量
local token_sum = 0
local members = redis.call('ZRANGEBYSCORE', key_tokens, now - window, now, 'WITHSCORES')
for i = 1, #members, 2 do
    token_sum = token_sum + tonumber(members[i+1])
end
local request_count = redis.call('ZCOUNT', key_count, now - window, now)

-- 判断
local allowed = 1
local reason = ""
if max_tokens > 0 and token_sum >= max_tokens then
    allowed = 0
    reason = "token_budget_exceeded"
elseif max_count > 0 and request_count >= max_count then
    allowed = 0
    reason = "request_budget_exceeded"
end

redis.call('EXPIRE', key_tokens, window + 60)
redis.call('EXPIRE', key_count,  window + 60)

return {allowed, token_sum, request_count, max_tokens, max_count, reason}
"""

_RECORD_USAGE_SCRIPT = """
local key_tokens = KEYS[1]
local key_count  = KEYS[2]
local now = tonumber(ARGV[1])
local input_tokens  = tonumber(ARGV[2])
local output_tokens = tonumber(ARGV[3])
local window = tonumber(ARGV[4])

redis.call('ZADD', key_tokens, now, now * 1000000 + input_tokens + output_tokens)
redis.call('ZADD', key_count,  now, now)
redis.call('EXPIRE', key_tokens, window + 120)
redis.call('EXPIRE', key_count,  window + 120)

return 1
"""


class CostTracker:
    """
    LLM 成本追踪器。

    按用户维度统计 token 消耗 + 请求次数，超预算时拒绝。

    存储:
      Redis Sorted Set: {key_prefix}:tokens:{user_id}  — score=timestamp, member=unique_id
      Redis Sorted Set: {key_prefix}:count:{user_id}   — score=timestamp, member=timestamp

    容错: Redis 不可用时所有检查放行，记录 Warning。
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        cfg = config or get_config().rate_limit
        self._cfg = cfg
        self._redis: Optional[redis.Redis] = None
        self._available = False

        if not cfg.enabled or not cfg.cost_enabled:
            logger.info("LLM 成本追踪已禁用")
            return

        try:
            from src.config import get_config as _gc
            redis_cfg = _gc().memory.redis
            self._redis = redis.Redis(
                host=redis_cfg.host,
                port=redis_cfg.port,
                db=redis_cfg.db,
                password=redis_cfg.password,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._redis.ping()
            self._check_script = self._redis.register_script(_COST_CHECK_SCRIPT)
            self._record_script = self._redis.register_script(_RECORD_USAGE_SCRIPT)
            self._available = True
            logger.info(
                f"LLM 成本追踪已启用: "
                f"max={cfg.llm_max_tokens_per_hour} tok/h, "
                f"max_requests={cfg.llm_max_requests_per_hour}/h"
            )
        except Exception as e:
            logger.warning(f"LLM 成本追踪 Redis 不可用，预算检查放行: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def check_budget(self, user_id: str) -> None:
        """
        检查用户 LLM 使用预算。

        在调用 LLM 之前调用，超限时抛 BudgetExceeded。

        参数:
          user_id: 用户标识

        抛出:
          BudgetExceeded — token 或请求次数超限
        """
        if not self._available:
            return

        try:
            key_tokens = f"rag:cost:tokens:{user_id}"
            key_count = f"rag:cost:count:{user_id}"
            now = time.time()
            window = 3600  # 1 小时窗口

            result = self._check_script(
                keys=[key_tokens, key_count],
                args=[
                    now,
                    window,
                    self._cfg.llm_max_tokens_per_hour,
                    self._cfg.llm_max_requests_per_hour,
                ],
            )
            allowed = bool(result[0])
            if not allowed:
                token_used = int(result[1])
                request_count = int(result[2])
                reason = result[5] if len(result) > 5 else "budget_exceeded"
                msg = (
                    f"LLM 使用预算已用尽: "
                    f"已消耗 {token_used} tokens, {request_count} 次请求"
                )
                logger.warning(f"[Budget] {user_id} {msg}")
                raise BudgetExceeded(msg)
        except BudgetExceeded:
            raise
        except Exception as e:
            logger.error(f"预算检查异常，放行: {e}")

    def record_usage(
        self,
        user_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """
        记录一次 LLM 调用的 token 消耗。

        在 LLM 调用成功后调用。

        参数:
          user_id:       用户标识
          input_tokens:  输入 token 数
          output_tokens: 输出 token 数
        """
        if not self._available:
            return

        try:
            key_tokens = f"rag:cost:tokens:{user_id}"
            key_count = f"rag:cost:count:{user_id}"
            now = time.time()
            window = 3600

            self._record_script(
                keys=[key_tokens, key_count],
                args=[now, input_tokens, output_tokens, window],
            )
            logger.debug(
                f"成本记录: user={user_id[:20]}... "
                f"in={input_tokens} out={output_tokens}"
            )
        except Exception as e:
            logger.error(f"成本记录失败: {e}")


# ═══════════════════════════════════════════════════════════════
# 模块级单例
# ═══════════════════════════════════════════════════════════════

_rate_limiter: Optional[RateLimiter] = None
_cost_tracker: Optional[CostTracker] = None
_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """获取全局限流器单例"""
    global _rate_limiter
    if _rate_limiter is None:
        with _lock:
            if _rate_limiter is None:
                _rate_limiter = RateLimiter()
    return _rate_limiter


def get_cost_tracker() -> CostTracker:
    """获取全局成本追踪器单例"""
    global _cost_tracker
    if _cost_tracker is None:
        with _lock:
            if _cost_tracker is None:
                _cost_tracker = CostTracker()
    return _cost_tracker
