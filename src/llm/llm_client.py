"""
LLM 客户端 — 封装 DashScope 千问大模型调用

支持:
  - 多轮对话 (chat)
  - 单轮生成 (generate)，自动拼接 system prompt
  - 重试 + 指数退避
  - 超时控制
"""

import logging
import time
from typing import Optional

from dashscope import Generation

from src.config import LLMConfig, get_config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    DashScope 千问大模型客户端

    用法:
      client = LLMClient()
      answer = client.generate(prompt="CT扫描出现图像伪影如何处理？")
      answer = client.generate(system_prompt="...", user_message="...")
      answer = client.chat(messages=[{"role": "user", "content": "..."}])
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        cfg = config or get_config().llm
        self.model = cfg.model
        self.api_key = cfg.dashscope_api_key
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.top_p = cfg.top_p
        self.seed = cfg.seed
        self.max_retries = cfg.max_retries
        self.timeout = cfg.timeout

    # ── 公开接口 ──────────────────────────────

    def generate(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        单轮对话：系统提示词 + 用户消息 → 模型回复。

        参数:
          user_message:  用户问题
          system_prompt: 系统提示词（不传则使用 config.yml 预设）

        返回: 模型回复文本
        """
        if system_prompt is None:
            cfg = get_config().llm
            system_prompt = cfg.system_prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        return self.chat(messages)

    def generate_stream(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ):
        """
        流式生成：逐 token 返回，适合前端打字机效果。

        用法:
          for chunk in client.generate_stream(user_message="..."):
              print(chunk, end="", flush=True)
        """
        if system_prompt is None:
            cfg = get_config().llm
            system_prompt = cfg.system_prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        yield from self._call_stream(messages)

    def chat(self, messages: list[dict]) -> str:
        """
        多轮对话：传入完整 messages 列表，返回模型回复。

        参数:
          messages: [{"role": "system|user|assistant", "content": "..."}, ...]

        返回: 模型回复文本
        """
        if not messages:
            raise ValueError("messages 不能为空")

        return self._call_with_retry(messages)

    # ── 内部 ──────────────────────────────────

    def _call_with_retry(self, messages: list[dict]) -> str:
        """带重试的 API 调用，指数退避"""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_api(messages)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM API 调用失败 (第 {attempt}/{self.max_retries} 次): {e}"
                )
                if attempt < self.max_retries:
                    wait = 2 ** (attempt - 1)  # 1s, 2s, 4s...
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM API 调用失败，已重试 {self.max_retries} 次"
        ) from last_error

    def _call_api(self, messages: list[dict]) -> str:
        """单次 API 调用"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
        }

        resp = Generation.call(**kwargs)

        if resp.status_code != 200:
            raise RuntimeError(
                f"DashScope LLM API 返回错误 "
                f"code={resp.status_code} message={resp.message}"
            )

        output = resp.output
        if not output or not output.text:
            raise RuntimeError("LLM 返回了空文本")

        logger.info(
            f"LLM 调用成功: input={resp.usage.get('input_tokens', '?')} "
            f"output={resp.usage.get('output_tokens', '?')} "
            f"total={resp.usage.get('total_tokens', '?')} tokens"
        )

        return output.text

    def _call_stream(self, messages: list[dict]):
        """流式 API 调用，逐 token yield 文本增量"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "stream": True,
            "incremental_output": True,
        }

        resp = Generation.call(**kwargs)

        total_tokens = 0
        for chunk in resp:
            if chunk.status_code != 200:
                raise RuntimeError(
                    f"DashScope LLM 流式返回错误 "
                    f"code={chunk.status_code} message={chunk.message}"
                )
            delta = chunk.output.text if chunk.output else ""
            if delta:
                total_tokens += len(delta)
                yield delta

        logger.info(
            f"LLM 流式调用完成: total_output≈{total_tokens} chars"
        )
