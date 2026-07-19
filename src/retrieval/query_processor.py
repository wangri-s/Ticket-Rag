"""
查询预处理模块 — 检索前对用户问题进行改写/扩展，弥合口语与专业术语差距

问题场景:
  用户口语化描述 vs 知识库专业术语之间存在 gap。例如：
    用户: "机器拍出来的片子不清楚"
    知识库: "CT扫描图像伪影，表现为条状低密度阴影..."
  直接检索时，embedding 对口语和术语的语义对齐不够精准。

方案:
  1. Query Rewrite（查询改写）: 用 LLM 将口语问题改写为关键词组合
  2. Multi-Query（多路召回）: 生成 2-3 个查询变体，各自检索后合并去重

用法:
  proc = QueryProcessor()
  keywords = proc.rewrite("机器拍出来的片子不清楚")
  # → "CT 图像伪影 分辨率下降 探测器校准"
"""

import logging
from functools import lru_cache
from typing import Optional

from src.llm.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ── Query Rewrite Prompt ──────────────────────

QUERY_REWRITE_PROMPT = """你是一个医疗设备运维领域的搜索查询优化专家。

【任务】
将用户的口语化故障描述改写为**利于向量检索的关键词组合**。

【规则】
1. 补充设备专业名称和型号（如"透析机"→"血液透析机"）
2. 补充故障专业术语（如"片子不清楚"→"图像伪影 分辨率下降"）
3. 补充可能的排查方向（如"报警了"→"报警 参数异常 传感器校准"）
4. 保留用户的核心意图，不要添加无关信息
5. **只输出关键词，用空格分隔，不要完整句子，不要解释**

【示例】
用户: 机器拍出来的片子不清楚
改写: CT 图像伪影 分辨率下降 探测器校准 影像模糊

用户: 透析机报警了
改写: 血液透析机 报警 电导率偏高 透析液 传感器校准

用户: 手术室设备漏气了
改写: 麻醉机 呼吸回路 漏气测试 泄漏量异常

用户: 彩超探头不好用了
改写: 超声探头 成像模糊 声学堆栈 阵元衰减 探头更换

现在请处理以下查询：
用户: {question}
改写:"""


# ── Multi-Query Expansion Prompt ─────────────

MULTI_QUERY_PROMPT = """你是一个搜索查询优化专家。用户正在搜索医疗设备运维工单知识库。

【任务】
为用户的原始问题生成 {n} 个**语义等价但表述不同**的查询变体，用于多路召回提高命中率。

【规则】
1. 每个变体从不同角度描述同一问题（不同措辞、不同术语、不同粒度）
2. 保持用户原意，不要引入新问题
3. 每行一个变体，不要编号，不要解释

【示例】
用户: CT扫描图像伪影怎么排查？
变体:
CT扫描出现伪影故障处理方案
CT图像质量下降原因及维修步骤
CT球管校准探测器故障伪影

现在请处理：
用户: {question}
变体:"""


# ── QueryProcessor ────────────────────────────

class QueryProcessor:
    """
    查询预处理器：改写 + 多路扩展。

    - rewrite():  口语→关键词，用 LLM 做 query2keywords
    - expand():   生成多个查询变体，各自检索后合并
    """

    def __init__(self):
        self._llm = _get_query_llm()

    # ── 查询改写 ──────────────────────────────

    def rewrite(self, question: str) -> str:
        """
        将用户口语问题改写为检索友好的关键词组合。

        参数:
          question: 用户原始问题

        返回:
          改写后的关键词串（空格分隔），失败时返回原始问题
        """
        prompt = QUERY_REWRITE_PROMPT.format(question=question)
        try:
            result = self._llm.generate(
                user_message=prompt,
                system_prompt="你是一个专业的查询优化助手，只输出关键词。",
            )
            rewritten = result.strip()
            if not rewritten or len(rewritten) < 2:
                logger.warning(f"Query rewrite 返回空，回退原查询")
                return question
            logger.info(
                f"Query rewrite: '{question[:30]}...' → '{rewritten[:60]}...'"
            )
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite 失败: {e}，使用原查询")
            return question

    # ── 多路查询扩展 ──────────────────────────

    def expand(self, question: str, n: int = 3) -> list[str]:
        """
        生成多个查询变体，用于多路召回。

        参数:
          question: 用户原始问题
          n:        生成变体数（含原始问题）

        返回:
          查询变体列表（原始问题 + LLM 生成的变体），失败时只返回 [question]
        """
        prompt = MULTI_QUERY_PROMPT.format(question=question, n=n - 1)
        try:
            result = self._llm.generate(
                user_message=prompt,
                system_prompt="你是一个搜索查询优化助手。",
            )
            variants = [
                line.strip()
                for line in result.strip().split("\n")
                if line.strip()
            ]
            # 去重 + 保留原问题
            seen = {question}
            unique = [question]
            for v in variants:
                if v not in seen and len(v) >= 2:
                    seen.add(v)
                    unique.append(v)
            logger.info(
                f"Multi-query: 原始 1 条 → 扩展 {len(unique)} 条变体"
            )
            return unique[:n]  # 截断到 n 个
        except Exception as e:
            logger.warning(f"Multi-query expand 失败: {e}，使用原查询")
            return [question]


# ── 懒加载 LLM（独立实例，低温度）────────────

@lru_cache(maxsize=1)
def _get_query_llm() -> LLMClient:
    """Query 预处理专用 LLM（温度更低，输出更稳定）"""
    return LLMClient()
