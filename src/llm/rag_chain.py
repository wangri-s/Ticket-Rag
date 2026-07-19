"""
RAG 核心链 — 串联检索 → Prompt 构建 → LLM 生成

流程:
  用户问题 → 向量检索 → 构建 Prompt → 调用 LLM → 返回答案 + 引用来源

用法:
  chain = RAGChain()
  result = chain.ask("CT扫描图像伪影怎么处理？")
  print(result["answer"])    # LLM 生成的回答
  for src in result["sources"]:  # 引用的工单来源
      print(src["ticket_id"], src["content"][:50])
"""

import logging
from functools import lru_cache
from typing import Literal, Optional

from src.config import ROOT_DIR, get_config
from src.embedding.embedding_client import EmbeddingClient
from src.embedding.sparse_embedder import BM25SparseEmbedder
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import MedicalWorkOrderChunker
from src.llm.llm_client import LLMClient
from src.llm.prompts import build_full_prompt, get_fallback_answer, get_json_fallback
from src.retrieval.milvus_client import MilvusStore
from src.retrieval.metadata_filter import build_filter_expr
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.reranker import Reranker
from src.memory.memory_manager import MemoryManager
from src.memory.qa_cache import QACache

logger = logging.getLogger(__name__)

SearchMode = Literal["semantic", "keyword", "hybrid"]


# ── 懒加载组件（首次使用初始化，后续复用）─────

@lru_cache(maxsize=1)
def _get_dense_embedder() -> EmbeddingClient:
    return EmbeddingClient()


@lru_cache(maxsize=1)
def _get_sparse_embedder() -> BM25SparseEmbedder:
    """加载语料并训练 BM25"""
    logger.info("正在训练 BM25 模型...")
    docs = DocumentLoader().load_directory(str(ROOT_DIR / "data/documents"))
    chunks = MedicalWorkOrderChunker().split_documents(docs)
    contents = [c.page_content for c in chunks]
    embedder = BM25SparseEmbedder()
    embedder.fit(contents)
    logger.info(f"BM25 训练完成: {len(contents)} 个 chunk")
    return embedder


@lru_cache(maxsize=1)
def _get_store() -> MilvusStore:
    return MilvusStore()


@lru_cache(maxsize=1)
def _get_llm() -> LLMClient:
    return LLMClient()


@lru_cache(maxsize=1)
def _get_reranker() -> Reranker:
    return Reranker()


@lru_cache(maxsize=1)
def _get_query_processor() -> QueryProcessor:
    return QueryProcessor()


@lru_cache(maxsize=1)
def _get_memory() -> MemoryManager:
    return MemoryManager()


@lru_cache(maxsize=1)
def _get_qa_cache() -> QACache:
    return QACache()


# ── RAG Chain ─────────────────────────────────

class RAGChain:
    """
    RAG 问答链：检索 + 生成 + 对话记忆 + 语义缓存。

    用法:
      chain = RAGChain()
      result = chain.ask("CT伪影怎么处理？")
      result = chain.ask("MRI啸叫", mode="keyword")
      result = chain.ask("设备故障", mode="hybrid", ticket_id_filter="GD-2026-03001")
      # 带对话记忆:
      result = chain.ask("那上次那个问题怎么解决？", session_id="sess-001")
    """

    def __init__(self):
        self._dense = _get_dense_embedder()
        self._sparse = _get_sparse_embedder()
        self._store = _get_store()
        self._llm = _get_llm()
        self._memory = _get_memory()
        self._qa_cache = _get_qa_cache()
        self._cfg = get_config()

    # ── 主入口 ──────────────────────────────

    def ask(
        self,
        question: str,
        mode: SearchMode = "hybrid",
        top_k: int = None,
        ticket_id_filter: str = None,
        device_type_filter: str = None,
        rerank: bool = None,
        query_expansion: bool = None,
        output_format: str = None,
        session_id: str = None,
    ) -> dict:
        """
        RAG 问答主入口。

        参数:
          question:          用户问题
          mode:              检索模式 semantic | keyword | hybrid
          top_k:             检索条数（默认取 config.yml）
          ticket_id_filter:  按工单号过滤
          device_type_filter:按设备类型过滤（如"CT 机"、"血液透析机"）
          rerank:            是否启用重排序（默认取 config.yml）
          query_expansion:   是否启用查询扩展（改写+多路召回，默认取 config.yml）
          output_format:     输出格式 "text" | "json"（默认取 config.yml）
          session_id:        会话 ID，传入则启用三级对话记忆

        返回:
          {
            "question": str,           # 原始问题
            "answer":   str,           # LLM 生成回答
            "sources":  list[dict],    # 引用的 chunk 来源
            "mode":     str,           # 检索模式
            "has_answer": bool,        # 是否找到相关工单
            "output_format": str,      # 实际使用的输出格式
            "session_id": str|None,    # 会话 ID（启用记忆时）
          }
        """
        if top_k is None:
            top_k = self._cfg.llm.retrieval_top_k
        if mode is None:
            mode = self._cfg.retrieval.default_mode
        if rerank is None:
            rerank = self._cfg.rerank.enabled
        if query_expansion is None:
            query_expansion = self._cfg.retrieval.query_preprocess.enabled
        if output_format is None:
            output_format = self._cfg.llm.output_format

        # 0. 保存用户消息到记忆系统
        if session_id and self._memory.available:
            self._memory.add_message(session_id, "user", question)

        # 0a. 语义缓存查找 — 相同/相似问题跳过 LLM
        cache_hit, cached_answer = self._qa_cache.lookup(question)
        if cache_hit:
            logger.info(
                f"RAG 缓存命中: question='{question[:40]}...' → 直接返回缓存答案"
            )
            if session_id and self._memory.available:
                self._memory.add_message(session_id, "assistant", cached_answer)
            return {
                "question": question,
                "answer": cached_answer,
                "sources": [],
                "mode": mode,
                "has_answer": True,
                "output_format": output_format,
                "session_id": session_id,
                "from_cache": True,
            }

        # 1. 构建过滤表达式（支持 ticket_id + device_type）
        expr = build_filter_expr(ticket_id=ticket_id_filter, device_type=device_type_filter)

        # 2. 查询预处理（口语→关键词改写 + 多路召回）
        search_query = question
        if query_expansion:
            search_query = self._expand_query(question)

        # 3. 检索（rerank 模式：多召回 → 精排）
        if rerank:
            chunks = self._retrieve_with_rerank(search_query, mode, top_k, expr)
        else:
            chunks = self._retrieve(search_query, mode, top_k, expr)

        # 4. 按分数阈值过滤低相关结果
        score_threshold = self._cfg.retrieval.score_threshold
        chunks = [c for c in chunks if c.get("score", 0.0) >= score_threshold]

        # 5. 构建 Prompt（用原始问题而非改写后的查询）
        system_prompt, user_message = build_full_prompt(
            question, chunks, ticket_id_filter, output_format=output_format,
        )

        # 5a. 注入对话历史记忆（如果有）
        if session_id and user_message is not None:
            memory_context = self._memory.build_memory_prompt(session_id)
            if memory_context:
                user_message = (
                    f"【对话背景】\n{memory_context}\n\n"
                    f"{user_message}"
                )

        # 6. 无结果 → 兜底（text / json 分开处理）
        if user_message is None:
            fallback = (
                get_json_fallback() if output_format == "json"
                else get_fallback_answer()
            )
            if session_id and self._memory.available:
                self._memory.add_message(session_id, "assistant", fallback)
            return {
                "question": question,
                "answer": fallback,
                "sources": [],
                "mode": mode,
                "has_answer": False,
                "output_format": output_format,
                "session_id": session_id,
            }

        # 7. 调用 LLM
        logger.info(
            f"RAG 问答: mode={mode} rerank={rerank} fmt={output_format} "
            f"session={session_id or 'none'} "
            f"qexpand={query_expansion} question='{question[:30]}...' chunks={len(chunks)}"
        )
        answer = self._llm.generate(
            user_message=user_message,
            system_prompt=system_prompt,
        )

        # 7a. 保存助手回答到记忆系统
        if session_id and self._memory.available:
            self._memory.add_message(session_id, "assistant", answer)

        # 7b. 存入语义缓存
        self._qa_cache.store(question, answer)

        # 8. 返回
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "ticket_id": c.get("ticket_id", ""),
                    "content": c.get("content", ""),
                    "source": c.get("source", ""),
                    "score": c.get("score", 0.0),
                }
                for c in chunks
            ],
            "mode": mode,
            "has_answer": True,
            "output_format": output_format,
            "session_id": session_id,
        }

    def ask_stream(
        self,
        question: str,
        mode: SearchMode = "hybrid",
        top_k: int = None,
        ticket_id_filter: str = None,
        device_type_filter: str = None,
        rerank: bool = None,
        query_expansion: bool = None,
        output_format: str = None,
        session_id: str = None,
    ) -> dict:
        """
        RAG 流式问答：检索后用 LLM 逐 token 输出。

        返回:
          {
            "question": str,
            "sources":  list[dict],
            "mode":     str,
            "has_answer": bool,
            "output_format": str,
            "session_id": str|None,
            "stream":   generator → yield text chunks,
          }

        用法:
          result = chain.ask_stream("CT伪影怎么处理？", session_id="sess-001")
          for chunk in result["stream"]:
              print(chunk, end="", flush=True)
        """
        if top_k is None:
            top_k = self._cfg.llm.retrieval_top_k
        if mode is None:
            mode = self._cfg.retrieval.default_mode
        if rerank is None:
            rerank = self._cfg.rerank.enabled
        if query_expansion is None:
            query_expansion = self._cfg.retrieval.query_preprocess.enabled
        if output_format is None:
            output_format = self._cfg.llm.output_format

        # 0. 保存用户消息到记忆系统
        if session_id and self._memory.available:
            self._memory.add_message(session_id, "user", question)

        # 0a. 语义缓存查找
        cache_hit, cached_answer = self._qa_cache.lookup(question)
        if cache_hit:
            logger.info(
                f"RAG 流式缓存命中: question='{question[:40]}...' → 直接返回"
            )
            if session_id and self._memory.available:
                self._memory.add_message(session_id, "assistant", cached_answer)
            def _cache_stream():
                yield cached_answer
            return {
                "question": question,
                "sources": [],
                "mode": mode,
                "has_answer": True,
                "output_format": output_format,
                "session_id": session_id,
                "from_cache": True,
                "stream": _cache_stream(),
            }

        expr = build_filter_expr(ticket_id=ticket_id_filter, device_type=device_type_filter)

        search_query = question
        if query_expansion:
            search_query = self._expand_query(question)

        if rerank:
            chunks = self._retrieve_with_rerank(search_query, mode, top_k, expr)
        else:
            chunks = self._retrieve(search_query, mode, top_k, expr)

        score_threshold = self._cfg.retrieval.score_threshold
        chunks = [c for c in chunks if c.get("score", 0.0) >= score_threshold]

        system_prompt, user_message = build_full_prompt(
            question, chunks, ticket_id_filter, output_format=output_format,
        )

        # 注入对话历史记忆
        if session_id and user_message is not None:
            memory_context = self._memory.build_memory_prompt(session_id)
            if memory_context:
                user_message = (
                    f"【对话背景】\n{memory_context}\n\n"
                    f"{user_message}"
                )

        # 无结果 → 兜底
        if user_message is None:
            def _fallback_stream():
                yield get_json_fallback() if output_format == "json" else get_fallback_answer()

            if session_id and self._memory.available:
                fb = get_json_fallback() if output_format == "json" else get_fallback_answer()
                self._memory.add_message(session_id, "assistant", fb)

            return {
                "question": question,
                "sources": [],
                "mode": mode,
                "has_answer": False,
                "output_format": output_format,
                "session_id": session_id,
                "stream": _fallback_stream(),
            }

        # 有结果 → 流式 LLM
        logger.info(
            f"RAG 流式问答: mode={mode} rerank={rerank} fmt={output_format} "
            f"session={session_id or 'none'} "
            f"qexpand={query_expansion} question='{question[:30]}...' chunks={len(chunks)}"
        )

        stream_gen = self._llm.generate_stream(
            user_message=user_message,
            system_prompt=system_prompt,
        )

        # 包装流式生成器，流结束后：保存到记忆 + 存入语义缓存
        _qa_cache = self._qa_cache
        _qa_question = question
        _memory = self._memory
        _sid = session_id

        def _stream_wrapper():
            full = ""
            for chunk in stream_gen:
                full += chunk
                yield chunk
            if full:
                if _sid and _memory.available:
                    _memory.add_message(_sid, "assistant", full)
                _qa_cache.store(_qa_question, full)

        final_stream = _stream_wrapper()

        return {
            "question": question,
            "sources": [
                {
                    "ticket_id": c.get("ticket_id", ""),
                    "content": c.get("content", ""),
                    "source": c.get("source", ""),
                    "score": c.get("score", 0.0),
                }
                for c in chunks
            ],
            "mode": mode,
            "has_answer": True,
            "output_format": output_format,
            "session_id": session_id,
            "stream": final_stream,
        }

    # ── 查询预处理 ──────────────────────────

    def _expand_query(self, question: str) -> str:
        """
        查询扩展：口语改写为检索友好的关键词。

        启用条件: config.yml → retrieval.query_preprocess.enabled
        策略:
          - rewrite: LLM 将口语改写为关键词组合（默认开启）
          - multi_query: 生成多个变体各自检索合并（默认关闭，延迟较高）

        注意: 改写后的查询只用于检索，LLM 生成仍用原始问题。
        """
        proc = _get_query_processor()
        pre_cfg = self._cfg.retrieval.query_preprocess

        expanded = question
        if pre_cfg.rewrite:
            expanded = proc.rewrite(question)

        # multi_query 暂不启用（延迟较高，保留扩展点）
        # if pre_cfg.multi_query:
        #     variants = proc.expand(expanded, n=pre_cfg.multi_query_n)
        #     ... merge results ...

        if expanded != question:
            logger.info(f"Query expanded: '{question[:30]}...' → '{expanded[:60]}...'")
        return expanded

    # ── 检索分发 ────────────────────────────

    def _retrieve_with_rerank(
        self, question: str, mode: SearchMode, top_k: int, expr: str,
    ) -> list[dict]:
        """
        两阶段检索：粗召回（多）→ 重排序（精）。

        Stage 1 — 粗召回: 检索 top_k × oversample_factor 条候选
        Stage 2 — 精排:   qwen3-rerank 交叉编码器对候选逐一打分，取 top_k
        """
        factor = self._cfg.rerank.oversample_factor
        oversample_k = min(top_k * factor, 20)  # 最多召回 20 条，避免超过 reranker 限制

        # Stage 1: 粗召回
        candidates = self._retrieve(question, mode, oversample_k, expr)
        if len(candidates) <= top_k:
            return candidates  # 候选不够，跳过 rerank

        # Stage 2: 精排
        reranker = _get_reranker()
        reranked = reranker.rerank(question, candidates, top_n=top_k)

        return reranked

    def _retrieve(
        self, question: str, mode: SearchMode, top_k: int, expr: str,
    ) -> list[dict]:
        """根据模式分发到不同检索方法"""
        if mode == "semantic":
            return self._retrieve_semantic(question, top_k, expr)
        elif mode == "keyword":
            return self._retrieve_keyword(question, top_k, expr)
        else:  # hybrid
            return self._retrieve_hybrid(question, top_k, expr)

    def _retrieve_semantic(self, question, top_k, expr) -> list[dict]:
        query_vec = self._dense.embed(question)
        return self._store.search(query_vec, top_k=top_k, expr=expr)

    def _retrieve_keyword(self, question, top_k, expr) -> list[dict]:
        query_sparse = self._sparse.encode_queries([question])[0]
        return self._store.sparse_search(query_sparse, top_k=top_k, expr=expr)

    def _retrieve_hybrid(self, question, top_k, expr) -> list[dict]:
        query_dense = self._dense.embed(question)
        query_sparse = self._sparse.encode_queries([question])[0]
        return self._store.hybrid_search(
            query_dense, query_sparse,
            top_k=top_k,
            dense_weight=self._cfg.retrieval.hybrid_dense_weight,
            sparse_weight=self._cfg.retrieval.hybrid_sparse_weight,
            expr=expr,
        )
