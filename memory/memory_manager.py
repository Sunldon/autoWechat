import logging
import os
import re
from collections import OrderedDict, deque
from typing import Optional

# 抑制 mem0 chroma 的冗余 INFO 日志
logging.getLogger("mem0.vector_stores.chroma").setLevel(logging.WARNING)

from rank_bm25 import BM25Okapi

from memory.short_term_memory import ShortTermMemory
from logger import get_logger
logger = get_logger(__name__)

# ============ 强制离线：禁止 HuggingFace 联网下载 ============
os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HUGGINGFACE_HUB_ENDPOINT", None)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ============ BM25 增量更新通道 ============
# _patched_add 提取出记忆文本后存入这里，store_context 消费后加入 BM25 索引
_pending_bm25_texts: dict[str, list[str]] = {}

# ============ Monkey-patch: Mem0 兼容 LM Studio ============
import mem0.llms.openai as _mem0_llm
import mem0.memory.main as _mem0_main

# _original_generate = _mem0_llm.OpenAILLM.generate_response


# def _patched_generate(
#     self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs
# ):
#     if response_format and isinstance(response_format, dict):
#         if response_format.get("type") == "json_object":
#             response_format = None
#     kwargs["max_tokens"] = 8192
#     return _original_generate(
#         self, messages, response_format, tools, tool_choice, **kwargs
#     )


# _mem0_llm.OpenAILLM.generate_response = _patched_generate

# Patch _add_to_vector_store: LLM extraction 失败时仍保存消息
_original_add = _mem0_main.Memory._add_to_vector_store


def _safe_parse_json(text):
    """解析 LLM 的 JSON 输出，自动修复截断问题。

    Mem0 的提取 prompt 可能因上下文过长导致 LLM 输出被截断，
    最后的 JSON 对象/数组不完整。此函数从右侧搜索最后一个
    完整 '}'，尝试解析出部分记忆。
    """
    import json as _json

    text = text.strip()
    # 1. 直接解析
    try:
        return _json.loads(text, strict=False)
    except _json.JSONDecodeError:
        pass

    # 2. 从右往左逐个尝试每个 '}'，补齐缺失的 ']' 和 ']}'
    decoder = _json.JSONDecoder(strict=False)
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "}":
            for suffix in ["", "]", "]}"]:
                candidate = text[: i + 1] + suffix
                try:
                    obj, _ = decoder.raw_decode(candidate)
                    if isinstance(obj, dict) and "memory" in obj:
                        logger.warning(
                            f"LLM 输出截断，已恢复 {len(obj.get('memory', []))} 条记忆"
                        )
                        return obj
                except _json.JSONDecodeError:
                    continue
    return None


def _patched_add(self, messages, metadata, filters, infer, prompt=None):
    import traceback as _tb
    import json

    if not infer:
        return _original_add(self, messages, metadata, filters, infer, prompt)

    from mem0.memory.utils import parse_messages
    from mem0.configs.prompts import (
        generate_additive_extraction_prompt,
        ADDITIVE_EXTRACTION_PROMPT,
        AGENT_CONTEXT_SUFFIX,
    )

    session_scope = None
    try:
        # Phase 0: Context gathering
        session_scope = _build_session_scope(filters)
        last_messages_raw = self.db.get_last_messages(session_scope, limit=10)
        parsed_messages = parse_messages(messages)
        # 补充 "other" 角色：Mem0 原生 parse_messages 不识别此角色
        user_id_name = filters.get("user_id", "对方")
        for msg in messages:
            if msg.get("role") == "other":
                parsed_messages += f"{user_id_name}: {msg['content']}\n"

        # Phase 1: Search existing
        search_filters = {
            k: v
            for k, v in filters.items()
            if k in ("user_id", "agent_id", "run_id") and v
        }
        query_embedding = self.embedding_model.embed(parsed_messages, "search")
        existing_results = self.vector_store.search(
            query=parsed_messages,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )
        existing_memories = [
            {"id": str(i), "text": m.payload.get("data", "")}
            for i, m in enumerate(existing_results)
        ]

        # Phase 2: LLM extraction (may fail, carry on)
        from mem0.memory.utils import remove_code_blocks, extract_json

        extracted_memories = []
        try:
            is_agent_scoped = bool(filters.get("agent_id")) and not filters.get(
                "user_id"
            )
            sp = ADDITIVE_EXTRACTION_PROMPT + (
                AGENT_CONTEXT_SUFFIX if is_agent_scoped else ""
            )
            # 添加微信对话场景说明，让 LLM 理解 role=user 和 role=other 的含义
            scene_hint = (
                "【对话场景说明】这是一个微信双人聊天记录。"
                "role='user' 代表用户自己（我），role='other' 代表聊天对方。"
                "请同时提取关于双方的个人特征以及双方之间的关系性事实。\n\n"
            )
            up = scene_hint + generate_additive_extraction_prompt(
                existing_memories=existing_memories,
                new_messages=parsed_messages,
                last_k_messages=last_messages_raw,
                custom_instructions=prompt or self.custom_instructions,
            )
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": sp},
                    {"role": "user", "content": up},
                ],
                response_format=None,
            )
            if response:
                clean = remove_code_blocks(response)
                if clean and clean.strip():
                    parsed = _safe_parse_json(clean)
                    if parsed:
                        extracted_memories = parsed.get("memory", [])
                    else:
                        logger.warning("LLM 输出无法解析为 JSON，已跳过提取")
        except Exception as e:
            logger.error(f"LLM 提取记忆解析失败: {e}")
            pass

        # ==================== 核心修复：正确的存储分流逻辑 ====================

        # 无论如何，先把本次原始微信对话存入 SQLite 关系表（供下次提取作上下文参考）
        self.db.save_messages(messages, session_scope)

        # 情况 1：大模型偷懒，什么都没提取出来
        if not extracted_memories:
            logger.debug("LLM extraction returned no memories")
            # 如果你希望【提取失败时】把原始对话强行当成记忆存入 ChromaDB 兜底，就留着下面这行；
            # 如果你觉得无价值的闲聊不配进 ChromaDB，直接返回 [] 即可。
            # return _original_add(self, messages, metadata, filters, False, prompt)
            return []

        # 情况 2：成功提取到了高价值的长期记忆事实！
        logger.info(f"LLM 成功提取到 {len(extracted_memories)} 条长期记忆，开始写入 ChromaDB..."
        )

        # ==================== 去重：按文本内容去重，避免 LLM 重复返回相同事实 ====================
        seen_texts = set()
        unique_memories = []
        for mem_item in extracted_memories:
            mem_text = mem_item.get("text", "")
            if mem_text and mem_text not in seen_texts:
                seen_texts.add(mem_text)
                unique_memories.append(mem_item)
        if len(unique_memories) < len(extracted_memories):
            logger.debug(f"去重: {len(extracted_memories)} -> {len(unique_memories)} 条唯一记忆"
            )
        extracted_memories = unique_memories
        # ============================================================

        stored_results = []
        for mem_item in extracted_memories:
            mem_text = mem_item.get("text", "")
            if mem_text:
                # ==================== 核心修复点 ====================
                # 将纯文本包装成 Mem0 标准的格式，防止它把字符串拆成字符
                formatted_message = [{"role": "user", "content": mem_text}]
                logger.debug(f"准备存储的记忆文本: {mem_text}")
                # 传入包装后的 formatted_message
                res = _original_add(
                    self,
                    formatted_message,
                    metadata,
                    filters,
                    infer=False,
                    prompt=prompt,
                )
                # ====================================================
                stored_results.append(res)

        # ==================== 将提取出的记忆文本送到 BM25 增量通道 ====================
        global _pending_bm25_texts
        user_id = filters.get("user_id", "")
        if user_id and extracted_memories:
            texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
            if texts:
                _pending_bm25_texts.setdefault(user_id, []).extend(texts)

        return stored_results

    except Exception as e:
        logger.error(f"_patched_add failed: {e}")
        _tb.print_exc()
        return []


# Need _build_session_scope reference
from mem0.memory.main import _build_session_scope

_mem0_main.Memory._add_to_vector_store = _patched_add

# ============ Mem0 配置（从 config.yaml 加载）============
import config as _app_config

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_mem0_config() -> dict:
    """从 config.MEMORY_CONFIG 构建 Mem0 原生格式的配置"""
    mc = _app_config.MEMORY_CONFIG

    embedder_cfg = mc.get("embedder", {})
    embed_model = embedder_cfg.get("model", "./models/BAAI/bge-m3")
    embed_model_abs = (
        embed_model
        if os.path.isabs(embed_model)
        else os.path.join(_BASE, embed_model.lstrip("./\\"))
    )

    vs_cfg = mc.get("vector_store", {})
    vs_path = vs_cfg.get("path", "./chat_db")
    vs_path_abs = (
        vs_path
        if os.path.isabs(vs_path)
        else os.path.join(_BASE, vs_path.lstrip("./\\"))
    )

    reranker_cfg = mc.get("search", {}).get("reranker", {})
    reranker_model = reranker_cfg.get("model", "BAAI/bge-reranker-v2-m3")
    reranker_model_abs = (
        reranker_model
        if os.path.isabs(reranker_model)
        else os.path.join(_BASE, "models", "BAAI", reranker_model.split("/")[-1])
    )

    return {
        "llm": {
            "provider": mc.get("llm", {}).get("provider", "openai"),
            "config": {
                "model": mc.get("llm", {}).get("model", "qwen3.5-9b"),
                "openai_base_url": mc.get("llm", {}).get(
                    "openai_base_url", "http://localhost:1234/v1"
                ),
                "api_key": mc.get("llm", {}).get("api_key", "not-needed"),
                "max_tokens": 8192,
            },
        },
        "embedder": {
            "provider": embedder_cfg.get("provider", "huggingface"),
            "config": {
                "model": embed_model_abs,
                "model_kwargs": {"device": embedder_cfg.get("device", "cpu")},
                "embedding_dims": embedder_cfg.get("embedding_dims", 1024),
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": vs_cfg.get("collection_name", "wechat_history"),
                "path": vs_path_abs,
            },
        },
        "reranker": {
            "provider": "sentence_transformer",
            "config": {
                "model": reranker_model_abs,
                "device": "cpu",
                "batch_size": 32,
                "show_progress_bar": False,
            },
        },
        "custom_instructions": (
            "1. 必须使用【中文】提取和记录所有记忆事实。\n"
            "2. 降低提取门槛：只要涉及用户的喜好、职业、日常活动、正在做的事情、"
            "情绪状态、关系人等，都应当提炼为一条具体的、短小精悍的事实陈述。\n"
            "3. 【双方特征提取】必须同时提取【用户自己（role=user）】和【聊天对方（role=other）】"
            "的个人特征，不能只关注一方。\n"
            "4. 【关系性事实】如果双方就某个话题表达了相同或不同的意见/经历，"
            "应当提取为关系性事实（如'两人都喜欢吃辣'、'两人对某电影看法不同'）。\n"
        ),
    }


DEFAULT_MEM0_CONFIG = _build_mem0_config()


# ==================== BM25 检索：rank_bm25 包装 ====================
def _tokenize(text: str) -> list[str]:
    """中文文本的字符 Bigram + Unigram 分词，用于 BM25 关键词匹配。

    无需外部分词器（jieba），通过字符级 n-gram 实现关键词匹配。
    例如 "公众号" → ['公', '众', '号', '公众', '众号']
    """
    chars = list(text)
    unigrams = chars
    bigrams = ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]
    return unigrams + bigrams


class IncrementalBM25:
    """包装 rank_bm25.BM25Okapi，支持惰性重建的增量更新。

    - 首次 build() 从全部记忆构建完整索引
    - add_document() 标记脏，下次 search() 前自动重建
    - 避免每次检索都重复 get_all_memories
    """

    def __init__(self):
        self._corpus: list[str] = []  # 原始文本
        self._tokenized: list[list[str]] = []  # 分词后
        self._bm25: BM25Okapi | None = None
        self._dirty = False

    def build(self, corpus_texts: list[str]) -> None:
        """全量重建 BM25 索引"""
        self._corpus = list(corpus_texts)
        self._tokenized = [_tokenize(t) for t in self._corpus]
        self._bm25 = BM25Okapi(self._tokenized)
        self._dirty = False

    def add_document(self, text: str) -> None:
        """添加单条文档，标记下次检索前重建"""
        self._corpus.append(text)
        self._tokenized.append(_tokenize(text))
        self._dirty = True

    def mark_dirty(self) -> None:
        """手动标记为脏（外部新增了记忆但不知道具体文本时使用）"""
        self._dirty = True

    def _ensure_ready(self) -> None:
        if self._dirty and self._corpus:
            self._bm25 = BM25Okapi(self._tokenized)
            self._dirty = False

    @property
    def is_empty(self) -> bool:
        return len(self._corpus) == 0

    @property
    def size(self) -> int:
        return len(self._corpus)

    def search(self, query_text: str, top_k: int = 10) -> list[dict]:
        """BM25 搜索，返回 [{"memory": str, "bm25_score": float}, ...]"""
        self._ensure_ready()
        if not self._bm25 or not self._corpus:
            return []
        query_tokens = _tokenize(query_text)
        scores = self._bm25.get_scores(query_tokens)
        # 按分数降序
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in indexed:
            if score > 0 and len(results) < top_k:
                results.append(
                    {
                        "memory": self._corpus[idx],
                        "bm25_score": float(score),
                    }
                )
        return results


class MemoryManager:
    """记忆编排层——所有外部代码只与这个类交互"""

    def __init__(
        self,
        mem0_config: Optional[dict] = None,
        window_size: int = 10,
    ):
        self.mem0 = self._init_mem0(mem0_config or DEFAULT_MEM0_CONFIG)
        self.short_term = ShortTermMemory(window_size=window_size)
        # 每个 user_id 维护一个已存储消息的指纹集合（自动去重 + 有界）
        self._stored_fingerprints: dict[str, OrderedDict] = {}
        # 指纹集合上限（覆盖屏幕上可见的消息数，50 绰绰有余）
        self._max_fingerprints = 50

        # ===== 检索增强配置 =====
        sc = _app_config.MEMORY_SEARCH_CONFIG
        self._hyde_max_chars = sc.get("hyde_max_chars", 3)  
        self._hybrid_enabled = sc.get("hybrid_search", True)  
        self._bm25_weight = sc.get("bm25_weight", 0.3)  
        self._score_threshold = sc.get("score_threshold", 0.96)  
        self._top_k = sc.get("top_k", 5)
        self._reranker_cfg = sc.get("reranker", {})  
        # BM25 索引（惰性构建 + 增量更新）
        self._bm25_models: dict[str, IncrementalBM25] = {}

        logger.info(f"MemoryManager 初始化完成 | "
            f"HyDE跳过≤{self._hyde_max_chars}字 | "
            f"混合检索={'开' if self._hybrid_enabled else '关'} | "
            f"分数阈值={self._score_threshold}"
        )

    def _make_fingerprint(self, msg: dict) -> int:
        """生成消息指纹：(sender, text) 的 hash"""
        return hash((msg.get("sender", ""), msg.get("text", "")))

    def _init_mem0(self, config: dict):
        """初始化 Mem0，失败时静默降级"""
        try:
            from mem0 import Memory

            memory = Memory.from_config(config)
            logger.info("Mem0 初始化成功 (collection: wechat_history)")
            return memory
        except ImportError:
            logger.warning("mem0ai 未安装，长期记忆功能不可用")
            return None
        except Exception as e:
            logger.warning(f"Mem0 初始化失败: {e}，长期记忆功能不可用")
            return None

    def _make_llm_client(self):
        """复用 config 的 LLM 配置创建 OpenAI 兼容客户端，用于 HyDE 等辅助 LLM 调用"""
        from openai import OpenAI

        mc = _app_config.MEMORY_CONFIG.get("llm", {})
        base_url = mc.get("openai_base_url", "http://localhost:1234/v1")
        model = mc.get("model", "qwen3.5-9b")
        api_key = mc.get("api_key", "not-needed")
        client = OpenAI(base_url=base_url, api_key=api_key)
        client.model = model
        return client

    # ==================== 方案A：HyDE 短查询跳过 ====================
    def _should_use_hyde(self, query: str) -> bool:
        """判断是否应该对当前查询使用 HyDE 改写。

        简短关键词（专有名词、人名、产品名等）不使用 HyDE，
        因为 HyDE 会脑补出不相干的语义，污染向量检索。

        规则：
        1. 纯英文/数字关键词（如 "uv", "BM25"）→ 跳过
        2. 中文字数 ≤ self._hyde_max_chars → 跳过
        3. 总字符数 ≤ 3 的非中文 → 跳过
        """
        stripped = query.strip()
        if not stripped:
            return False

        # 统计中文字符数
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", stripped))
        total_chars = len(stripped)

        # 规则1: 纯英文/数字关键词（无中文）
        if chinese_chars == 0 and total_chars <= 10:
            return False

        # 规则2: 中文字数太少
        if 0 < chinese_chars <= self._hyde_max_chars:
            return False

        # 规则3: 总字符太少（包含标点等）
        if total_chars <= 3:
            return False

        return True

    def _hyde_rewrite(self, query: str, user_id: str) -> str:
        """HyDE：用 LLM 生成假设文档，弥补短查询的语义鸿沟

        Args:
            query: 原始用户消息（如 "晚上打球吗"）
            user_id: 联系人名称

        Returns:
            hyde_query: 改写后的查询（原始查询+假设文档拼接），
                        如果跳过 HyDE 则直接返回原始查询
        """
        
        if not self._should_use_hyde(query):
            logger.debug(f"[HyDE] 跳过（短关键词）: {query}")
            return query

        try:
            llm = self._make_llm_client()
            hyde_prompt = (
                "你是一个记忆检索助手。给定一条微信聊天消息，请想象一下："
                "如果要检索与该消息相关的用户长期记忆，这些记忆可能包含哪些事实？\n\n"
                "请用一段陈述性文字描述这些潜在相关记忆的内容，不要写对话或评价，"
                "一句话直接输出事实陈述。\n\n"
                f"聊天消息: {query}\n"
                "相关记忆:"
            )
            response = llm.chat.completions.create(
                model=llm.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个记忆检索助手，输出简洁的相关事实描述。",
                    },
                    {"role": "user", "content": hyde_prompt},
                ],
                max_tokens=256,
                temperature=0.3,
            )
            hyde_doc = response.choices[0].message.content.strip()
            if hyde_doc:
                hyde_query = f"{query}\n\n{hyde_doc}"
                logger.debug(f"[HyDE] 原始查询: {query}")
                logger.debug(f"[HyDE] 假设文档: {hyde_doc}")
                return hyde_query
        except Exception as e:
            logger.warning(f"[HyDE] 生成失败，回退原始查询: {e}")
        return query

    # ==================== 方案B：rank_bm25 关键词检索 ====================
    def _bm25_search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        """对用户所有记忆执行 BM25 关键词检索，返回排序后的结果。

        使用 rank_bm25（BM25Okapi 算法），惰性构建 + 增量更新。
        - 首次调用时从 mem0 拉取全部记忆建索引
        - 后续调用仅在有新记忆存储时重建
        """
        # 1. 延迟创建用户模型
        if user_id not in self._bm25_models:
            self._bm25_models[user_id] = IncrementalBM25()

        model = self._bm25_models[user_id]

        # 2. 首次构建（跨 Session 首次使用时从数据库全量拉取）
        if model.is_empty:
            memories = self.get_all_memories(user_id)
            corpus = [m.get("memory", "") for m in memories if m.get("memory")]
            if corpus:
                model.build(corpus)
                logger.debug(f"  BM25索引 [{user_id}]: 构建完成，{len(corpus)} 篇文档")

        # 3. 搜索
        if model.is_empty:
            return []
        return model.search(query, top_k=top_k)

    # ==================== 方案B：RRF 融合 ====================
    @staticmethod
    def _rrf_fusion(
        vector_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
        bm25_weight: float = 0.3,
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion：融合向量检索和 BM25 结果

        Args:
            vector_results: mem0 搜索结果 [{"memory": str, "score": float}, ...]
            bm25_results: BM25 搜索结果 [{"memory": str, "bm25_score": float}, ...]
            top_k: 返回前 K 条
            bm25_weight: BM25 在融合中的权重（0~1），越大越偏向关键词匹配
            k: RRF 常数（通常 60）
        Returns:
            融合排序后的结果 [{"memory": str, "fusion_score": float}, ...]
        """
        # 用 memory 文本作为唯一标识
        doc_ranks: dict[str, float] = {}

        # 向量排名贡献
        for rank, item in enumerate(vector_results):
            text = item.get("memory", "")
            if text:
                doc_ranks[text] = doc_ranks.get(text, 0) + 1.0 / (k + rank)

        # BM25 排名贡献（加权）
        for rank, item in enumerate(bm25_results):
            text = item.get("memory", "")
            if text:
                doc_ranks[text] = doc_ranks.get(text, 0) + bm25_weight / (k + rank)

        # 按 fusion score 排序
        sorted_docs = sorted(doc_ranks.items(), key=lambda x: x[1], reverse=True)
        for text, score in sorted_docs:
            logger.debug(f"  {text}: {score:.4f}")
        return [
            {"memory": text, "fusion_score": score}
            for text, score in sorted_docs[:top_k]
        ]

    # ==================== 方案C：Mem0 Cross-Encoder Reranker ====================
    def _mem0_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """使用 mem0 内置的 Cross-Encoder 精排模型对候选记忆重排序。

        复用 mem0 配置中加载的 reranker 实例，避免重复加载模型。
        """
        if not self._reranker_cfg.get("enabled", False):
            return candidates
        if not self.mem0 or not self.mem0.reranker:
            logger.debug("[Reranker] mem0 无 reranker 实例，跳过精排")
            return candidates

        try:
            reranked = self.mem0.reranker.rerank(
                query, candidates, top_k=len(candidates)
            )
            logger.debug(f"[Reranker] 重排序: {len(candidates)} → {len(reranked)} 条")
            return reranked
        except Exception as e:
            logger.warning(f"[Reranker] 精排失败: {e}")
            return candidates

    # ==================== 核心检索管道 ====================
    def _hybrid_search(
        self,
        query: str,
        user_id: str,
        search_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> list[dict]:
        """完整的混合检索管道：向量检索 + BM25 + RRF 融合 + Reranker

        Args:
            query: 检索查询
            user_id: 联系人名称
            search_top_k: 初始检索的候选数量（越大召回越高），默认 self._top_k * 4
            final_top_k: 最终返回数量，默认 self._top_k

        Returns:
            排序后的记忆列表 [{"memory": str, ...}]
        """
        search_top_k = search_top_k or self._top_k * 4
        final_top_k = final_top_k or self._top_k

        if not self.mem0:
            return []

        try:
            # 1. 向量检索（Mem0，低阈值以召回更多候选让 reranker 精排）
            memories = self.mem0.search(
                query,
                filters={"user_id": user_id},
                top_k=search_top_k,
                threshold=0.0,
            )
            vector_results = []
            if memories and isinstance(memories, dict):
                results = memories.get("results", [])
                for mem in results:
                    if isinstance(mem, dict):
                        text = mem.get("memory", "")
                        score = mem.get("score", 0.0)
                        if text:
                            vector_results.append({"memory": text, "score": score})
                            logger.debug(f"  向量 [{user_id}]: score={score:.4f} | {text}")

            # 2. BM25 关键词检索（方案B）
            bm25_results = []
            if self._hybrid_enabled:
                bm25_results = self._bm25_search(query, user_id, top_k=search_top_k)
                if bm25_results:
                    logger.debug(f"  BM25 [{user_id}]: 命中 {len(bm25_results)} 条关键词匹配")
            for bm25 in bm25_results:
                logger.debug(f"  BM25 [{user_id}]: score={bm25['bm25_score']:.4f} | {bm25['memory']}"
                )

            # 3. RRF 融合（方案B）
            if self._hybrid_enabled and bm25_results:
                fused = self._rrf_fusion(
                    vector_results,
                    bm25_results,
                    top_k=search_top_k,
                    bm25_weight=self._bm25_weight,
                )
                logger.debug(f"  RRF融合 [{user_id}]: {len(vector_results)}向量 + "
                    f"{len(bm25_results)}BM25 → {len(fused)}候选"
                )
            else:
                # 纯向量检索
                fused = [
                    {"memory": m["memory"], "fusion_score": m["score"]}
                    for m in vector_results[:search_top_k]
                ]

            # 4. 送入精排
            candidates = fused[:search_top_k]

            # 5. Mem0 Cross-Encoder 精排
            reranked = self._mem0_rerank(query, candidates)

            # 最终根据 Cross-Encoder 的置信度进行过滤
            filtered_results = [m for m in reranked if m.get("rerank_score", 0) >= 0.1]
            results = filtered_results if filtered_results else reranked[:2]
            return results[:final_top_k]

        except Exception as e:
            import traceback as _tb

            logger.error(f"混合检索失败: {e}")
            _tb.print_exc()
            return []

    def search(self, query: str, user_id: str, top_k: int = 5, search_top_k: int | None = None, use_hyde: bool = True) -> list[dict]:
        """外部调用的完整检索管道：HyDE → 向量检索 + BM25 → RRF → Reranker → Top-K

        Args:
            query: 用户原始查询
            user_id: 联系人名称
            top_k: 返回的记忆条数
            search_top_k: 初始检索候选数（越大召回越高，但 reranker 越慢）
            use_hyde: 是否使用 HyDE 查询改写

        Returns:
            排序后的记忆列表
        """
        search_query = self._hyde_rewrite(query, user_id) if use_hyde else query
        return self._hybrid_search(
            search_query,
            user_id,
            search_top_k=search_top_k if search_top_k is not None else max(top_k * 6, 30),
            final_top_k=top_k,
        )

    def read_context(self, current_msg: str, user_id: str) -> str:
        """『读』阶段：任务开始前加载所有相关记忆

        Args:
            current_msg: 当前对方的最新消息文本（用于语义检索）
            user_id: 联系人名称

        Returns:
            格式化的上下文文本，供注入 system prompt
        """
        # 0. HyDE 查询改写（方案A 控制是否跳过）
        search_query = self._hyde_rewrite(current_msg, user_id)

        parts = []

        # 1. 混合检索：向量 + BM25 + Reranker（方案B + 方案C）
        results = self._hybrid_search(search_query, user_id)

        memory_texts = []
        for mem in results:
            text = mem.get("memory", "")
            score = mem.get("fusion_score", mem.get("score", 0))
            rerank = mem.get("rerank_score", None)
            score_str = f"{score:.4f}"
            if rerank is not None:
                score_str += f" (rerank={rerank:.4f})"
            logger.debug(f"  最终记忆 [{user_id}]: score={score_str} | {text[:50]}")
            if text:
                memory_texts.append(text)

        if memory_texts:
            parts.append("[长期记忆]\n" + "\n".join(memory_texts))
            logger.info(f"检索完成 [{user_id}]: 注入记忆：{memory_texts}")
        else:
            logger.info(f"检索完成 [{user_id}]: 无相关记忆")

        # 2. 短期记忆摘要
        summary = self.short_term.get_summary(user_id)
        if summary:
            parts.append(summary)

        logger.debug(f"parts: {parts}")
        return "\n\n".join(parts) if parts else ""

    def store_context(self, messages: list[dict], user_id: str):
        """『写』阶段：任务结束后存储新记忆（按内容去重）

        Args:
            messages: 当前轮次的对话消息列表，格式 [{"sender": "self"/"other", "text": "..."}]
            user_id: 联系人名称
        """
        # 初始化指纹集合
        if user_id not in self._stored_fingerprints:
            self._stored_fingerprints[user_id] = OrderedDict()
        
        stored = self._stored_fingerprints[user_id]

        new_messages = []        
        for msg in messages:
            fp = self._make_fingerprint(msg)
            if fp not in stored:
                new_messages.append(msg)
                stored[fp] = True
                # 如果超过上限，O(1) 弹出最老的一条 (last=False 表示 FIFO)
                if len(stored) > self._max_fingerprints:
                    stored.popitem(last=False)

        if not new_messages:
            logger.debug(f"无新消息需要存储 [{user_id}]")
            return

        logger.info(f"增量存储 [{user_id}]: {len(messages)} 条总量 → "
            f"跳过 {len(messages) - len(new_messages)} 条已存 → "
            f"新增 {len(new_messages)} 条"
        )

        # 1. Mem0 存储对话 + 自动提取实体/事实
        if self.mem0:
            try:
                conv_messages = []
                for msg in new_messages:
                    conv_messages.append(
                        {
                            "role": "other" if msg.get("sender") == "other" else "user",
                            "content": msg.get("text", ""),
                        }
                    )
                if conv_messages:
                    self.mem0.add(conv_messages, user_id=user_id)
                    logger.info(f"Mem0 存储完成 [{user_id}]: {len(conv_messages)} 条新消息")
                    # BM25 增量更新：从刚提取的记忆中直接添加到索引
                    global _pending_bm25_texts
                    texts = _pending_bm25_texts.pop(user_id, [])
                    if texts:
                        if user_id not in self._bm25_models:
                            self._bm25_models[user_id] = IncrementalBM25()
                        model = self._bm25_models[user_id]
                        for text in texts:
                            model.add_document(text)
                        logger.debug(f"  BM25增量 [{user_id}]: 新增 {len(texts)} 篇文档")
            except Exception as e:
                logger.error(f"Mem0 存储失败: {e}")

        # 2. 更新短期记忆窗口（也只用新消息）
        self.short_term.update(new_messages, user_id)
        logger.info(f"短期记忆更新完成 [{user_id}]: {len(new_messages)} 条新消息")

    def get_all_memories(self, user_id: str, top_k: int = 1000) -> list:
        """获取用户的所有记忆（用于调试）"""
        if not self.mem0:
            return []
        try:
            results = self.mem0.get_all(filters={"user_id": user_id}, top_k=top_k)
            if isinstance(results, dict):
                return results.get("results", [])
            return results if isinstance(results, list) else []
        except Exception as e:
            logger.error(f"获取全部记忆失败: {e}")
            return []

