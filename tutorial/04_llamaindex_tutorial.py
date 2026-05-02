"""
===========================================================
LlamaIndex 入门教程
===========================================================
本教程承接 01~03（LangChain / LangGraph / Milvus）。前三份让你能调模型、
能编排状态图、能把向量存进库；本教程教你 LlamaIndex —— 一个**专为 RAG**
设计的数据框架。它和 LangChain 不是替代关系，而是分工不同：

  · LangChain 把"调模型 + 写链 + 写 Agent"做成统一接口；
  · LlamaIndex 把"喂数据 → 切片 → 建索引 → 检索 → 答题"这条 RAG 流水线
    做成一等公民，并且在以下场景里写起来比 LangChain 优雅一个数量级：

      ① 多索引路由（自动选 docs 索引 / code 索引 / 摘要索引）
      ② 复杂问题自动分解成子问题（SubQuestionQueryEngine）
      ③ 层次检索：先看摘要、命中后下钻到 chunk（RecursiveRetriever）
      ④ 文档 / 节点 / 索引 / 检索器 / 查询引擎 五层抽象，可单独换零件

LlamaIndex 的核心五层抽象（一定要先记住）：

      Document  →  Node  →  Index  →  Retriever  →  QueryEngine
       (原始)    (切片后) (索引后)   (返回 Node)  (返回 Answer)

  · Document    ：一份原始资料 + 元数据
  · Node        ：Document 切完片后的一段（带父子关系）
  · Index       ：对一组 Node 建立的"可检索结构"（向量 / 摘要 / 关键词 / 图）
  · Retriever   ：给一个 query，吐出 top-k 个 Node
  · QueryEngine ：给一个 query，吐出"答案 + 引用"（=Retriever + LLM 合成）

本文件按 Demo 顺序看完，你能：
  Demo 1  5 行代码跑通最小 RAG（VectorStoreIndex）
  Demo 2  显式拆 Document → Node → Index：看清 Demo 1 隐藏了什么
  Demo 3  持久化与加载：StorageContext + persist / load_index_from_storage
  Demo 4  把 LlamaIndex 接到 Milvus Lite（承接 03_milvus_tutorial）
  Demo 5  Retriever vs QueryEngine：两种抽象层次的差别
  Demo 6  RouterQueryEngine：多索引自动路由（CodeLens 阶段 7 改造点 A）
  Demo 7  SubQuestionQueryEngine：复杂问题自动分解（改造点 B）
  Demo 8  RecursiveRetriever：摘要 → chunk 的两级层次检索（改造点 C）

运行前准备（agent 环境）：
    pip install llama-index llama-index-llms-openai-like \\
                llama-index-embeddings-huggingface \\
                llama-index-vector-stores-milvus
    # Demo 4 之外的 demo 不需要 milvus 包

可以单独跑某一个 demo：
    python 04_llamaindex_tutorial.py 1
    python 04_llamaindex_tutorial.py
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

# .env 在仓库根目录：/data/ghs/agent-learn/.env
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

TUTORIAL_DIR = Path(__file__).resolve().parent
STORAGE_DIR = TUTORIAL_DIR / "_llamaindex_storage"
STORAGE_DIR.mkdir(exist_ok=True)


# ===========================================================
# 公共：配置全局 LLM 与 Embedding（一次配，所有 demo 复用）
# ===========================================================
# LlamaIndex 默认连 OpenAI 官方端点，国内 / DeepSeek 用户必须显式覆盖：
#
#   Settings.llm         —— 全局默认 LLM
#   Settings.embed_model —— 全局默认 embedding 模型
#
# 不传的话，每个 Index/QueryEngine 会自动用 Settings.* 上的值。
# 这就是 LlamaIndex 比 LangChain 写起来更短的原因之一：不用每一处都传依赖。
# -----------------------------------------------------------
def _setup_settings():
    """所有 demo 共用，第一次调用时初始化 Settings。"""
    from llama_index.core import Settings

    if getattr(Settings, "_tutorial_initialized", False):
        return

    # 1) LLM：用 OpenAILike 接 DeepSeek（OpenAI 兼容协议）
    from llama_index.llms.openai_like import OpenAILike

    Settings.llm = OpenAILike(
        model="deepseek-chat",
        api_base=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        is_chat_model=True,         # 必须显式声明，否则 LlamaIndex 当成 completion
        temperature=0.2,
        timeout=60,
    )

    # 2) Embedding：本地 bge-small-zh-v1.5，跟 CodeLens / 03_milvus_tutorial 同款
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
    )

    # 3) 顺手把默认切片大小调小一点（默认 1024 / 20，对短文本太大）
    Settings.chunk_size = 256
    Settings.chunk_overlap = 32

    Settings._tutorial_initialized = True
    print(f"[setup] LLM=deepseek-chat  Embed=bge-small-zh-v1.5  "
          f"chunk={Settings.chunk_size}/{Settings.chunk_overlap}")


# 一些后续 demo 会用到的迷你"知识库"
TECH_DOCS = [
    ("milvus.md", "Milvus 是一款开源向量数据库，支持十亿级向量检索。"
                  "常用索引包括 HNSW、IVF_FLAT、FLAT。HNSW 适合读多写少的场景，"
                  "构建慢但查询快、召回率高。"),
    ("langchain.md", "LangChain 是大模型应用开发框架，提供 LCEL 链式调用、"
                     "Memory、Retriever、Tool、Agent 等抽象。LangGraph 是其衍生品，"
                     "把链升级成有状态的图，更适合写多步推理 Agent。"),
    ("llamaindex.md", "LlamaIndex 是面向 RAG 的数据框架，核心抽象包括 "
                      "Document、Node、Index、Retriever、QueryEngine。"
                      "在多索引路由、子问题分解、层次检索这类场景下表达力强。"),
    ("react.md", "ReAct 是一种让大模型边推理边调用工具的提示范式，"
                 "通过 Thought → Action → Observation 循环逐步逼近答案。"),
]

HISTORY_DOCS = [
    ("ai_history_1.md", "1956 年达特茅斯会议是人工智能学科正式诞生的标志，"
                        "John McCarthy 等人首次提出 'Artificial Intelligence' 一词。"),
    ("ai_history_2.md", "深度学习三巨头 Geoffrey Hinton、Yann LeCun、"
                        "Yoshua Bengio 因在神经网络方面的奠基工作获得 2018 年图灵奖。"),
    ("ai_history_3.md", "Transformer 架构在 2017 年由 Google 团队在论文 "
                        "《Attention Is All You Need》中提出，是 BERT/GPT 的基础。"),
]


def _write_corpus(name: str, items) -> Path:
    """把 (filename, content) 列表落盘成一个临时目录，方便 SimpleDirectoryReader 读。"""
    d = STORAGE_DIR / "corpus" / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for fname, content in items:
        (d / fname).write_text(content, encoding="utf-8")
    return d


# ===========================================================
# Demo 1 — 5 行代码跑通最小 RAG
# ===========================================================
# LlamaIndex 的招牌就是"短"。下面 5 行覆盖：读文件 → 切片 → embedding →
# 建向量索引 → 用 query_engine 提问。所有"切片大小、metric、top-k"
# 都走默认值。
#
# from_documents 内部其实做了：
#   1) 用 Settings.chunk_size 把每个 Document 切成 Node
#   2) 用 Settings.embed_model 给每个 Node 算向量
#   3) 把 Node + 向量塞进 SimpleVectorStore（内存版）
# Demo 2 会把这层"自动化"掀开看清楚。
# -----------------------------------------------------------
def demo_1_hello():
    _setup_settings()

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    corpus = _write_corpus("d1", TECH_DOCS)

    docs = SimpleDirectoryReader(str(corpus)).load_data()        # 1
    index = VectorStoreIndex.from_documents(docs)                # 2
    query_engine = index.as_query_engine(similarity_top_k=2)     # 3
    answer = query_engine.query("HNSW 索引适合什么场景？")        # 4
    print("=== Demo 1: 最小 RAG（5 行）===")
    print(f"answer: {answer}")
    # answer 不是裸字符串，而是 Response 对象，带引用
    print(f"引用了 {len(answer.source_nodes)} 个 Node：")
    for sn in answer.source_nodes:
        print(f"  [{sn.score:.3f}] {sn.node.metadata.get('file_name')} "
              f"-> {sn.node.text[:50]}...")
    print()


# ===========================================================
# Demo 2 — 显式拆 Document → Node → Index
# ===========================================================
# 把 Demo 1 的 from_documents 拆开，看清三件事：
#   ① Document 长什么样（id / text / metadata）
#   ② NodeParser（这里用 SentenceSplitter）按句子边界切片，
#     比按字符硬切语义损失更小
#   ③ 节点之间的父子关系（Node.relationships）
#
# 这层透明度在 CodeLens 这种"代码 + 文档异构数据"的场景里尤其重要：
# 你需要给代码 Node 和文档 Node 用不同的切分策略。
# -----------------------------------------------------------
def demo_2_pipeline():
    _setup_settings()

    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter

    # 1) 手工构造 Document（也可以从文件 load）
    docs = [
        Document(text=content, metadata={"source": fname, "type": "tech"})
        for fname, content in TECH_DOCS
    ]
    print("=== Demo 2: Document → Node → Index ===")
    print(f"Documents: {len(docs)}")
    print(f"  doc[0].id_  = {docs[0].id_}")
    print(f"  doc[0].metadata = {docs[0].metadata}")

    # 2) NodeParser：按"句子"切，再按 chunk_size 兜底
    splitter = SentenceSplitter(chunk_size=120, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"\n切完得到 {len(nodes)} 个 Node：")
    for n in nodes[:4]:
        print(f"  [{n.metadata['source']}] {n.text[:50]}...")
        # Node 自动记住自己的"父 Document"
        parents = n.relationships
        print(f"    relationships keys: {[k.name for k in parents.keys()]}")

    # 3) 用 Node（而不是 Document）建索引——更可控
    index = VectorStoreIndex(nodes)

    qe = index.as_query_engine(similarity_top_k=2)
    print(f"\n问：'LlamaIndex 的核心抽象有哪些？'")
    print(f"答：{qe.query('LlamaIndex 的核心抽象有哪些？')}")
    print()


# ===========================================================
# Demo 3 — 持久化与加载（StorageContext + persist / load）
# ===========================================================
# 上面两个 demo 用的是 in-memory store，进程一退就没了。
# 真项目里你要把 embedding 算好的索引存盘，下次直接加载，不重新算。
#
# 最朴素的做法：
#   ① index.storage_context.persist(persist_dir=...)
#      把 docstore / index_store / vector_store 三个 JSON/sqlite 文件写到目录
#   ② 下次：
#      storage_context = StorageContext.from_defaults(persist_dir=...)
#      index = load_index_from_storage(storage_context)
#
# Demo 4 会展示"换成 Milvus 后这一切由谁负责"。
# -----------------------------------------------------------
def demo_3_persistence():
    _setup_settings()

    from llama_index.core import (
        Document, VectorStoreIndex, StorageContext, load_index_from_storage,
    )

    persist_dir = STORAGE_DIR / "d3_persist"
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    # ---- 第一次：建索引并落盘 ----
    docs = [Document(text=c, metadata={"source": f}) for f, c in TECH_DOCS]
    index_a = VectorStoreIndex.from_documents(docs)
    index_a.storage_context.persist(persist_dir=str(persist_dir))

    files = sorted(p.name for p in persist_dir.iterdir())
    print("=== Demo 3: 持久化与加载 ===")
    print(f"persist_dir 里出现的文件: {files}")

    # ---- 第二次：模拟"重启"，从磁盘加载 ----
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index_b = load_index_from_storage(storage_context)
    qe = index_b.as_query_engine(similarity_top_k=1)
    print(f"\n重新加载后问：'ReAct 是什么？'")
    print(f"答：{qe.query('ReAct 是什么？')}")
    print()


# ===========================================================
# Demo 4 — 把 LlamaIndex 接到 Milvus Lite
# ===========================================================
# 默认 SimpleVectorStore 的向量存在 docstore.json 里，几万条之后又慢又占内存。
# 真项目把 vector_store 换成专业向量库就行：
#
#   vector_store = MilvusVectorStore(uri=..., dim=..., collection_name=...)
#   storage_context = StorageContext.from_defaults(vector_store=vector_store)
#   index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
#
# 注意 dim 必须和 Settings.embed_model 的输出维度对得上，否则建索引时会报错。
# bge-small-zh-v1.5 的输出维度是 512。
# -----------------------------------------------------------
def demo_4_milvus_backend():
    _setup_settings()

    try:
        from llama_index.vector_stores.milvus import MilvusVectorStore
    except ImportError:
        print("=== Demo 4: LlamaIndex × Milvus ===")
        print("(skip: 需要 `pip install llama-index-vector-stores-milvus`)")
        print()
        return

    from llama_index.core import (
        Document, VectorStoreIndex, StorageContext,
    )

    db_file = STORAGE_DIR / "d4_milvus.db"
    if db_file.exists():
        db_file.unlink()

    # 1) 用 Milvus Lite 当 vector_store
    vector_store = MilvusVectorStore(
        uri=str(db_file),
        collection_name="d4_demo",
        dim=512,                # bge-small-zh-v1.5 的维度
        overwrite=True,         # 教程方便：每次跑都重建
        similarity_metric="COSINE",
        index_config={
                    # "index_type": "HNSW",
                    "index_type": "FLAT",
                    #   "params": {"M": 16, "efConstruction": 200}
                      },
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2) 用 storage_context 建 index —— 此后所有 embedding 自动落地到 Milvus
    docs = [Document(text=c, metadata={"source": f}) for f, c in TECH_DOCS]
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    qe = index.as_query_engine(similarity_top_k=2)
    answer = qe.query("LangChain 和 LlamaIndex 是什么关系？")

    print("=== Demo 4: LlamaIndex × Milvus ===")
    print(f"答：{answer}")
    print(f"引用 {len(answer.source_nodes)} 条：")
    for sn in answer.source_nodes:
        print(f"  [{sn.score:.3f}] {sn.node.metadata.get('source')}")
    print()


# ===========================================================
# Demo 5 — Retriever vs QueryEngine：两层抽象的差别
# ===========================================================
# 这是 LlamaIndex 比 LangChain 思路清晰的地方。
#
#   Retriever   —— 输入 query，返回 top-k 个 NodeWithScore
#                  （没有 LLM 介入，纯检索）
#   QueryEngine —— 输入 query，先内部调 Retriever，再用 LLM 合成最终答案
#                  （= Retriever + ResponseSynthesizer）
#
# 业务上"先看候选 → 自己拼 prompt"用 Retriever；
# "一步出最终答案"用 QueryEngine。
# 后面 Router/SubQuestion 这些高级组件全部是基于 QueryEngine 的。
# -----------------------------------------------------------
def demo_5_retriever_vs_query_engine():
    _setup_settings()

    from llama_index.core import Document, VectorStoreIndex

    docs = [Document(text=c, metadata={"source": f}) for f, c in TECH_DOCS]
    index = VectorStoreIndex.from_documents(docs)

    print("=== Demo 5: Retriever vs QueryEngine ===")

    # ---- 当 Retriever 用：只拿候选，不让模型答 ----
    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve("HNSW 适合什么样的场景？")
    print("[Retriever] 返回的候选 Node：")
    for n in nodes:
        print(f"  [{n.score:.3f}] {n.node.metadata['source']}: "
              f"{n.node.text[:60]}...")

    # ---- 当 QueryEngine 用：直接拿答案 ----
    query_engine = index.as_query_engine(similarity_top_k=2)
    resp = query_engine.query("HNSW 适合什么样的场景？")
    print(f"\n[QueryEngine] 返回的最终答案：")
    print(f"  {resp}")
    print()


# ===========================================================
# Demo 6 — RouterQueryEngine：多索引自动路由
# ===========================================================
# 实战痛点：CodeLens 现在把 markdown 文档和 C++ 源码塞在同一个 collection 里，
# 问"项目历史"问"代码实现"返回的相关性都不高。LlamaIndex 给的标准答案：
#
#   每类资料各建一个 Index → 各自包成 QueryEngine → 用 RouterQueryEngine
#   按 query 语义自动选一个（或多个）路由过去。
#
# Selector 类型：
#   · LLMSingleSelector  —— LLM 选 1 个最相关的 engine
#   · LLMMultiSelector   —— LLM 选 N 个 engine 然后合并答案
#   · PydanticSingle/Multi —— 走 function calling 拿结构化输出，更稳
#
# 选型经验：DeepSeek 走 PydanticSingleSelector 比 LLMSingleSelector 稳。
# -----------------------------------------------------------
def demo_6_router():
    _setup_settings()

    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.selectors import LLMSingleSelector

    # 1) 两组文档分别建两个索引
    tech_docs = [Document(text=c, metadata={"source": f}) for f, c in TECH_DOCS]
    history_docs = [Document(text=c, metadata={"source": f}) for f, c in HISTORY_DOCS]

    tech_index = VectorStoreIndex.from_documents(tech_docs)
    history_index = VectorStoreIndex.from_documents(history_docs)

    # 2) 各自包成 QueryEngine + 加 description（路由器靠这段描述选 engine）
    tech_qe = tech_index.as_query_engine(similarity_top_k=2)
    history_qe = history_index.as_query_engine(similarity_top_k=2)

    tools = [
        QueryEngineTool(
            query_engine=tech_qe,
            metadata=ToolMetadata(
                name="tech_docs",
                description="技术文档：Milvus / LangChain / LlamaIndex / ReAct 等概念解释",
            ),
        ),
        QueryEngineTool(
            query_engine=history_qe,
            metadata=ToolMetadata(
                name="ai_history",
                description="AI 发展史：达特茅斯会议、图灵奖、Transformer 论文等历史事件",
            ),
        ),
    ]

    # 3) 用 LLMSingleSelector：让 LLM 自己挑
    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=tools,
        verbose=True,           # 打印路由决策，调试很有用
    )

    print("=== Demo 6: RouterQueryEngine ===")
    for q in ["HNSW 是什么算法？",
              "深度学习三巨头是谁？"]:
        print(f"\nQ: {q}")
        print(f"A: {router.query(q)}")
    print()


# ===========================================================
# Demo 7 — SubQuestionQueryEngine：复杂问题自动分解
# ===========================================================
# 单步 RAG 的死穴：
#   "对比 Milvus 和 LangChain 在 RAG 里各自扮演什么角色？"
#   一次检索的 top-k 里大概率只有其中一方的内容，模型瞎编另一方。
#
# SubQuestionQueryEngine 的解法：
#   ① 让 LLM 把原问题拆成 N 个子问题
#   ② 每个子问题独立路由到一个 QueryEngineTool 上跑
#   ③ 把 N 个子答案合成最终回答
#
# 注意：子问题分解本身要花一次 LLM 调用，子问题越多调用越多。
# 适合"多目标对比 / 多实体盘点"这类问题，不适合所有 query。
# -----------------------------------------------------------
def demo_7_subquestion():
    _setup_settings()

    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.core.query_engine import SubQuestionQueryEngine
    # 内置的纯 prompt-based 子问题生成器，不依赖 OpenAI SDK，DeepSeek 也能跑。
    from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator

    # 给每个概念准备一个独立 engine —— 让子问题有"分别检索"的载体
    engines = {}
    for fname, content in TECH_DOCS:
        idx = VectorStoreIndex.from_documents(
            [Document(text=content, metadata={"source": fname})]
        )
        engines[fname] = idx.as_query_engine(similarity_top_k=1)

    tools = [
        QueryEngineTool(
            query_engine=engines["milvus.md"],
            metadata=ToolMetadata(name="milvus", description="关于 Milvus 向量数据库"),
        ),
        QueryEngineTool(
            query_engine=engines["langchain.md"],
            metadata=ToolMetadata(name="langchain", description="关于 LangChain/LangGraph"),
        ),
        QueryEngineTool(
            query_engine=engines["llamaindex.md"],
            metadata=ToolMetadata(name="llamaindex", description="关于 LlamaIndex"),
        ),
    ]

    # ⚠️ SubQuestionQueryEngine.from_defaults() 默认会去 import
    # `llama-index-question-gen-openai`，那个包只有老版本（强制 core<0.13），
    # 跟我们用的 core 0.14 冲突。显式传一个内置 LLMQuestionGenerator 即可绕开
    # —— 它走纯 prompt + LLM 推理，对 DeepSeek/OpenAILike 完全友好。
    question_gen = LLMQuestionGenerator.from_defaults(llm=Settings.llm)

    sub_qe = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        question_gen=question_gen,
        verbose=True,           # 打印每个子问题的分解 + 子答案
        use_async=False,        # 简单起见关掉 async（省去 nest_asyncio 的坑）
    )

    print("=== Demo 7: SubQuestionQueryEngine ===")
    q = "对比 Milvus、LangChain、LlamaIndex 三者在 RAG 系统里的分工。"
    print(f"\nQ: {q}\n")
    print(f"最终答案：\n{sub_qe.query(q)}")
    print()


# ===========================================================
# Demo 8 — RecursiveRetriever：摘要 → chunk 两级层次检索
# ===========================================================
# 真实场景里一个文件可能有几千行（cpp-httplib 单文件 1 万多行）。
# 平铺切完检索回来都是碎片，模型很难"看到全局"。
#
# 层次检索的思路：
#   顶层：每个文件一段 1~3 句话的摘要（IndexNode，带 index_id 指向底层）
#   底层：原文 chunk
#   先用顶层 Index 检索摘要 → 命中后 RecursiveRetriever 自动切到底层 Index
#   去取真正的 chunk 上下文。
#
# 收益：query 第一跳只扫几十条摘要，命中后才去几百条 chunk 里精确找；
#       既省 token，又给模型一个"先粗后精"的视野。
# -----------------------------------------------------------
def demo_8_recursive():
    _setup_settings()

    from llama_index.core import VectorStoreIndex, Document
    from llama_index.core.schema import IndexNode, TextNode
    from llama_index.core.retrievers import RecursiveRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine

    # 1) 给每篇 tech 文档手写一段摘要 + 切几个底层 chunk
    summary_nodes: list[IndexNode] = []
    sub_indices = {}                 # name -> VectorStoreIndex（底层）

    for fname, content in TECH_DOCS:
        # —— 底层：把原文切成 chunk Node 建一个小索引 ——
        chunks = [TextNode(text=s.strip())
                  for s in content.split("。") if s.strip()]
        sub_idx = VectorStoreIndex(chunks)
        sub_indices[fname] = sub_idx

        # —— 顶层：用一句话摘要生成 IndexNode，index_id 指向底层 ——
        summary_text = f"文档 {fname} 概要：{content[:30]}…"
        summary_nodes.append(IndexNode(text=summary_text, index_id=fname))

    top_index = VectorStoreIndex(summary_nodes)
    top_retriever = top_index.as_retriever(similarity_top_k=2)

    # 2) 把顶层 retriever 和 N 个底层 retriever 注册到 RecursiveRetriever
    retriever_dict = {"root": top_retriever}
    for fname, sub_idx in sub_indices.items():
        retriever_dict[fname] = sub_idx.as_retriever(similarity_top_k=2)

    recursive = RecursiveRetriever(
        "root",
        retriever_dict=retriever_dict,
        verbose=True,
    )

    # 3) 包一个 QueryEngine 出来直接问答
    rqe = RetrieverQueryEngine.from_args(recursive)

    print("=== Demo 8: RecursiveRetriever（摘要 → chunk）===")
    for q in ["HNSW 索引的构建特点是什么？",
              "ReAct 是怎么循环的？"]:
        print(f"\nQ: {q}")
        print(f"A: {rqe.query(q)}")
    print()


# ===========================================================
# 把 Demo 对应到 CodeLens 的 LlamaIndex 改造路径
# ===========================================================
# 02_集成LlamaIndex_AutoGen_MCP方案.md 里给 LlamaIndex 列了 5 条改造路径：
#
#   路径 A: RouterQueryEngine 替换"一锅炖"检索       ⇐ 看懂 Demo 6
#   路径 B: SubQuestionQueryEngine 处理复杂连问      ⇐ 看懂 Demo 7
#   路径 C: 层次检索（Recursive）                    ⇐ 看懂 Demo 8
#   路径 D: KnowledgeGraphIndex（不推荐）             —— 略
#   路径 E: 用 LlamaIndex 同时连 Milvus              ⇐ 看懂 Demo 4
#
# 推荐起步顺序：Demo 1~3 (理解 LlamaIndex 抽象)
#               → Demo 4 (打通 Milvus 后端，半天能跑通改造路径 E)
#               → Demo 5 (理解 Retriever / QueryEngine 的差别)
#               → Demo 6 (一天完工改造路径 A)
#               → Demo 7 (再一天完工路径 B)
#               → Demo 8 看时间
#
# 看完这 8 个 demo 再回去看 codelens/app/retriever.py，就知道哪几行
# 该被替换、能换出多大价值。
# -----------------------------------------------------------

DEMOS = {
    # "1": demo_1_hello,
    # "2": demo_2_pipeline,
    # "3": demo_3_persistence,
    # "4": demo_4_milvus_backend,
    # "5": demo_5_retriever_vs_query_engine,
    # "6": demo_6_router,
    "7": demo_7_subquestion,
    "8": demo_8_recursive,
}


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which is None:
        for key in sorted(DEMOS):
            DEMOS[key]()
    else:
        DEMOS[which]()
