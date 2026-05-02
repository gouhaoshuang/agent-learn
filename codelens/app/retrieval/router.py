# -*- coding: utf-8 -*-
"""
RouterQueryEngine + SubQuestionQueryEngine 组装。

三个原子工具 + 一个顶层路由：
    docs_qe        ：只查文档（codelens_docs collection）
    code_qe        ：只查代码（codelens_code collection）
    subq_engine    ：跨类型对比题：自动拆子问题、并行查 docs_qe / code_qe、合成
    router         ：顶层 LLMSingleSelector 路由，让 LLM 看 query 选上面三选一

为什么 SubQuestion 没有用第四个独立 collection？
    它本身不是新数据源，是"在 docs/code 之上的元能力"——把一个复合问题拆成多个
    子问题，每个子问题独立完整查回答，最后合成。所以 SubQuestion 的 sub-tool 就
    是上面已有的 docs_qe / code_qe，不需要重建任何索引。

模块级单例：QueryEngine 构造时会做一些 LLM-side 的 schema 准备工作，没必要每
次查询都重做。setup_llamaindex() 是幂等的，所以哪怕第一次访问 router 时才触发
也没事。
"""

from __future__ import annotations

from app.retrieval.settings import setup_llamaindex
from app.retrieval.vector_stores import get_docs_store, get_code_store


def _make_robust_selector(fallback_choice: int):
    """构造一个带容错的 LLMSingleSelector。

    问题背景：LlamaIndex 的 selection prompt 模板里把"输出格式举例"写成 `{{ ... }}`
    形式（双花括号是 prompt 模板转义，渲染给 LLM 看时应该是 `{ ... }`）。但
    DeepSeek 偶发会把双花括号当字面量原样输出，触发内置 `_SelectionOutputParser`
    的 OutputParserException。

    本函数返回一个 LLMSingleSelector：通过包装它内部 prompt 的 output_parser，
      ① 在 parse() 前先做字符串清理（`{{` → `{`、`}}` → `}`）
      ② 仍然失败时降级到 fallback_choice 指定分支（避免直接报错让查询失败）
    """
    from llama_index.core.selectors import LLMSingleSelector
    from llama_index.core.types import BaseOutputParser

    base = LLMSingleSelector.from_defaults()
    original_parser: BaseOutputParser = base._prompt.output_parser

    class RobustParser(BaseOutputParser):
        """包装原 parser，先清理 `{{` `}}` 再 parse；失败则返回 fallback。"""

        def parse(self, output: str):
            cleaned = output.replace("{{", "{").replace("}}", "}")
            try:
                return original_parser.parse(cleaned)
            except Exception as e:
                # 用原 parser 的 Answer 类型造一个 fallback 回值。LlamaIndex 0.14
                # 的 Answer 是 pydantic Model，字段为 choice/reason；返回单选用列表。
                from llama_index.core.output_parsers.selection import Answer
                print(f"[router] selector parse failed, fallback to choice "
                      f"{fallback_choice}: {e}")
                return [Answer(choice=fallback_choice + 1, reason="(fallback)")]

        def format(self, prompt_template: str) -> str:
            return original_parser.format(prompt_template)

    base._prompt.output_parser = RobustParser()
    return base


_router = None
_docs_qe = None
_code_qe = None
_subq_engine = None


def _build_engines():
    """构造四个 engine。仅首次访问时调用。"""
    global _router, _docs_qe, _code_qe, _subq_engine
    if _router is not None:
        return

    setup_llamaindex()

    from llama_index.core import VectorStoreIndex
    from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.core.question_gen import LLMQuestionGenerator

    # ---- 原子 QueryEngine：docs / code ----
    docs_index = VectorStoreIndex.from_vector_store(get_docs_store())
    code_index = VectorStoreIndex.from_vector_store(get_code_store())
    _docs_qe = docs_index.as_query_engine(similarity_top_k=5)
    _code_qe = code_index.as_query_engine(similarity_top_k=5)

    # ---- SubQuestion：跨类型对比时把子问题分配给 docs_qe / code_qe ----
    # 注意：SubQuestion 的 sub-tool 名字会被 LLM 直接拿来在 sub-question 的 JSON
    # 里引用（"tool_name": "docs"），描述写得越具体，子问题分配越准。
    #
    # ⚠️ 显式传 question_gen=LLMQuestionGenerator：默认 from_defaults() 会去 import
    #    `llama_index.question_gen.openai`，但这个子包（llama-index-question-gen-openai）
    #    最新 0.3.x 只兼容 core<0.13，是绑死在旧版本的"孤儿包"（详见 codelens
    #    requirements.txt 的注释）。core>=0.14 必须手动构造 question_generator。
    #    LLMQuestionGenerator 在 llama-index-core 自带，直接用 Settings.llm 工作，零额外依赖。
    sub_tools = [
        QueryEngineTool(
            query_engine=_docs_qe,
            metadata=ToolMetadata(
                name="docs",
                description="技术文档片段检索：概念、设计、规范、使用方法",
            ),
        ),
        QueryEngineTool(
            query_engine=_code_qe,
            metadata=ToolMetadata(
                name="code",
                description="C/C++ 源码片段检索：类实现、函数定义、调用关系",
            ),
        ),
    ]
    _subq_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=sub_tools,
        question_gen=LLMQuestionGenerator.from_defaults(),
        use_async=False,   # 子问题串行跑；DeepSeek 走 HTTP，串行能稳避免并发上限
    )

    # ---- 顶层 Router：LLMSingleSelector 看 query 选三选一 ----
    # 用 _make_robust_selector 包一层容错：DeepSeek 偶发把 LlamaIndex prompt
    # 里的 `{{` `}}` 转义符号当字面量输出，导致内置 _SelectionOutputParser 抛
    # OutputParserException。本 wrapper 在 parse 前先做字符串清理（`{{` → `{`、
    # `}}` → `}`），解析失败时降级到 multi_aspect (index=2) 兜底分支。
    _router = RouterQueryEngine(
        selector=_make_robust_selector(fallback_choice=2),
        query_engine_tools=[
            QueryEngineTool(
                query_engine=_docs_qe,
                metadata=ToolMetadata(
                    name="doc_only",
                    description=(
                        "适用于只需要查文档的问题：概念解释、使用方法、设计文档、"
                        "API 说明、规范定义。明确只问文档时选这个。"
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=_code_qe,
                metadata=ToolMetadata(
                    name="code_only",
                    description=(
                        "适用于只需要看源码的问题：某个类怎么实现、某个函数定义"
                        "在哪、调用关系、源码具体行为。明确只问代码时选这个。"
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=_subq_engine,
                metadata=ToolMetadata(
                    name="multi_aspect",
                    description=(
                        "适用于多目标对比题或跨文档代码连问，例如『X 和 Y 的对比』、"
                        "『A 在文档里写的和代码里实际行为是否一致』。问题需要拆成"
                        "多个子问题分别查再合成时选这个。"
                    ),
                ),
            ),
        ],
    )


def get_router():
    """返回顶层路由 QueryEngine（智能挑 docs / code / multi_aspect）。"""
    _build_engines()
    return _router


def get_docs_query_engine():
    """直接拿文档原子 QueryEngine，跳过路由。"""
    _build_engines()
    return _docs_qe


def get_code_query_engine():
    """直接拿代码原子 QueryEngine，跳过路由。"""
    _build_engines()
    return _code_qe


def get_subquestion_query_engine():
    """直接拿 SubQuestion 引擎，跳过路由（明知问题就是对比题时用）。"""
    _build_engines()
    return _subq_engine
