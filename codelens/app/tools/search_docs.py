"""检索工具（阶段 8.1 起改由 LlamaIndex Router 提供能力）。

旧版本：直接用 langchain `Chroma.as_retriever()` 拿 top-k 片段拼接返回。
新版本：调 LlamaIndex `RouterQueryEngine`——它会让 LLM 看 query 自动从
`doc_only` / `code_only` / `multi_aspect` 三个分支里挑一个，再走对应的
QueryEngine 合成回答。返回的 string 包含 LLM 合成的答案 + 引用的片段元数据。

为什么不再保留"只返回片段"的语义？
  · Router 的价值在于"自动选 + 合成"，只取片段就只剩选——退化成 selector。
  · LangGraph agent 拿到合成答案后还可以继续用其他工具（grep / read_file）
    交叉验证，并不影响 ReAct 的多步流程。
  · 真要拿原始片段，绕过 Router 直接用 `app.retrieval.router.get_docs_query_engine()`
    或 `get_code_query_engine()` 调 `.retrieve(query)` 即可。
"""

from langchain_core.tools import tool

from app.retrieval.router import get_router


@tool
def search_docs(query: str, k: int = 5) -> str:
    """在本地文档与代码向量库中检索与 query 相关的内容并合成答案。

    内部走 LlamaIndex RouterQueryEngine：
      · 概念/规范/使用方法类问题 → docs collection
      · 类实现/函数定义/调用关系类问题 → code collection
      · 多目标对比 / 跨文档代码连问 → SubQuestionQueryEngine 自动拆子问题

    返回：合成后的回答正文 + 末尾追加每个引用片段的 source 路径。

    `k` 参数为兼容旧签名保留，目前不直接生效（QueryEngine 的 top_k 在
    构造时已固定为 5）。后续如有需要可在 router.py 里参数化。
    """
    response = get_router().query(query)

    body = str(response).strip()

    # 把 source_nodes 的来源整理成"参考片段"附在末尾，给上层 agent 留出引用线索
    refs = []
    for i, n in enumerate(getattr(response, "source_nodes", []) or []):
        src = (n.node.metadata or {}).get("source", "(unknown)")
        snippet = n.node.get_content().strip().replace("\n", " ")[:160]
        refs.append(f"  [{i}] {src} :: {snippet}...")

    if refs:
        return f"{body}\n\n--- 引用片段 ---\n" + "\n".join(refs)
    return body