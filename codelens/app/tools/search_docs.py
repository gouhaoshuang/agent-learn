
from langchain_core.tools import tool
from app.retriever import get_retriever

@tool
def search_docs(query: str, k: int = 5) -> str:
    """在本地文档与代码向量库中检索与 query 相关的片段。返回拼接后的片段文本。"""
    docs = get_retriever(k=k).invoke(query)
    return "\n\n".join(f"[{d.metadata.get('source')}]\n{d.page_content}" for d in docs)