import sys
from pathlib import Path

# 把 codelens/ 加到 sys.path，让 `from app.*` 在任意 cwd 下都能 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from app.graph.build import build_graph
from app.llm import get_llm
from app.retriever import get_retriever


SYSTEM = ("你是 CodeLens，代码/文档智能助手。"
          "优先使用工具获取证据，不要凭空回答。每次调用工具后总结发现。")

graph = build_graph()

def format_docs(docs):
    return "\n\n".join(f"[{i}] {d.metadata.get('source')}\n{d.page_content}"
                       for i, d in enumerate(docs))

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "ThreadPool 如何实现？"
    init = {"messages": [SystemMessage(SYSTEM), HumanMessage(q)], "iterations": 0}
    for event in graph.stream(init, stream_mode="values"):
        last = event["messages"][-1]
        print(f"[{last.type}] {getattr(last, 'content', '')[:4000]}")
        if getattr(last, "tool_calls", None):
            for tc in last.tool_calls:
                print(f"  ↳ tool_call: {tc['name']}({tc['args']})")