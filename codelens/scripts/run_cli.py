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

# 单次 tool 输出在终端最多显示几行，避免 read_file 一口气刷屏；
# 注意只截显示，ToolMessage.content 完整内容仍然进模型 context。
TOOL_LOG_MAX_LINES = 10


def _format_head(msg) -> str:
    # tool 消息带上具体工具名，调试时一眼看出是谁返回的
    if msg.type == "tool":
        return f"[tool:{getattr(msg, 'name', '?')}]"
    return f"[{msg.type}]"


def _format_body(msg) -> str:
    """只截 tool 消息的显示；ai/human/system 一律原样输出，
    避免模型最终答案被折叠。"""
    content = getattr(msg, "content", "") or ""
    if msg.type != "tool":
        return content
    lines = content.splitlines() or [content]
    if len(lines) <= TOOL_LOG_MAX_LINES:
        return content
    preview = "\n".join(lines[:TOOL_LOG_MAX_LINES])
    return f"{preview}\n  ... ({len(lines) - TOOL_LOG_MAX_LINES} more lines truncated)"


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "ThreadPool 如何实现？"
    init = {"messages": [SystemMessage(SYSTEM), HumanMessage(q)], "iterations": 0}
    for event in graph.stream(init, stream_mode="values"):
        last = event["messages"][-1]
        print(f"{_format_head(last)} {_format_body(last)}")
        if getattr(last, "tool_calls", None):
            for tc in last.tool_calls:
                print(f"  ↳ tool_call: {tc['name']}({tc['args']})")