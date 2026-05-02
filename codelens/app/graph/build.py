
from langgraph.graph import StateGraph, END
from app.graph.state import CodeLensState
from app.graph.nodes import agent_node, tools_node, reflect_node

from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Agent 节点累计最多跑 N 次后强制结束。
# 设计动机：阶段 8.1 改造后 search_docs 返回 LlamaIndex 合成答案 + 引用文件列表，
# 模型容易把多个引用 source 误解为"还有别的可查"，陷入反复 search 死循环。
# 硬上限作为兜底，避免触发 LangGraph 的 recursion_limit (25/50) 抛错。
AGENT_MAX_ITERATIONS = 50


def should_continue(state):
    iters = state.get("iterations", 0)
    last = state["messages"][-1]

    # 硬保险：超过上限即使还想调工具也强制 END
    if iters >= AGENT_MAX_ITERATIONS:
        return END

    if getattr(last, "tool_calls", None):
        return "tools"
    # 没有工具调用，但迭代次数少 → 进入反思；迭代够了 → 直接结束
    if iters < 3:
        return "reflect"
    return END

def after_reflect(state):
    # reflect 仅在 iterations < 3 时被进入，所以这里不需要再加上限保护
    if "需要继续检索" in state.get("reflection", ""):
        return "agent"
    return END

def build_graph(checkpointer=None):
    g = StateGraph(CodeLensState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("reflect", reflect_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue,
                             {"tools": "tools", "reflect": "reflect", END: END})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("reflect", after_reflect,
                             {"agent": "agent", END: END})
    return g.compile(checkpointer=checkpointer)