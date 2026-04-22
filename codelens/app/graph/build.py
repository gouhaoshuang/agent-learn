
from langgraph.graph import StateGraph, END
from app.graph.state import CodeLensState
from app.graph.nodes import agent_node, tools_node, reflect_node

def should_continue(state):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    # 没有工具调用，但迭代次数少 → 进入反思；迭代够了 → 直接结束
    if state.get("iterations", 0) < 3:
        return "reflect"
    return END

def after_reflect(state):
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