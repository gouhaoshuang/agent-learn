"""
===========================================================
LangGraph 入门教程
===========================================================
本教程承接 01_langchain_tutorial.py。前者让你会用 LangChain 调模型、写链、
做结构化输出；本教程带你把"一条链"升级成"一张有状态的图"。

为什么需要 LangGraph？
  · LangChain 的 LCEL 链是"一次性、无状态、不回头"的；
  · 真实 Agent 要做「观察 → 思考 → 动手 → 再观察」的循环，
    还可能要「自省 → 重试」；
  · LangGraph 用「状态 (State) + 节点 (Node) + 条件边 (Edge)」建模这件事，
    显式、可调试、可恢复。

本文件最终要为 CodeLens 阶段 3 的 agent/tools/reflect 三节点 StateGraph
打下全部地基。按 Demo 顺序过一遍就够了：

  Demo 1  最小图：一个节点
  Demo 2  多节点管道：State 合并规则（reducer 的由来）
  Demo 3  messages State + add_messages reducer（后面所有 Agent 的地基）
  Demo 4  条件边 + 图结构可视化
  Demo 5  Tool Calling + ToolNode：手搓 ReAct 循环
  Demo 6  加一个 reflect 节点：自省与重试
  Demo 7  Checkpointer 多轮记忆（thread_id）
  Demo 8  流式执行 stream_mode="values"

运行前准备（agent 环境已装好）：
    pip install langgraph langchain langchain-openai python-dotenv

可以单独跑某一个 demo：
    python 02_langgraph_tutorial.py 1      # 只跑 Demo 1
    python 02_langgraph_tutorial.py        # 全跑
"""

import os
from dotenv import load_dotenv

# .env 在仓库根目录：/data/ghs/agent-learn/.env
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# -----------------------------------------------------------
# 公共：复用 01 里的模型初始化
# -----------------------------------------------------------
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=temperature,
        base_url=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ===========================================================
# Demo 1 — 最小图：一个节点
# ===========================================================
# 核心对象：
#   · StateGraph(SchemaClass)  —— 图的建造者，SchemaClass 定义"状态长什么样"
#   · add_node(name, fn)       —— 往图里加一个节点；节点 fn 接收 state，返回 dict
#   · set_entry_point(name)    —— 图从哪个节点开始
#   · add_edge(a, b) / add_edge(a, END) —— 无条件边
#   · compile()                —— 把建造好的图冻结成可调用对象
#   · invoke(initial_state)    —— 从入口开始跑，返回最终 state
#
# State 用 TypedDict 描述。节点函数返回的 dict 会被「合并」到 state。
# -----------------------------------------------------------
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class HelloState(TypedDict):
    question: str
    answer: str


def demo_1_minimal_graph():
    def answer_node(state: HelloState) -> dict:
        # 节点只关心自己要读/写的字段；返回 dict 即可
        q = state["question"]
        resp = get_llm().invoke(q)
        return {"answer": resp.content}

    g = StateGraph(HelloState)
    g.add_node("answer", answer_node)
    g.set_entry_point("answer")           # 等价于 g.add_edge(START, "answer")
    g.add_edge("answer", END)             # END 是特殊终止节点
    graph = g.compile()

    final_state = graph.invoke({"question": "一句话解释什么是 LangGraph？"})
    print("=== Demo 1: 最小图 ===")
    print("Q:", final_state["question"])
    print("A:", final_state["answer"])
    print()


# ===========================================================
# Demo 2 — 多节点管道 & State 的"合并规则"
# ===========================================================
# 关键概念：reducer
#   · 默认：节点返回 {"key": value} 会直接"覆盖"旧值；
#   · 如果想「追加而不是覆盖」(比如消息列表)，要用 Annotated[type, reducer]。
#
# 这里先用默认覆盖规则，演示三个节点串成管道：
#   extract_topic  →  generate_outline  →  polish
# -----------------------------------------------------------
class PipelineState(TypedDict):
    user_input: str
    topic: str          # 被 extract_topic 写
    outline: str        # 被 generate_outline 写
    final: str          # 被 polish 写


def demo_2_pipeline():
    llm = get_llm()

    def extract_topic(state: PipelineState) -> dict:
        resp = llm.invoke(f"用 5 个字以内概括主题：{state['user_input']}")
        return {"topic": resp.content.strip()}

    def generate_outline(state: PipelineState) -> dict:
        resp = llm.invoke(f"为主题『{state['topic']}』写一个 3 条要点的大纲，每条一行。")
        return {"outline": resp.content.strip()}

    def polish(state: PipelineState) -> dict:
        resp = llm.invoke(f"润色以下大纲，使语气更专业：\n{state['outline']}")
        return {"final": resp.content.strip()}

    g = StateGraph(PipelineState)
    g.add_node("extract", extract_topic)
    g.add_node("outline", generate_outline)
    g.add_node("polish", polish)
    g.set_entry_point("extract")
    g.add_edge("extract", "outline")
    g.add_edge("outline", "polish")
    g.add_edge("polish", END)
    graph = g.compile()

    out = graph.invoke({"user_input": "介绍一下 Transformer 为什么能取代 RNN"})
    print("=== Demo 2: 多节点管道 ===")
    print("topic   :", out["topic"])
    print("outline :", out["outline"])
    print("final   :", out["final"])
    print()


# ===========================================================
# Demo 3 — messages State + add_messages reducer
# ===========================================================
# 这是后面所有 Agent 代码的地基，一定要看懂。
#
# 场景：一个 Agent 要边执行边往对话历史里加消息（user → ai → tool → ai ...）。
# 如果你写 `messages: list[AnyMessage]`，每个节点 `return {"messages": [x]}`
# 会把整个列表"覆盖"成只有一条，历史就丢了。
#
# LangGraph 提供了 add_messages reducer：
#   messages: Annotated[list[AnyMessage], add_messages]
# 它会把节点返回的列表「追加」到现有 messages 里，还会帮你处理 id 去重。
# -----------------------------------------------------------
from typing import Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def demo_3_messages_state():
    llm = get_llm()

    def chat_node(state: ChatState) -> dict:
        ai = llm.invoke(state["messages"])
        # 只返回「这一步新增的消息」，add_messages 会帮你拼到历史末尾
        return {"messages": [ai]}

    g = StateGraph(ChatState)
    g.add_node("chat", chat_node)
    g.set_entry_point("chat")
    g.add_edge("chat", END)
    graph = g.compile()

    init = {"messages": [
        SystemMessage("你是一个简洁的助手，回答不超过两句话。"),
        HumanMessage("什么是有状态 Agent？"),
    ]}
    final = graph.invoke(init)

    print("=== Demo 3: messages State ===")
    for m in final["messages"]:
        # 注意：虽然 chat_node 只 return 了 1 条新消息，
        # 但最终 state.messages 里既有最初的 2 条，也有新加的 1 条
        print(f"  [{m.type}] {m.content[:80]}")
    print()


# ===========================================================
# Demo 4 — 条件边 + 图结构可视化
# ===========================================================
# add_conditional_edges(源节点, 路由函数, 映射表)
#   · 路由函数接收 state，返回一个字符串 key；
#   · 映射表把 key 翻成目标节点名（或 END）。
#
# 典型用法：根据最后一条消息里有没有 tool_calls 决定是 "tools" 还是 END。
# 这里先用一个更简单的场景：根据问题是"代码题"还是"文档题"路由。
# -----------------------------------------------------------
class RouteState(TypedDict):
    question: str
    route: str
    answer: str


def demo_4_conditional_edges():
    llm = get_llm()

    def classify(state: RouteState) -> dict:
        # 让模型给出一个单词分类
        resp = llm.invoke(
            f"下面这个问题属于 code 还是 doc？只回答一个词。\n问题：{state['question']}"
        )
        label = resp.content.strip().lower()
        route = "code" if "code" in label else "doc"
        return {"route": route}

    def answer_code(state: RouteState) -> dict:
        resp = llm.invoke(f"作为 C++ 专家回答：{state['question']}")
        return {"answer": "[CODE] " + resp.content}

    def answer_doc(state: RouteState) -> dict:
        resp = llm.invoke(f"作为技术文档助手回答：{state['question']}")
        return {"answer": "[DOC]  " + resp.content}

    # 路由函数：读 state.route，返回映射表里的 key
    def pick_branch(state: RouteState) -> str:
        return state["route"]

    g = StateGraph(RouteState)
    g.add_node("classify", classify)
    g.add_node("answer_code", answer_code)
    g.add_node("answer_doc", answer_doc)
    g.set_entry_point("classify")
    g.add_conditional_edges(
        "classify",
        pick_branch,
        {"code": "answer_code", "doc": "answer_doc"},
    )
    g.add_edge("answer_code", END)
    g.add_edge("answer_doc", END)
    graph = g.compile()

    # 小彩蛋：把图结构用 ASCII 打出来，调试时超好用
    print("=== Demo 4: 条件边 + 图可视化 ===")
    print("--- graph topology ---")
    try:
        graph.get_graph().print_ascii()
    except Exception as e:
        # print_ascii 需要 grandalf；没装就跳过
        print(f"(print_ascii 不可用：{e})")
    print()

    for q in ["std::shared_ptr 的引用计数是线程安全的吗？",
              "敏捷开发里的 sprint 是什么？"]:
        out = graph.invoke({"question": q})
        print(f"[route={out['route']}] {out['answer'][:120]}")
    print()


# ===========================================================
# Demo 5 — Tool Calling + ToolNode：手搓 ReAct 循环
# ===========================================================
# 这是 LangGraph 真正的威力所在，也是 CodeLens 阶段 3 的核心模式。
#
# 形状：
#
#        ┌──────── tool_calls? ───────┐
#        │                            ▼
#   [ agent ] ◀── 工具结果追加 ── [ tools ]
#        │
#        └── 无 tool_calls ──▶ END
#
# 关键点：
#   · bind_tools([t1, t2]) 让 LLM 可以在 AIMessage.tool_calls 里声明想调谁；
#   · ToolNode(tools) 是 prebuilt 节点，读 state.messages[-1].tool_calls，
#     执行工具后把 ToolMessage 追加到 messages；
#   · 条件边根据"最后一条 AIMessage 里有没有 tool_calls"决定是循环还是结束。
# -----------------------------------------------------------
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


@tool
def add(a: int, b: int) -> int:
    """计算 a + b，返回整数和。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算 a * b，返回整数积。"""
    return a * b


def build_react_graph(tools, extra_nodes: bool = False, checkpointer=None):
    """构造 agent + tools 的 ReAct 图；extra_nodes=True 时在 Demo 6 里加 reflect。"""
    llm_with_tools = get_llm(temperature=0).bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: ChatState) -> dict:
        ai = llm_with_tools.invoke(state["messages"])
        return {"messages": [ai]}

    def should_continue(state: ChatState) -> str:
        last = state["messages"][-1]
        # AIMessage 可能有 tool_calls（列表）；没有就结束
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    g = StateGraph(ChatState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue,
                            {"tools": "tools", END: END})
    g.add_edge("tools", "agent")        # 工具跑完无条件回 agent
    return g.compile(checkpointer=checkpointer)


def demo_5_react():
    graph = build_react_graph([add, multiply])

    question = "先算 13 加 29，再把结果乘以 7，请用工具算，不要自己口算。"
    init = {"messages": [
        SystemMessage("你是一个会用工具的计算助手。优先调用工具，不要心算。"),
        HumanMessage(question),
    ]}

    final = graph.invoke(init)

    # 小彩蛋：把图结构用 ASCII 打出来，调试时超好用
    print("=== Demo 5: Tool Calling + ToolNode ===")
    print("--- Tool Calling ---")
    try:
        graph.get_graph().print_ascii()
    except Exception as e:
        # print_ascii 需要 grandalf；没装就跳过
        print(f"(print_ascii 不可用：{e})")
    print()



    print("=== Demo 5: ReAct (agent ↔ tools) ===")
    for m in final["messages"]:
        prefix = f"[{m.type}]"
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"{prefix} tool_call: {tc['name']}({tc['args']})")
        elif m.type == "tool":
            print(f"{prefix} → {m.content}")
        else:
            content = (m.content or "").strip()
            if content:
                print(f"{prefix} {content[:120]}")
    print()


# ===========================================================
# Demo 6 — 加 reflect 节点：自省与重试
# ===========================================================
# 目标：Agent 答完后，用同一个 LLM 以"审稿人"身份检查一次；
#      如果反思觉得不够，就再回 agent 多跑一轮。
#
# 这里扩展状态，加一个 iterations 字段做循环上限保护。
# 这个结构和 CodeLens 阶段 3 的 agent/tools/reflect 几乎 1:1。
# -----------------------------------------------------------
class ReactWithReflectState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iterations: int
    reflection: str


def demo_6_reflect():
    tools = [add, multiply]
    llm_with_tools = get_llm(temperature=0).bind_tools(tools)
    tool_node = ToolNode(tools)

    MAX_ITERS = 5

    def agent_node(state: ReactWithReflectState) -> dict:
        ai = llm_with_tools.invoke(state["messages"])
        return {
            "messages": [ai],
            "iterations": state.get("iterations", 0) + 1,
        }

    def reflect_node(state):              
        print(f"  >>> [reflect] 正在审稿...")                                       
        critic = get_llm(temperature=0)                                             
        crit_msg = critic.invoke(state["messages"] + [HumanMessage(
            "请以审稿人身份，用一句话判断上面的回答是否足够准确。"                  
            "若仍有数字或步骤没验算，请只回复 '需要继续'；否则回复 '可以结束'。"    
        )])                                                                         
        print(f"  >>> [reflect] 判定：{crit_msg.content}")                          
        return {"reflection": crit_msg.content}     

    def route_after_agent(state: ReactWithReflectState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        if state.get("iterations", 0) >= MAX_ITERS:
            return END
        return "reflect"

    def route_after_reflect(state: ReactWithReflectState) -> str:
        if "需要继续" in state.get("reflection", ""):
            # 给 agent 再喂一条提示消息，迫使它再尝试一次
            return "agent"
        return END

    g = StateGraph(ReactWithReflectState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("reflect", reflect_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route_after_agent,
                            {"tools": "tools", "reflect": "reflect", END: END})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("reflect", route_after_reflect,
                            {"agent": "agent", END: END})
    graph = g.compile()


    try:
        graph.get_graph().print_ascii()
    except Exception as e:
        # print_ascii 需要 grandalf；没装就跳过
        print(f"(print_ascii 不可用：{e})")
    print()


    init = {
        "messages": [
            SystemMessage("你是严谨的计算助手，必须调用工具验算。"),
            HumanMessage("帮我算 (13 + 29) * 7 的结果。"),
        ],
        "iterations": 0,
        "reflection": "",
    }
    final = graph.invoke(init)


    print("=== Demo 6: ReAct + Reflect ===")
    print(f"iterations: {final['iterations']}")
    print(f"reflection: {final['reflection']}")
    # 最后一条 AIMessage 通常是答案
    ai_contents = [m.content for m in final["messages"]
                   if m.type == "ai" and m.content]
    if ai_contents:
        print(f"final answer: {ai_contents[-1][:200]}")
    print()

    print("=== Demo 6: 加 reflect 节点：自省与重试 ===")
    for m in final["messages"]:
        prefix = f"[{m.type}]"
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"{prefix} tool_call: {tc['name']}({tc['args']})")
        elif m.type == "tool":
            print(f"{prefix} → {m.content}")
        else:
            content = (m.content or "").strip()
            if content:
                print(f"{prefix} {content[:120]}")
    print()



# ===========================================================
# Demo 7 — Checkpointer 多轮记忆（thread_id）
# ===========================================================
# 把 state 持久化，同一个 thread_id 就是同一场对话。
#   · MemorySaver：进程内字典，重启即失忆，适合 demo / 单测；
#   · SqliteSaver：写 sqlite，适合本机持久化（CodeLens 阶段 4 用这个）。
#
# 只要在 compile(checkpointer=...) 时挂上，并且 invoke 时传
#   config={"configurable": {"thread_id": "xxx"}}
# LangGraph 会自动把上一轮 state 读出来、把新状态写回去。
# -----------------------------------------------------------
from langgraph.checkpoint.memory import MemorySaver


def demo_7_memory():
    saver = MemorySaver()
    graph = build_react_graph([add, multiply], checkpointer=saver)

    cfg = {"configurable": {"thread_id": "tut-session-1"}}

    print("=== Demo 7: Checkpointer + 多轮记忆 ===")

    # 第一轮：告诉它一个事实
    r1 = graph.invoke(
        {"messages": [SystemMessage("你是一个会用工具的助手，记住用户说过的事。"),
                      HumanMessage("我最喜欢的数字是 42。")]},
        config=cfg,
    )
    print("round 1:", r1["messages"][-1].content[:120])

    # 第二轮：只发新消息，靠 checkpointer 自动把历史接上
    r2 = graph.invoke(
        {"messages": [HumanMessage("请把我最喜欢的数字乘以 3，用工具算。")]},
        config=cfg,
    )
    print("round 2:", r2["messages"][-1].content[:120])

    # 换个 thread_id → 上下文是空的，模型就不知道 42 这件事
    r3 = graph.invoke(
        {"messages": [HumanMessage("我最喜欢的数字是多少？")]},
        config={"configurable": {"thread_id": "tut-session-OTHER"}},
    )
    print("other thread:", r3["messages"][-1].content[:120])
    print()


# ===========================================================
# Demo 8 — 流式执行（stream_mode="values"）
# ===========================================================
# invoke() 只给你最终 state，中间发生了什么看不到。
# stream() 则在每一步结束后 yield 一个事件，方便实时展示"Agent 在想什么"。
#
# 常用的 stream_mode：
#   · "values"  —— 每步 yield 当前 state 的完整快照（最直观）
#   · "updates" —— 每步只 yield 这一步"新增"的那部分（省流量）
#   · "messages" —— 面向 chat UI 的 token 级流（更细）
#
# CodeLens 的 CLI/UI 最常用的就是 "values"。
# -----------------------------------------------------------
def demo_8_stream():
    graph = build_react_graph([add, multiply])

    init = {"messages": [
        SystemMessage("你是会用工具的助手，优先调用工具。"),
        HumanMessage("用工具算 (8 + 4) * 6 是多少。"),
    ]}

    print("=== Demo 8: 流式执行 ===")
    seen = 0
    for event in graph.stream(init, stream_mode="values"):
        # event 就是当前 state 快照；我们只打印最新一条消息
        msgs = event["messages"]
        if len(msgs) <= seen:
            continue
        seen = len(msgs)
        last = msgs[-1]
        if getattr(last, "tool_calls", None):
            for tc in last.tool_calls:
                print(f"  [ai → tool] {tc['name']}({tc['args']})")
        elif last.type == "tool":
            print(f"  [tool → ai] {last.content}")
        elif last.type == "ai" and last.content:
            print(f"  [ai] {last.content[:120]}")
        elif last.type == "human":
            print(f"  [human] {last.content[:120]}")
    print()


# ===========================================================
# 把 Demo 对应到 CodeLens 阶段
# ===========================================================
# · Demo 1~4   ⇒  理解 StateGraph / 合并规则 / 条件边。这是所有后面代码的地基。
# · Demo 5     ⇒  CodeLens 阶段 2 的 `bind_tools` + 阶段 3 `ToolNode`。
# · Demo 6     ⇒  CodeLens 阶段 3 的 reflect 节点与 iteration 上限保护。
# · Demo 7     ⇒  CodeLens 阶段 4 的 SqliteSaver 多轮记忆，区别只是把
#                 MemorySaver 换成 SqliteSaver.from_conn_string("storage/checkpoints.db")。
# · Demo 8     ⇒  CodeLens 阶段 3/6 的 CLI 与 Streamlit 展示走的就是 stream("values")。
#
# 做完这 8 个 demo，你再去看 CodeLens 的 app/graph/ 目录，几乎每一行都能对上号。
# -----------------------------------------------------------

DEMOS = {
    # "1": demo_1_minimal_graph,
    # "2": demo_2_pipeline,
    # "3": demo_3_messages_state,
    # "4": demo_4_conditional_edges,
    # "5": demo_5_react,
    # "6": demo_6_reflect,
    "7": demo_7_memory,
    # "8": demo_8_stream,
}


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which is None:
        for key in sorted(DEMOS):
            DEMOS[key]()
    else:
        DEMOS[which]()
