from langgraph.prebuilt import ToolNode
from app.llm import get_llm
from app.tools.search_docs import search_docs
from app.tools.grep_code import grep_code

TOOLS = [search_docs, grep_code]
llm_with_tools = get_llm(temperature=0).bind_tools(TOOLS)

def agent_node(state):
    """ReAct 决策节点：让模型决定继续用工具还是给出答案"""
    return {"messages": [llm_with_tools.invoke(state["messages"])],
            "iterations": state.get("iterations", 0) + 1}

tools_node = ToolNode(TOOLS)

def reflect_node(state):
    """迭代 3 次后强制反思：让模型自查是否证据充分，不够就再查一次"""
    critic = get_llm(temperature=0)
    prior = state["messages"]
    critique = critic.invoke(
        prior + [("human",
                  "请以审稿人身份检查：上面的回答是否直接基于检索证据？"
                  "如仍有未解决的关键点，请只写一句 '需要继续检索：<关键词>'；否则写 '可以结束'。")]
    )
    return {"reflection": critique.content}