"""聊天区核心逻辑：历史回放 + 一轮流式问答。

流式策略延续 `scripts/run_cli_memory.py`：
  · stream_mode=["values", "messages"] 同时开
  · "messages" 负责 AI 文字的 token 流
  · "values" 负责工具调用 / 工具结果 / 历史去重
"""

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from ui.render import (
    claim_tool_slot,
    fill_tool_slot,
    render_tool_expander_complete,
    render_user_message,
)
from ui.runtime import SYSTEM_PROMPT, build_config, load_thread_messages


# ---------------------------------------------------------
# 历史回放：进入页面 / 切 thread 时把 sqlite 里存着的聊到什么地步重现出来
# ---------------------------------------------------------

def replay_history(graph, thread_id: str) -> None:
    msgs = load_thread_messages(graph, thread_id)
    i = 0
    n = len(msgs)
    while i < n:
        m = msgs[i]
        t = getattr(m, "type", None)

        if t == "system":
            i += 1
            continue

        if t == "human":
            render_user_message(m.content or "")
            i += 1
            continue

        if t == "ai":
            # 把这条 ai + 紧邻的 tool 消息合并到一个 assistant 气泡里渲染。
            # 这样 ReAct 的一个推理步骤在 UI 上是一个整体，避免被切成 N 个气泡。
            with st.chat_message("assistant"):
                content = (m.content or "").strip()
                if content:
                    st.markdown(content)
                tool_calls = getattr(m, "tool_calls", None) or []
                if tool_calls:
                    # 收集紧随的 ToolMessage，用 tool_call_id 匹配结果
                    j = i + 1
                    tool_results = {}
                    while j < n and getattr(msgs[j], "type", None) == "tool":
                        tm = msgs[j]
                        tcid = getattr(tm, "tool_call_id", None)
                        if tcid is not None:
                            tool_results[tcid] = tm.content or ""
                        j += 1
                    for tc in tool_calls:
                        render_tool_expander_complete(tc, tool_results.get(tc.get("id"), ""))
                    i = j
                    continue
            i += 1
            continue

        # tool 消息如果 orphan（前面没对应 ai）就跳过
        i += 1


# ---------------------------------------------------------
# 流式新轮：执行一条新提问并实时渲染
# ---------------------------------------------------------

def run_turn_streaming(graph, question: str, thread_id: str) -> None:
    config = build_config(thread_id)
    prev_msgs = load_thread_messages(graph, thread_id)
    seen = len(prev_msgs)  # 从这里往后才是"本轮新增"

    # 1. 先把用户消息气泡 draw 出来（此刻还没写回 state，纯 UI）
    render_user_message(question)

    # 2. 构造 init：若 thread 空才注入 SystemMessage
    msgs = [] if prev_msgs else [SystemMessage(SYSTEM_PROMPT)]
    msgs.append(HumanMessage(question))
    init = {"messages": msgs, "iterations": 0}

    # 3. 开一个 assistant 气泡，流式内容都塞进去
    with st.chat_message("assistant"):
        text_slot = None           # 当前 AI 文字段的 st.empty()
        text_buf = ""              # 文字累积缓冲
        tool_slots = {}            # tool_call_id -> (slot, tc)

        try:
            for mode, payload in graph.stream(
                init, config=config, stream_mode=["values", "messages"]
            ):
                if mode == "messages":
                    chunk, meta = payload
                    # reflect 节点的 critic 也会流 token，但它不入 state.messages，
                    # 别打到 UI 上。
                    if meta.get("langgraph_node") not in ("agent", None):
                        continue
                    delta = getattr(chunk, "content", "") or ""
                    if not delta:
                        continue
                    if text_slot is None:
                        text_slot = st.empty()
                        text_buf = ""
                    text_buf += delta
                    text_slot.markdown(text_buf)
                    continue

                # mode == "values"
                all_msgs = payload["messages"]
                while seen < len(all_msgs):
                    m = all_msgs[seen]
                    seen += 1
                    t = getattr(m, "type", None)

                    if t == "ai":
                        # 当前 AI 文字段收尾：让 text_slot 的内容固化，下一段新建 slot
                        text_slot = None
                        text_buf = ""
                        # 工具调用：给每个 tool_call 占一个 slot，等结果来了回填
                        for tc in getattr(m, "tool_calls", None) or []:
                            tcid = tc.get("id")
                            tool_slots[tcid] = (claim_tool_slot(tc), tc)

                    elif t == "tool":
                        tcid = getattr(m, "tool_call_id", None)
                        entry = tool_slots.pop(tcid, None)
                        if entry is None:
                            # 少见：tool 消息没匹配到 slot，直接渲染一个独立 expander
                            fake_tc = {"name": getattr(m, "name", "?"), "args": {}}
                            render_tool_expander_complete(fake_tc, m.content or "")
                        else:
                            slot, tc = entry
                            fill_tool_slot(slot, tc, m.content or "")
                    # human / system 跳过（不会出现在"新增"里，但保险起见）

        except Exception as e:
            st.error(f"运行出错：{type(e).__name__}: {e}")
