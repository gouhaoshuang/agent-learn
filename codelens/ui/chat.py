"""聊天区核心逻辑：历史回放 + 一轮模式分发问答。

模式分发（阶段 8.4 起）：
  · Quick → 走 LangGraph 单 Agent，**token 级**流式输出
  · Deep  → 走 AutoGen 4-Agent GroupChat，**消息级**流式输出（每个 Agent 发完
           一段就立刻显示，每次工具调用 / 工具结果立刻显示）
  · Auto  → 先调 mode_router.decide_mode 选档，UI 上 caption 决策结果，再 dispatch

Quick 与 Deep 流式粒度的差异：
  · Quick 用 LangGraph 的 `stream_mode=["values", "messages"]`，能拿到每个
    LLM token，体验是字符一个一个蹦出来。
  · Deep 用 AutoGen 的 `team.run_stream()`，事件粒度是"一条完整消息"——
    每个 Agent 生成完一段话才 yield。所以 UI 上是 4 个角色依次出现，
    单条消息内部不是 token 流。这是 AutoGen 0.7.5 的内置粒度，不是 UI 实现的限制。
  · Streamlit 主线程同步：用 `asyncio.new_event_loop()` + `gen.__anext__()` 拉
    单个事件、立刻渲染——streamlit 的 widget 调用会即时 push 到浏览器，
    所以即使在 sync loop 里也能"边拉边出"，无需后台线程 + queue。
"""

import asyncio

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from ui.render import (
    claim_tool_slot,
    fill_tool_slot,
    render_groupchat_event,
    render_tool_expander_complete,
    render_user_message,
)
from ui.runtime import (
    DEFAULT_MODE,
    SYSTEM_PROMPT,
    build_config,
    decide_mode_cached,
    load_thread_messages,
)


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
# 入口：按 sidebar 选的模式 dispatch（Quick / Deep / Auto）
# ---------------------------------------------------------

def run_turn_streaming(graph, question: str, thread_id: str) -> None:
    """新一轮提问的统一入口；按 session_state['mode'] 分发到对应实现。"""
    mode = st.session_state.get("mode", DEFAULT_MODE)

    # Auto: 先决策、UI 上提示一下决策结果、再 dispatch 到 Quick/Deep
    if mode == "Auto":
        with st.spinner("Auto 路由判定中..."):
            decided = decide_mode_cached(question)
        st.toast(f"Auto 决策: {decided}", icon="🧭")
        mode = decided

    if mode == "Deep":
        _run_turn_groupchat(question)
    else:
        _run_turn_langgraph(graph, question, thread_id)


# ---------------------------------------------------------
# Quick：原有 LangGraph 单 Agent 流式
# ---------------------------------------------------------

def _run_turn_langgraph(graph, question: str, thread_id: str) -> None:
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


# ---------------------------------------------------------
# Deep：AutoGen 4-Agent GroupChat（消息级流式）
# ---------------------------------------------------------

def _run_turn_groupchat(question: str) -> None:
    """Deep 模式：流式跑一次 GroupChat。

    流式粒度是**双层**的：
      · 消息级（默认）：每条 ToolCall / TextMessage 抵达就调 render_groupchat_event
      · token 级（agents.py 里 model_client_stream=True 才会出现）：
        同一 source 连续的 ModelClientStreamingChunkEvent 累积到**同一个**
        st.empty() placeholder，等到对应 TextMessage（完整版）抵达时收尾。
        这样 token 级流式只在一个气泡内累积，不会每个 token 冒一个新气泡。

    Deep 模式不接 LangGraph 的 SqliteSaver——每次提问都是独立的 GroupChat
    会话，不接续 thread 历史。
    """
    from app.teams.runner import stream_groupchat
    from ui.render import AGENT_AVATAR

    render_user_message(question)
    st.caption("🤝 Deep mode · 4-Agent GroupChat 流式协作中（约 1-2 分钟）")

    final_answer = ""
    stop_reason = None
    tool_call_names: list[str] = []

    # token 级 chunk 聚合状态：
    #   source —— 当前正在流的 Agent 名字（chunk 切换 source 时开新气泡）
    #   buffer —— 累积的文本（每来一个 chunk 追加一段）
    #   slot   —— st.empty() placeholder，每次 chunk 到达就 .markdown(buffer) 更新
    chunk_state = {"source": None, "buffer": "", "slot": None}

    def _reset_chunk_state():
        chunk_state["source"] = None
        chunk_state["buffer"] = ""
        chunk_state["slot"] = None

    loop = asyncio.new_event_loop()
    try:
        gen = stream_groupchat(question, use_selector=False).__aiter__()
        while True:
            try:
                ev = loop.run_until_complete(gen.__anext__())
            except StopAsyncIteration:
                break

            typ = ev.get("type", "")
            src = ev.get("source", "")
            content = ev.get("content", "") or ""

            if typ == "TaskResult":
                stop_reason = content
                continue

            # ---------- token 级流式：聚合到同一气泡的 placeholder ----------
            if typ == "ModelClientStreamingChunkEvent":
                if chunk_state["source"] != src:
                    # 新 source：开新气泡 + 新 placeholder
                    avatar = AGENT_AVATAR.get(src, "🤖")
                    with st.chat_message(src, avatar=avatar):
                        st.markdown(f"**{src}**")
                        chunk_state["slot"] = st.empty()
                    chunk_state["source"] = src
                    chunk_state["buffer"] = ""
                chunk_state["buffer"] += content
                if chunk_state["slot"] is not None:
                    chunk_state["slot"].markdown(chunk_state["buffer"])
                continue

            # ---------- 同 source 的 TextMessage：流结束，最后精确替换一次 ----------
            # 用 TextMessage.content 替换累积 buffer（防止累积时的 markdown 中间态错乱），
            # 同时收集 reporter 的最终答案、然后清空 chunk state。
            if typ == "TextMessage" and chunk_state["source"] == src:
                if chunk_state["slot"] is not None and content:
                    chunk_state["slot"].markdown(content)
                if src == "reporter":
                    final_answer = content.strip()
                _reset_chunk_state()
                continue

            # ---------- 其它消息级事件：先重置 chunk state、再走常规渲染 ----------
            _reset_chunk_state()
            render_groupchat_event(ev)

            if src == "reporter" and typ == "TextMessage":
                final_answer = content.strip()
            if typ == "ToolCallRequestEvent" and "name='" in content:
                name = content.split("name='", 1)[1].split("'", 1)[0]
                tool_call_names.append(name)

    except Exception as e:
        st.error(f"GroupChat 运行出错：{type(e).__name__}: {e}")
        return
    finally:
        loop.close()

    # 去掉 reporter 答案末尾的 DONE 标记
    for marker in ("\nDONE", "DONE."):
        if marker in final_answer:
            final_answer = final_answer.split(marker)[0].rstrip()
    if final_answer.endswith("DONE"):
        final_answer = final_answer[:-4].rstrip()

    # 收尾：最终答案高亮 + 元信息
    st.markdown("---")
    st.markdown("### 最终答案")
    st.markdown(final_answer or "(reporter 未给出最终答案，请查看上方对话流)")
    st.caption(
        f"stop_reason: `{stop_reason or 'unknown'}`  ·  "
        f"tool_calls: {tool_call_names or '(none)'}"
    )
