"""消息级渲染原语。

设计要点：
- 每个工具调用 + 结果渲染成一个可折叠 `st.expander`，默认折叠，点开看 args/result。
- 工具调用抵达（AI message）时暂时用 `st.empty()` 占位显示 "Running..."；
  工具结果（ToolMessage）抵达时替换占位为完整 expander。这是 Streamlit 中
  实现"渐进式填充"UI 的标准手法。
"""

import json
from typing import Any

import streamlit as st


def render_user_message(content: str) -> None:
    with st.chat_message("user"):
        st.markdown(content)


def _format_args_preview(args: dict) -> str:
    """给 expander 标题用的一行参数预览。"""
    try:
        parts = []
        for k, v in args.items():
            s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else f'"{v}"'
            if len(s) > 40:
                s = s[:37] + "..."
            parts.append(f"{k}={s}")
        preview = ", ".join(parts)
        return preview if len(preview) <= 80 else preview[:77] + "..."
    except Exception:
        return str(args)[:80]


def render_tool_expander_complete(tc: dict, result: str) -> None:
    """一次性渲染完整的 tool_call + result expander（用于历史回放）。"""
    name = tc.get("name", "?")
    args = tc.get("args", {}) or {}
    label = f"Tool: {name}({_format_args_preview(args)})"
    with st.expander(label, expanded=False):
        st.markdown("**Arguments**")
        st.json(args)
        st.markdown("**Result**")
        st.code(result or "", language="text")


def claim_tool_slot(tc: dict) -> Any:
    """流式场景下：工具调用消息先到、工具结果还在跑。
    先拿一个 st.empty 作为占位，显示 "Running..." 文案。
    返回这个占位 slot，用于结果到来后回填。
    """
    slot = st.empty()
    name = tc.get("name", "?")
    args = tc.get("args", {}) or {}
    with slot.container():
        st.caption(f"Running tool: {name}({_format_args_preview(args)}) ...")
    return slot


def fill_tool_slot(slot: Any, tc: dict, result: str) -> None:
    """结果回来了，把占位 slot 替换成完整的 expander。"""
    name = tc.get("name", "?")
    args = tc.get("args", {}) or {}
    with slot.container():
        with st.expander(f"Tool: {name}({_format_args_preview(args)})", expanded=False):
            st.markdown("**Arguments**")
            st.json(args)
            st.markdown("**Result**")
            st.code(result or "", language="text")


# ---------------------------------------------------------
# GroupChat（Deep 模式）渲染原语
# ---------------------------------------------------------

# 每个 Agent 一个 emoji 让对话流更易扫读（Streamlit 的 chat_message 支持
# 字符串作为 avatar——注意是 emoji 字符串而非 path，简单用就够了）
AGENT_AVATAR = {
    "user":      "🧑",
    "retriever": "🔍",
    "analyst":   "📖",
    "critic":    "🧐",
    "reporter":  "✍️",
    "_system":   "⚙️",
}


def render_groupchat_event(m: dict) -> None:
    """渲染 GroupChat 的一条**已完成**消息（不处理 ModelClientStreamingChunkEvent）。

    流式 token 级 chunk 由 chat.py 的循环用 placeholder 自己管理（避免每个 token
    单独冒一个气泡）；本函数只负责"消息级"事件的最终渲染。

    传入的 m 是 dict {source, type, content}：
      · TextMessage             → 文本气泡
      · ToolCallRequestEvent    → 工具调用 expander（"准备调"）
      · ToolCallExecutionEvent  → 工具结果 expander（"调完了"）
      · 其它（ThoughtEvent 等） → 折叠 caption，不抢视线
    """
    source = m.get("source", "_unknown")
    typ = m.get("type", "")
    content = m.get("content", "") or ""

    # 用户消息已经在 _run_turn_groupchat 里单独渲染过了，这里跳过
    if source == "user":
        return

    # 流式 chunk：调用方自己维护 placeholder，本函数不渲染（避免每个 token 一个气泡）
    if typ == "ModelClientStreamingChunkEvent":
        return

    avatar = AGENT_AVATAR.get(source, "🤖")

    if typ == "TextMessage":
        with st.chat_message(source, avatar=avatar):
            st.markdown(f"**{source}**")
            st.markdown(content)
        return

    if typ == "ToolCallRequestEvent":
        with st.chat_message(source, avatar=avatar):
            st.markdown(f"**{source}** · 调工具")
            with st.expander(f"ToolCall: {content[:100]}", expanded=False):
                st.code(content, language="text")
        return

    if typ == "ToolCallExecutionEvent":
        with st.chat_message(source, avatar=avatar):
            st.markdown(f"**{source}** · 工具结果")
            with st.expander("ToolResult（含引用片段）", expanded=False):
                st.code(content, language="text")
        return

    # 其它事件（ThoughtEvent 等）——简单 caption 不抢视线
    with st.chat_message(source, avatar=avatar):
        st.caption(f"{source} · {typ}")
        if content:
            with st.expander("详情", expanded=False):
                st.code(content, language="text")
