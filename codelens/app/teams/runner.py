# -*- coding: utf-8 -*-
"""
GroupChat 对外接口（阶段 8.3）。

提供两个函数：
    run_groupchat(question)     —— 一次性跑完，返回 Reporter 的最终答案 + 完整对话流
    stream_groupchat(question)  —— 异步生成器，每个 Agent 发言时实时 yield，
                                   给 UI / 终端流式渲染用

设计上保持简单：调用方不需要懂 AutoGen 的 async / TaskResult 内部结构，
拿到 string（或者一串 string 事件）就能用。
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator


async def _run_async(question: str, use_selector: bool = False):
    from app.teams.groupchat import build_team
    team = build_team(use_selector=use_selector)
    return await team.run(task=question)


def run_groupchat(question: str, use_selector: bool = False) -> dict:
    """同步接口：跑一次完整 GroupChat，返回结构化结果。

    Returns:
        dict {
            "final_answer": str,             # Reporter 的最后一段（去掉 DONE）
            "stop_reason":  str,             # 终止原因：DONE / MaxMessage / ...
            "messages":     list[dict],      # 完整对话 [{source, type, content}]
            "tool_calls":   list[str],       # Retriever 调过的工具名（含顺序）
        }
    """
    result = asyncio.run(_run_async(question, use_selector=use_selector))

    messages = []
    tool_calls = []
    final_answer = ""

    for m in result.messages:
        msg_type = type(m).__name__
        # m.content 可能是 str（TextMessage）或 list（ToolCallRequestEvent 等）
        if isinstance(m.content, list):
            content_str = []
            for c in m.content:
                content_str.append(str(c))
                # 提取 tool 调用名
                if hasattr(c, "name") and msg_type == "ToolCallRequestEvent":
                    tool_calls.append(c.name)
            content = " | ".join(content_str)
        else:
            content = m.content or ""
        messages.append({
            "source": m.source,
            "type": msg_type,
            "content": content,
        })

        # Reporter 的最后一段 TextMessage 就是最终答案
        if m.source == "reporter" and msg_type == "TextMessage":
            final_answer = (m.content or "").strip()

    # 把 DONE 关键词从最终答案末尾去掉，给上层显示更干净
    if final_answer.endswith("DONE"):
        final_answer = final_answer[:-4].rstrip()
    # 也处理 "...DONE." 或独立行 "\nDONE" 这种变体
    for marker in ("\nDONE", "DONE."):
        if marker in final_answer:
            final_answer = final_answer.split(marker)[0].rstrip()

    return {
        "final_answer": final_answer,
        "stop_reason": result.stop_reason,
        "messages": messages,
        "tool_calls": tool_calls,
    }


async def stream_groupchat(
    question: str, use_selector: bool = False
) -> AsyncIterator[dict]:
    """异步生成器：每条新消息（含工具调用 / 工具结果 / Agent 文本）都 yield。

    每个 yield 出来的 event 是 dict：
        {"source": str, "type": str, "content": str}

    UI 层可以拿这个流接 Streamlit `st.write_stream` 或自定义渲染。
    """
    from app.teams.groupchat import build_team

    team = build_team(use_selector=use_selector)
    async for m in team.run_stream(task=question):
        msg_type = type(m).__name__

        # team.run_stream 的最后一个 yield 是 TaskResult，type 名是 'TaskResult'
        if msg_type == "TaskResult":
            yield {
                "source": "_system",
                "type": "TaskResult",
                "content": f"stop_reason={m.stop_reason}",
            }
            return

        # 普通消息：和 run_groupchat 里的处理保持一致
        if isinstance(getattr(m, "content", None), list):
            content = " | ".join(str(c) for c in m.content)
        else:
            content = getattr(m, "content", "") or ""

        yield {
            "source": getattr(m, "source", "_unknown"),
            "type": msg_type,
            "content": content,
        }
