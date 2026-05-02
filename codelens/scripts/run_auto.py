# -*- coding: utf-8 -*-
"""
Auto 模式命令行入口（阶段 8.4 前置）。

根据问题自动判定 Quick / Deep，再分发到对应的执行栈：
  · Quick → LangGraph 单 Agent ReAct（沿用 `scripts/run_cli.py` 那条管线）
  · Deep  → AutoGen 4-Agent GroupChat（`scripts/run_groupchat.py` 那条管线）

用法：
    python scripts/run_auto.py "ThreadPool 怎么实现？"            # → Quick
    python scripts/run_auto.py "对比 ThreadPool 的文档建议和源码实现"  # → Deep
    python scripts/run_auto.py --force quick "..."   # 强制走 Quick，跳过决策
    python scripts/run_auto.py --force deep "..."    # 强制走 Deep
    python scripts/run_auto.py --selector "..."      # Deep 模式用 SelectorGroupChat

环境变量（沿用 retrieval/teams 那套）：
    CUDA_VISIBLE_DEVICES=4   先看 nvidia-smi 挑空闲卡
    HF_HUB_OFFLINE=1         离线加载 embedding，规避 hf-mirror.com 卡顿
"""

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_quick(question: str) -> None:
    """走 Quick 模式：LangGraph 单 Agent ReAct，复用现有 run_cli 的执行逻辑。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from app.graph.build import build_graph
    from app.prompts import SYSTEM_PROMPT

    graph = build_graph()
    init = {
        "messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)],
        "iterations": 0,
    }

    print("─" * 60)
    print("[Quick mode] LangGraph 单 Agent ReAct")
    print("─" * 60)

    final_text = ""
    for event in graph.stream(init, stream_mode="values"):
        last = event["messages"][-1]
        head = f"[{last.type}]"
        if getattr(last, "name", None):
            head = f"[tool:{last.name}]"
        body = (getattr(last, "content", "") or "")
        print(f"\n{head}")
        print(body[:500] + ("..." if len(body) > 500 else ""))
        if last.type == "ai" and last.content and not getattr(last, "tool_calls", None):
            final_text = last.content

    print("\n" + "=" * 60)
    print(">>> Quick 最终答案 <<<\n")
    print(final_text)


async def _run_deep_stream(question: str, use_selector: bool) -> None:
    """走 Deep 模式：AutoGen 4-Agent GroupChat，流式打印每个角色的发言。"""
    from app.teams.runner import stream_groupchat

    print("─" * 60)
    print(f"[Deep mode] AutoGen 4-Agent "
          f"{'SelectorGroupChat' if use_selector else 'RoundRobinGroupChat'}")
    print("─" * 60)

    async for ev in stream_groupchat(question, use_selector=use_selector):
        src = ev["source"]; typ = ev["type"]; content = ev["content"]
        if len(content) > 600:
            content = content[:600] + "...(truncated)"
        print(f"\n── [{src}/{typ}] " + "─" * (60 - len(src) - len(typ)))
        print(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="要问的问题")
    ap.add_argument("--force", choices=["quick", "deep"], default=None,
                    help="跳过 Auto 决策，强制走指定模式")
    ap.add_argument("--selector", action="store_true",
                    help="Deep 模式时用 SelectorGroupChat（默认 RoundRobin）")
    args = ap.parse_args()

    question = " ".join(args.question) or "ThreadPool 是怎么实现的？"
    print(f"[Q] {question}\n")

    # 决定模式
    if args.force == "quick":
        mode = "Quick"
        print(f"[mode] 强制 Quick（跳过 Auto 决策）")
    elif args.force == "deep":
        mode = "Deep"
        print(f"[mode] 强制 Deep（跳过 Auto 决策）")
    else:
        from app.teams.mode_router import decide_mode
        mode = decide_mode(question, verbose=True)
        print(f"[mode] Auto 决策: {mode}")

    # 分发
    if mode == "Quick":
        _run_quick(question)
    else:
        asyncio.run(_run_deep_stream(question, use_selector=args.selector))


if __name__ == "__main__":
    main()
