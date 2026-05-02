# -*- coding: utf-8 -*-
"""
AutoGen 4-Agent GroupChat 命令行入口（阶段 8.3）。

用法：
    python scripts/run_groupchat.py "对比 ThreadPool 和 Server 的加锁策略差异"
    python scripts/run_groupchat.py --selector "..."        # 用 SelectorGroupChat
    python scripts/run_groupchat.py --stream "..."          # 流式打印（推荐）

环境变量（沿用 retrieval/ 那套）：
    CUDA_VISIBLE_DEVICES=4   先看 nvidia-smi 挑空闲卡，避免抢已满的 GPU
    HF_HUB_OFFLINE=1         离线加载 embedding，规避 hf-mirror.com 卡顿
"""

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _print_event(ev: dict) -> None:
    """把单条事件打到 stdout，让用户能像看直播一样看 4 个 Agent 协作。"""
    src = ev["source"]
    typ = ev["type"]
    content = ev["content"]

    # 折断 content 防止 ToolCallRequestEvent 这种长内容刷屏
    if len(content) > 600:
        content = content[:600] + "...(truncated)"

    print(f"\n── [{src}/{typ}] " + "─" * (60 - len(src) - len(typ)))
    print(content)


async def _run_stream(question: str, use_selector: bool):
    from app.teams.runner import stream_groupchat
    async for ev in stream_groupchat(question, use_selector=use_selector):
        _print_event(ev)


def _run_sync(question: str, use_selector: bool):
    from app.teams.runner import run_groupchat
    result = run_groupchat(question, use_selector=use_selector)

    print("=" * 60)
    print("CodeLens GroupChat")
    print("=" * 60)
    for m in result["messages"]:
        _print_event(m)

    print("\n" + "=" * 60)
    print(f"stop_reason: {result['stop_reason']}")
    print(f"tool_calls : {result['tool_calls']}")
    print("=" * 60)
    print("\n>>> 最终答案 <<<\n")
    print(result["final_answer"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="要问的问题（多个词会被空格拼起来）")
    ap.add_argument("--selector", action="store_true",
                    help="用 SelectorGroupChat（LLM 选下一个发言者）；默认 RoundRobin")
    ap.add_argument("--stream", action="store_true",
                    default=True,
                    help="流式打印每个 Agent 的发言；默认一次性打完整结果")
    args = ap.parse_args()

    question = " ".join(args.question) or "对比 ThreadPool 和 Server 在加锁策略上的差异。"
    print(f"[Q] {question}")

    if args.stream:
        asyncio.run(_run_stream(question, use_selector=args.selector))
    else:
        _run_sync(question, use_selector=args.selector)


if __name__ == "__main__":
    main()
