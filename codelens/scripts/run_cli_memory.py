"""
阶段 4：Memory + 多轮对话 CLI。

- 用 SqliteSaver 把 state 持久化到 storage/checkpoints.db；
  同一 thread_id 的对话**跨进程**可恢复（退出再进来还能接上）。
- 默认进入 REPL：反复读输入 → 流式打印 agent 轨迹 → 继续下一轮。
- 提供 --demo 一键跑经典两轮（"ThreadPool 实现" → "那它的锁粒度呢？"），
  用来快速验证"第二轮能接住第一轮的上下文"。

用法：
    # REPL，默认 thread_id=default
    python scripts/run_cli_memory.py

    # 切到另一个会话
    python scripts/run_cli_memory.py --thread projX

    # 一问一答就退，适合嵌脚本
    python scripts/run_cli_memory.py --once "ThreadPool 如何实现？"

    # 演示多轮记忆（第 2 问会引用第 1 问的 ThreadPool 结论）
    python scripts/run_cli_memory.py --demo

REPL 元命令（以 ':' 开头，避免撞真问题）：
    :q              退出
    :thread <name>  切换到指定 thread_id
    :new            用时间戳生成一个全新 thread_id
"""

import argparse
import sys
import time
from pathlib import Path

# 把 codelens/ 加到 sys.path，让 `from app.*` 在任意 cwd 下都能 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph.build import build_graph


SYSTEM = ("你是 CodeLens，代码/文档智能助手。"
          "优先使用工具获取证据，不要凭空回答。每次调用工具后总结发现。")

CHECKPOINT_DB = PROJECT_ROOT / "storage" / "checkpoints.db"

# 单次 tool 输出在终端最多显示几行（只截显示，完整内容仍进模型 context）
TOOL_LOG_MAX_LINES = 10

# 每轮 ReAct 硬上限。即使模型贪心地反复 tool_call，也会在这里兜底退出，
# 避免长对话累积下 LangGraph 默认 25 被撞穿后抛异常。
RECURSION_LIMIT = 25


# ----- 流式打印（与 run_cli.py 保持一致的显示规则）------------------------

def _format_head(msg) -> str:
    if msg.type == "tool":
        return f"[tool:{getattr(msg, 'name', '?')}]"
    return f"[{msg.type}]"


def _format_body(msg) -> str:
    """只截 tool 消息显示；ai / human / system 原样输出，避免最终答案被折叠。"""
    content = getattr(msg, "content", "") or ""
    if msg.type != "tool":
        return content
    lines = content.splitlines() or [content]
    if len(lines) <= TOOL_LOG_MAX_LINES:
        return content
    preview = "\n".join(lines[:TOOL_LOG_MAX_LINES])
    return f"{preview}\n  ... ({len(lines) - TOOL_LOG_MAX_LINES} more lines truncated)"


def _print_message(msg) -> None:
    print(f"{_format_head(msg)} {_format_body(msg)}")
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            print(f"  ↳ tool_call: {tc['name']}({tc['args']})")


# ----- 一轮问答 ----------------------------------------------------------

def _snapshot_messages(graph, config) -> list:
    """读一下当前 thread 里已有的消息（可能是正常历史轮，也可能是上次崩溃留下的残骸）。"""
    try:
        snap = graph.get_state(config)
        return list(snap.values.get("messages", [])) if snap.values else []
    except Exception:
        return []


def run_turn(graph, question: str, config: dict) -> None:
    """把一条用户问题喂进图，只**增量**打印本轮新产生的消息。"""
    # 本轮开始前 state 里已有多少消息 —— 这些全部视为"历史"，不再重复打印。
    # 即使上轮因网络等原因中途崩溃，checkpointer 也可能把 [System, Human]
    # 留在 state 里；用长度做起点，能正确跳过那些残骸。
    prev_msgs = _snapshot_messages(graph, config)
    print(f"[debug] before: {len(prev_msgs)} msgs in state")   
    seen = len(prev_msgs)

    # 只在 thread 真的是空的时候才注入 SystemMessage，否则 add_messages 会把它
    # 追加到末尾，造成 system 消息堆积。
    msgs = [] if prev_msgs else [SystemMessage(SYSTEM)]
    msgs.append(HumanMessage(question))
    init = {"messages": msgs, "iterations": 0}

    # stream_mode="values" 每步 yield 完整 state 快照；seen 之前的是历史，不再打印。
    for event in graph.stream(init, config=config, stream_mode="values"):
        all_msgs = event["messages"]
        while seen < len(all_msgs):
            _print_message(all_msgs[seen])
            seen += 1


# ----- REPL / demo / once 三种模式 ---------------------------------------

def run_repl(graph, thread_id: str) -> None:
    print(f"CodeLens memory REPL | thread={thread_id}  "
          f"(:q quit, :thread X switch, :new fresh id)")
    while True:
        try:
            line = input(f"\n[{thread_id}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == ":q":
            break
        if line == ":new":
            thread_id = f"session-{int(time.time())}"
            print(f"  (switched to new thread: {thread_id})")
            continue
        if line.startswith(":thread"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                thread_id = parts[1]
                print(f"  (switched to thread: {thread_id})")
            else:
                print("  usage: :thread <name>")
            continue

        config = {"configurable": {"thread_id": thread_id},
                  "recursion_limit": RECURSION_LIMIT}
        run_turn(graph, line, config)


def run_demo(graph) -> None:
    """两轮经典演示：第 2 问依赖第 1 问的上下文，成立即证明记忆生效。"""
    thread_id = f"demo-{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id},
              "recursion_limit": RECURSION_LIMIT}
    for q in ["ThreadPool 是如何实现的？",
              "那它的锁粒度是怎样的？有什么潜在的并发风险？"]:
        print(f"\n========== [demo] thread={thread_id} | Q: {q} ==========")
        run_turn(graph, q, config)


# ----- main --------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="CodeLens 多轮记忆 CLI（阶段 4）")
    ap.add_argument("--thread", default="default", help="会话 thread_id（默认 default）")
    ap.add_argument("--once", default=None, help="非交互：问一个问题就退出")
    ap.add_argument("--demo", action="store_true",
                    help="跑两轮经典演示证明记忆生效，忽略 --thread / --once")
    return ap.parse_args()


def main():
    args = parse_args()
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)

    # SqliteSaver.from_conn_string 是 context manager，必须用 with
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        graph = build_graph(checkpointer=saver)

        if args.demo:
            run_demo(graph)
        elif args.once is not None:
            config = {"configurable": {"thread_id": args.thread},
                      "recursion_limit": RECURSION_LIMIT}
            run_turn(graph, args.once, config)
        else:
            run_repl(graph, args.thread)


if __name__ == "__main__":
    main()
