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
    """把一条用户问题喂进图，流式打印本轮新产生的消息：
      · AI 文本内容 → token 级流式（stream_mode="messages"，边生成边打印）
      · human / tool / system → 一次性打整条（stream_mode="values"）
      · AI 的 tool_calls → 等 AIMessage 完成时从 values 里拿完整清单补打

    两种 stream_mode 一起开，LangGraph 会以 (mode, payload) 元组逐个 yield。
    """
    prev_msgs = _snapshot_messages(graph, config)
    seen = len(prev_msgs)

    msgs = [] if prev_msgs else [SystemMessage(SYSTEM)]
    msgs.append(HumanMessage(question))
    init = {"messages": msgs, "iterations": 0}

    ai_streaming = False  # 当前是否正在"边打印 AI token 边走"

    def _end_ai_line():
        """给正在流式输出的 AI 行补个换行收尾。"""
        nonlocal ai_streaming
        if ai_streaming:
            print()
            ai_streaming = False

    for mode, payload in graph.stream(
        init, config=config, stream_mode=["values", "messages"]
    ):
        if mode == "messages":
            chunk, meta = payload
            # 只打 agent 节点的 LLM 输出；reflect 节点的 critic 也会流 token，
            # 但它只写 state.reflection、不入 messages，没必要进终端。
            if meta.get("langgraph_node") not in ("agent", None):
                continue
            text = getattr(chunk, "content", "") or ""
            if not text:
                continue  # tool_call 构造阶段的空 chunk、纯 metadata 都跳过
            if not ai_streaming:
                print("[ai] ", end="", flush=True)
                ai_streaming = True
            print(text, end="", flush=True)
            continue

        # mode == "values"：完整 state 快照；只处理 seen 之后新冒出来的消息
        all_msgs = payload["messages"]
        while seen < len(all_msgs):
            m = all_msgs[seen]
            seen += 1
            if m.type == "ai":
                # AI 的文字内容已通过 messages 模式流过了，这里只做两件事：
                #   1) 给流式行补换行
                #   2) 把 tool_calls 清单打出来（messages chunk 里的 tool_call 是
                #      边解析边拼的，从这里拿已组装好的完整 args 最稳）
                _end_ai_line()
                tcs = getattr(m, "tool_calls", None)
                has_content = bool((getattr(m, "content", "") or "").strip())
                if tcs and not has_content:
                    print("[ai]")  # 纯 tool_call，没流过内容，补个头避免孤儿 tool_call 行
                if tcs:
                    for tc in tcs:
                        print(f"  ↳ tool_call: {tc['name']}({tc['args']})")
            else:
                _end_ai_line()
                _print_message(m)  # human / tool / system 整条打

    _end_ai_line()  # 循环结束兜底换行


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
