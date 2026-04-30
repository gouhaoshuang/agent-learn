"""UI 共享的运行时：graph / saver 缓存、thread 管理、元信息查询。

设计要点：
- 用 `@st.cache_resource` 缓存 `build_graph + SqliteSaver`，避免 Streamlit 每次 rerun
  都重开 sqlite / 重建图（重建本身很快，但 SqliteSaver 重入 connection 会踩坑）。
- 不用 `SqliteSaver.from_conn_string()` 的 context manager（Streamlit 场景下没干净的
  lifecycle 能 `__exit__`），改成直接 `SqliteSaver(sqlite3.connect(...))` 构造；
  进程退出时 sqlite 连接自然释放，不造成 ghost lock。
- `list_threads()` 直接读 sqlite，不经 langgraph API —— API 没提供"列所有 thread"接口。
"""

import sqlite3
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph.build import build_graph
from app.prompts import SYSTEM_PROMPT  # noqa: F401 re-export 给 chat.py 用


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DB = PROJECT_ROOT / "storage" / "checkpoints.db"

RECURSION_LIMIT = 25

# 侧边栏元信息（静态）
MODEL_NAME = "deepseek-chat"
VECTORSTORE_NAME = "Chroma"


@st.cache_resource(show_spinner="Initializing CodeLens...")
def get_graph_and_saver():
    """应用生命周期内共享一份 graph + SqliteSaver。"""
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    saver = SqliteSaver(conn)
    graph = build_graph(checkpointer=saver)
    return graph, saver


def build_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }


def list_threads() -> list[str]:
    """从 checkpoints.db 里 query 所有出现过的 thread_id，按字母序。"""
    if not CHECKPOINT_DB.exists():
        return []
    try:
        with sqlite3.connect(str(CHECKPOINT_DB)) as conn:
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.DatabaseError:
        return []


def load_thread_messages(graph, thread_id: str) -> list:
    """读 thread 已有的消息列表（可能是历史，也可能是崩溃残骸）。"""
    try:
        snap = graph.get_state(build_config(thread_id))
        return list(snap.values.get("messages", [])) if snap.values else []
    except Exception:
        return []


@st.cache_data(ttl=60)
def index_size() -> int | None:
    """读 Chroma 集合的 chunk 数；读不到返回 None。"""
    try:
        from app.retriever import _get_chroma  # noqa: WPS433 — 内部 API
        return _get_chroma()._collection.count()
    except Exception:
        return None
