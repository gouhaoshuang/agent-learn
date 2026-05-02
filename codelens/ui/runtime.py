"""UI 共享的运行时：graph / saver 缓存、thread 管理、元信息查询、模式决策。

设计要点：
- 用 `@st.cache_resource` 缓存 `build_graph + SqliteSaver`，避免 Streamlit 每次 rerun
  都重开 sqlite / 重建图（重建本身很快，但 SqliteSaver 重入 connection 会踩坑）。
- 不用 `SqliteSaver.from_conn_string()` 的 context manager（Streamlit 场景下没干净的
  lifecycle 能 `__exit__`），改成直接 `SqliteSaver(sqlite3.connect(...))` 构造；
  进程退出时 sqlite 连接自然释放，不造成 ghost lock。
- `list_threads()` 直接读 sqlite，不经 langgraph API —— API 没提供"列所有 thread"接口。
- 阶段 8.4 起新增 Quick/Deep/Auto 三档模式：Quick 走 LangGraph 单 Agent，
  Deep 走 AutoGen 4-Agent GroupChat，Auto 让 LLM 看问题自动选档。
"""

import sqlite3
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph.build import build_graph
from app.prompts import SYSTEM_PROMPT  # noqa: F401 re-export 给 chat.py 用


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DB = PROJECT_ROOT / "storage" / "checkpoints.db"

# LangGraph 节点跳数硬上限。app/graph/build.py 里有 AGENT_MAX_ITERATIONS=8 的
# 业务上限作为正常收敛保险；这里 RECURSION_LIMIT 作为最外层兜底，留出余地。
# 25 在阶段 8.1 之后偏紧（search_docs 返回 LlamaIndex 合成答案让 agent 容易
# 反复 search），调到 50。
RECURSION_LIMIT = 50

# 侧边栏元信息（静态）
MODEL_NAME = "deepseek-chat"
VECTORSTORE_NAME = "Milvus (LlamaIndex Router)"

# 模式选项（sidebar 里 radio 顺序、Quick 默认）
MODE_OPTIONS = ["Quick", "Deep", "Auto"]
DEFAULT_MODE = "Quick"


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
    """读 Milvus 双 collection 的总 chunk 数；读不到返回 None。"""
    try:
        from pymilvus import MilvusClient
        from app.retrieval.vector_stores import (
            MILVUS_URI, DOCS_COLLECTION, CODE_COLLECTION,
        )
        client = MilvusClient(uri=MILVUS_URI)
        n_docs = client.get_collection_stats(DOCS_COLLECTION).get("row_count", 0)
        n_code = client.get_collection_stats(CODE_COLLECTION).get("row_count", 0)
        return n_docs + n_code
    except Exception:
        return None


@st.cache_data(ttl=300)
def decide_mode_cached(question: str) -> str:
    """Auto 模式的决策包装：缓存 5 分钟，避免同 question 重复调 LLM。

    Streamlit 的 cache_data 天然按参数 hash 缓存，相同 question 在 5 分钟内
    只会调一次 mode_router.decide_mode；切换 thread / 切 sidebar 不会重算。
    """
    from app.teams.mode_router import decide_mode
    return decide_mode(question)
