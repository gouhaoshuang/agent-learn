"""CodeLens Streamlit 入口。

运行：
    cd /data/ghs/agent-learn/codelens
    streamlit run ui/app.py
"""

import sys
import time
from pathlib import Path

# 把 codelens/ 加到 sys.path：让 `from app.*`（代码/索引/图）和 `from ui.*`（UI 模块）
# 都可 import。streamlit 只会把 ui/ 目录加进去，这里补上 codelens/。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from ui.chat import replay_history, run_turn_streaming  # noqa: E402
from ui.runtime import get_graph_and_saver  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402


def _bootstrap_session_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"web-{int(time.time())}"


def main() -> None:
    st.set_page_config(
        page_title="CodeLens",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("CodeLens")
    st.caption("LangGraph ReAct Agent · C++ / Markdown RAG")

    _bootstrap_session_state()
    graph, _saver = get_graph_and_saver()

    render_sidebar()

    thread_id = st.session_state.thread_id

    # 历史回放（切 thread 后点任何按钮都会 rerun，这里会重新画一次）
    replay_history(graph, thread_id)

    # 输入框：置于页面底部（Streamlit chat_input 自动 pin 到底）
    question = st.chat_input("Ask about the codebase...")
    if question:
        run_turn_streaming(graph, question, thread_id)


if __name__ == "__main__":
    main()
