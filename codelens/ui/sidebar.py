"""侧边栏：Mode toggle + Sessions 列表 + New/Clear + 静态元信息。"""

import time

import streamlit as st

from ui.runtime import (
    DEFAULT_MODE,
    MODE_OPTIONS,
    MODEL_NAME,
    VECTORSTORE_NAME,
    index_size,
    list_threads,
)


# 每档模式的解释文案（hover 时 Streamlit 在 caption 下显示）
_MODE_HELP = {
    "Quick": "LangGraph 单 Agent ReAct（快、便宜）",
    "Deep":  "AutoGen 4-Agent GroupChat: Retriever→Analyst→Critic→Reporter（慢、深）",
    "Auto":  "LLM 看问题自动挑 Quick / Deep（多一次轻量 LLM 决策）",
}


def _switch_thread(new_id: str) -> None:
    st.session_state.thread_id = new_id
    st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        # ---------- 模式 toggle（最显眼，置顶） ----------
        st.markdown("### Mode")
        mode = st.radio(
            "运行模式",
            MODE_OPTIONS,
            index=MODE_OPTIONS.index(st.session_state.get("mode", DEFAULT_MODE)),
            label_visibility="collapsed",
            horizontal=True,
            key="mode",
        )
        st.caption(_MODE_HELP.get(mode, ""))

        st.markdown("---")
        st.markdown("### Sessions")

        current = st.session_state.get("thread_id")
        threads = list_threads()

        # New / Clear 放在列表上方（最常点，不用滚动）
        col_new, col_clear = st.columns(2)
        with col_new:
            if st.button("New", use_container_width=True):
                _switch_thread(f"web-{int(time.time())}")
        with col_clear:
            # "Clear" 不删 sqlite 里的历史，只是切到一个新 thread。
            # 真删历史需要更谨慎的确认流程，MVP 不做。
            if st.button("Clear", use_container_width=True, help="切到新会话（不删历史）"):
                _switch_thread(f"web-{int(time.time())}-new")

        # 当前 thread 高亮（即使不在 sqlite 里也要显示，不然新建后看不到自己）
        all_ids = list(dict.fromkeys([current] + threads)) if current else threads

        for tid in all_ids:
            prefix = "●" if tid == current else "○"
            label = f"{prefix}  {tid}"
            if st.button(label, key=f"thread_btn_{tid}", use_container_width=True):
                if tid != current:
                    _switch_thread(tid)

        st.markdown("---")
        st.markdown("### Info")
        st.markdown(f"- Model: `{MODEL_NAME}`")
        st.markdown(f"- Vectorstore: `{VECTORSTORE_NAME}`")
        n = index_size()
        if n is not None:
            st.markdown(f"- Index: `{n:,}` chunks")
        st.caption(f"Current thread: `{current}`")
