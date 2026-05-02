# -*- coding: utf-8 -*-
"""
Auto 模式：根据问题自动决定走 Quick (LangGraph 单 Agent) 还是 Deep (AutoGen GroupChat)。

设计要点：
  · 一次轻量 LLM 调用（200~300 token prompt）让 DeepSeek 二选一
  · 输出**严格只允许 "Quick" 或 "Deep"**，靠 prompt 强约束
  · LLM 输出无法解析时走关键词 fallback（典型对比题词汇 → Deep）
  · 关键词 fallback 仍然失败时默认 Quick（更便宜、更快，对简单题影响最小）

判断标准（写进 prompt 的核心规则）：
    Quick：单一来源、概念解释、API 用法、定义类问题
    Deep ：跨域对比、多目标连问、文档 vs 源码一致性、需要多视角质疑

为什么不直接复用 LlamaIndex Router 的 selector？
  · Router 的 selector 是"选检索策略"（doc / code / multi_aspect），
    它选 multi 不等于必须走 Deep——SubQuestion 在 Quick 模式下也能跑。
  · 一个独立的"模式选择器"语义更纯净，prompt 也能更专注于"问题复杂度"
    而非"问题类型"。
"""

from __future__ import annotations

import os
import re
from typing import Literal

from dotenv import load_dotenv


# .env 在仓库根目录
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))


# 触发 Deep 的关键词（覆盖中英）。命中任一就路由到 Deep。
# 写得偏保守——这只是 LLM fallback 时的兜底，宁可漏判（让 Quick 接），不要错判
# （让 Deep 烧钱）。
_DEEP_KEYWORDS = [
    "对比", "差异", "区别", "比较", "对照", "异同", "一致",
    "vs", " vs ", "vs.", "对照", "并列",
    "和.*的差异", "和.*的区别", "和.*的不同",
    "compare", "difference", "vs ",
]
_DEEP_PATTERN = re.compile("|".join(_DEEP_KEYWORDS), re.IGNORECASE)


_DECIDE_PROMPT = """你是 CodeLens 的模式路由器。根据用户问题选择处理模式：

【Quick 模式】适用于：
  · 单一来源就能回答（只看文档 或 只看代码）
  · 概念解释、API 用法、定义、配置项查询
  · 简单的"是什么 / 怎么用 / 在哪里"类问题

【Deep 模式】适用于：
  · 跨域对比题（"X 和 Y 的差异 / 对比 / 一致性"）
  · 多目标连问（同时涉及多个类、多个文件、文档 vs 源码）
  · 需要多视角质疑、批判性审视的问题
  · 任何带"对比 / 差异 / 一致 / 是否相同 / vs"等关键词的问题

用户问题：{question}

只输出一个词：Quick 或 Deep
不要任何解释、不要标点、不要换行后多余内容。"""


def _llm_decide(question: str) -> str | None:
    """调一次 DeepSeek 让它在 Quick / Deep 二选一。失败返回 None。"""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
        )
        resp = client.chat.completions.create(
            model="deepseek-chat",   # ⚠️ 别名，详见 agents.py 的踩坑注释
            messages=[
                {"role": "user", "content": _DECIDE_PROMPT.format(question=question)}
            ],
            temperature=0.0,         # 决策类调用完全 deterministic
            max_tokens=10,           # 只期望"Quick"或"Deep"一个词，硬截断防过度生成
            timeout=15,
        )
        text = (resp.choices[0].message.content or "").strip()
        # LLM 可能输出 "Quick"、"Deep"、"Quick模式"、"应该用 Deep" 等变体
        # 用宽松匹配抓第一个出现的关键词
        match = re.search(r"\b(quick|deep)\b", text, re.IGNORECASE)
        if match:
            word = match.group(1).lower()
            return "Quick" if word == "quick" else "Deep"
        return None
    except Exception as e:
        print(f"[mode_router] LLM decide failed: {e}")
        return None


def _keyword_fallback(question: str) -> str:
    """关键词路由：命中 Deep 关键词走 Deep，否则 Quick。"""
    if _DEEP_PATTERN.search(question):
        return "Deep"
    return "Quick"


def decide_mode(question: str, verbose: bool = False) -> Literal["Quick", "Deep"]:
    """根据问题决定走 Quick 还是 Deep 模式。

    决策顺序：
      1. 一次轻量 LLM 调用，prompt 强约束输出 "Quick" / "Deep"
      2. LLM 解析失败 → 关键词 fallback
      3. 关键词也未命中 → 默认 Quick（更便宜，对简单题影响最小）

    Args:
        question: 用户原始问题
        verbose: True 时打印决策过程（哪一层定的、用了哪个关键词）

    Returns:
        "Quick" 或 "Deep"
    """
    decision = _llm_decide(question)
    if decision is not None:
        if verbose:
            print(f"[mode_router] LLM 决策: {decision}")
        return decision

    fallback = _keyword_fallback(question)
    if verbose:
        match = _DEEP_PATTERN.search(question)
        if match:
            print(f"[mode_router] LLM 失败，关键词 fallback: Deep (命中 '{match.group(0)}')")
        else:
            print(f"[mode_router] LLM 失败、无关键词命中，默认: Quick")
    return fallback
