# -*- coding: utf-8 -*-
"""
AutoGen 多 Agent 协同层（阶段 8.3）。

目录结构：
    retrieval_tools.py —— 把 LlamaIndex Router/SubQuestion 包成 AutoGen 可调用的
                          Python 函数（Retriever Agent 的工具集）
    agents.py          —— 4 个 AssistantAgent 的定义（Retriever / Analyst /
                          Critic / Reporter）+ DeepSeek 兼容的 model_client 工厂
    groupchat.py       —— RoundRobinGroupChat 装配 + 终止条件
    runner.py          —— `run_groupchat(question) -> str` 对外接口

为什么单独一个 teams/ 模块、而不是混进 graph/？
  · graph/ 是 LangGraph 单 Agent ReAct（Quick 模式）
  · teams/ 是 AutoGen 多 Agent GroupChat（Deep 模式）
  · 两者并存：UI 上由 toggle 决定走哪条线，下层共享 retrieval/ 检索栈
"""
