# -*- coding: utf-8 -*-
"""
GroupChat 装配 + 终止条件（阶段 8.3）。

起步用 RoundRobinGroupChat：固定顺序 Retriever → Analyst → Critic → Reporter
→ Retriever → ...，跑通后再升级 SelectorGroupChat（让 LLM 看局面挑下一个谁
说话）。RoundRobin 实现简单、调试容易，是基线方案。

终止条件（必须有保险丝）：
    TextMentionTermination("DONE")  —— Reporter 写完最终答案以 DONE 结尾时触发
    MaxMessageTermination(12)       —— 最多 12 条消息兜底
                                       4 个 Agent × 3 轮 = 12，正常 1 轮就该结束
                                       如果超 12 条还没说 DONE 说明 Critic 反复补查
                                       陷入循环，直接停掉避免烧 token。
"""

from __future__ import annotations


def build_team(use_selector: bool = False):
    """构造一个 4-Agent 团队。

    Args:
        use_selector: True 用 SelectorGroupChat（LLM 决定下一个谁发言），
                      False 用 RoundRobinGroupChat（固定顺序）。默认 False。

    Returns:
        Team 实例（autogen_agentchat.teams.Base*GroupChat），可直接调
        `await team.run(task=...)` 或 `team.run_stream(task=...)`。
    """
    from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
    from autogen_agentchat.conditions import (
        TextMentionTermination, MaxMessageTermination,
    )

    from app.teams.agents import (
        get_model_client, make_retriever, make_analyst,
        make_critic, make_reporter,
    )

    # 共享同一个 model client：减少 httpx 客户端实例数量；如果之后想给每个 Agent
    # 独立超时/温度，再拆开各自构造即可。
    client = get_model_client()
    retriever = make_retriever(client)
    analyst = make_analyst(client)
    critic = make_critic(client)
    reporter = make_reporter(client)

    termination = (
        TextMentionTermination("DONE")
        | MaxMessageTermination(12)
    )

    if use_selector:
        # SelectorGroupChat 自己也需要一个 LLM 来选下一个发言者
        return SelectorGroupChat(
            [retriever, analyst, critic, reporter],
            model_client=client,
            termination_condition=termination,
            allow_repeated_speaker=False,    # 不让同一个 Agent 连续发言
        )

    return RoundRobinGroupChat(
        [retriever, analyst, critic, reporter],
        termination_condition=termination,
    )
