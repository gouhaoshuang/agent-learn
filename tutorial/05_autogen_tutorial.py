"""
===========================================================
AutoGen 入门教程（autogen-agentchat 0.4+ 新 API）
===========================================================
本教程承接 01~04（LangChain / LangGraph / Milvus / LlamaIndex）。
前几份让你能调模型、能搭单 Agent、能做 RAG；本教程教你 **多 Agent 协同**：
让多个有"人格"的 Agent 在同一个对话里互相磋商、辩论、纠错，最后给出
比单 Agent 更好的答案。

⚠️⚠️⚠️ 重要前提：版本警告 ⚠️⚠️⚠️
-----------------------------------------------------------
AutoGen 在 2024 年经历过一次大重构。**不要看老教程！**

  · 老 API：`pyautogen` 包（0.2.x），`ConversableAgent`/`AssistantAgent`/
            `UserProxyAgent`/`GroupChat`，**全部同步**。YouTube/掘金/CSDN
            搜 "AutoGen 教程" 大半是这个版本，跟着写代码无法升级。
  · 新 API：`autogen-agentchat`（0.4+），微软主推的官方版本，
            **全部 async**，runtime 完全重写。本教程用的就是这个。

辨别方法：看 import 语句。
    `from autogen import ...`            ← 老的 pyautogen，别用
    `from autogen_agentchat.agents ...`  ← 新的 agentchat，本教程
-----------------------------------------------------------

为什么需要 AutoGen？（和 LangGraph 的差别）
  · LangGraph 是"图节点 + state"，节点之间走结构化 message passing；
  · AutoGen 是"多个有人格的 Agent 在对话"，agent 之间走自然语言交互，
    更适合写"角色分工型"协作（评审者、重构者、批判者、汇总者）。
  · LangGraph 适合**确定性流程**（A → B → 条件 → C），AutoGen 适合
    **开放式磋商**（多 Agent 自由发言、由 Selector 决定下一个谁说）。
  · 两者不是替代关系：CodeLens 真实改造方案里两者并存——
    LangGraph 跑"快速模式"单 Agent ReAct；AutoGen 跑"深度模式"
    多 Agent GroupChat。

AutoGen 核心抽象（一定要先记住）：

      Agent       ：一个有人格、有工具、能发言的实体
      ModelClient ：Agent 背后的 LLM 调用方式（OpenAI/DeepSeek/...）
      Team        ：把多个 Agent 组织成一个对话场（轮次/选举/Swarm）
      Termination ：终止条件（说了什么/到了几轮/哪个 Agent 完工）
      Message     ：Agent 之间传递的消息（文本/工具调用/工具结果）

本文件按 Demo 顺序看完，你能：
  Demo 1  Hello AssistantAgent + DeepSeek model_info 的正确配法
  Demo 2  Tool 调用：把 Python 函数直接挂到 Agent 上（无需装饰器）
  Demo 3  两 Agent 对话：Assistant + UserProxy + RoundRobinGroupChat
  Demo 4  多角色 GroupChat：Planner/Coder/Critic 三角色协同
  Demo 5  SelectorGroupChat：让 LLM 决定下一个谁发言（vs RoundRobin）
  Demo 6  Termination 组合：MaxMessageTermination | TextMentionTermination
  Demo 7  CodeExecutorAgent：让一个 Agent 真的去跑 Python 代码
  Demo 8  Console UI 流式观察 + 把 Demo 4 的多 Agent 直播出来

运行前准备（agent 环境）：
    pip install -U "autogen-agentchat>=0.4" "autogen-ext[openai]>=0.4"
    # Demo 7 需要本地代码执行器，已包含在 autogen-agentchat 里

⚠️ AutoGen 0.4+ 全部 async，每个 demo 都用 asyncio.run() 包了一层。
   跑某一个 demo：
       python 05_autogen_tutorial.py 1
       python 05_autogen_tutorial.py
"""

import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# .env 在仓库根目录：/data/ghs/agent-learn/.env
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

TUTORIAL_DIR = Path(__file__).resolve().parent
WORK_DIR = TUTORIAL_DIR / "_autogen_workspace"
WORK_DIR.mkdir(exist_ok=True)


# ===========================================================
# 公共：模型客户端工厂（DeepSeek via OpenAI-compatible）
# ===========================================================
# AutoGen 不像 LlamaIndex 有全局 Settings —— 每个 Agent 都要显式传入
# `model_client`。所以工厂模式最方便：每个 demo 调一次 get_model_client()。
#
# **关键陷阱**：DeepSeek（以及任何 OpenAI 兼容但不是真 OpenAI 的）模型，
# 必须显式传 model_info，否则 AutoGen 不知道这个模型支持哪些能力
# （function_calling? vision? json_output?）会拒绝调用 tools。
# -----------------------------------------------------------
def get_model_client():
    """返回一个对接 DeepSeek 的 OpenAIChatCompletionClient。"""
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core.models import ModelFamily

    # 模型选型说明（实测得到）：
    # - DeepSeek v4 系列（pro / flash）都是 thinking/reasoning 模型，返回
    #   `reasoning_content` 字段，下一轮**必须**把它回传给 API，否则 400：
    #   "The reasoning_content in the thinking mode must be passed back."
    #   AutoGen 0.7.5 的 OpenAI client 序列化时丢掉了该字段，所以只要发生
    #   多轮调用（max_tool_iterations>1 或 reflect_on_tool_use=True），
    #   走真名 deepseek-v4-pro / deepseek-v4-flash 都会被拒。
    # - 用别名 `deepseek-chat` 调用时，服务端走"兼容层"（不要求 reasoning_content
    #   回传），AutoGen 能跑通；代价是 thinking 内容偶发以 <｜｜DSML｜｜...>
    #   形式漏到 content 里冒充工具调用，需要 system_message 强约束抑制。
    # 当前最务实路线：别名 + 强 prompt + max_tool_iterations。
    return OpenAIChatCompletionClient(
        model="deepseek-chat",
        base_url=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        # 必须显式声明能力，否则 AutoGen 走最保守路径
        model_info={
            "vision": False,
            "function_calling": True,    # DeepSeek 支持 function calling
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        temperature=0.2,
        timeout=60,
    )


# ===========================================================
# Demo 1 — Hello AssistantAgent
# ===========================================================
# 最小可用：一个 LLM Agent，给它一个 task 字符串，看它回什么。
# 学会三件事：
#   ① AssistantAgent 怎么构造（name + model_client + system_message）
#   ② await agent.run(task=...) 拿 TaskResult，里面是消息列表
#   ③ AutoGen 的消息类型（TextMessage / ToolCallRequestEvent / ...）
# -----------------------------------------------------------
async def demo_1_hello():
    from autogen_agentchat.agents import AssistantAgent

    client = get_model_client()
    agent = AssistantAgent(
        name="assistant",
        model_client=client,
        system_message="你是一位简洁的中文技术助手，回答不超过两句话。",
    )

    result = await agent.run(task="一句话说明：什么是 ReAct 范式？")

    print("=== Demo 1: Hello AssistantAgent ===")
    print(f"消息总数: {len(result.messages)}")
    for m in result.messages:
        # m.source 是 Agent 名字（包括 "user" 这个伪 Agent，代表 task）
        # m.content 是文本（也可能是 list，工具调用 / 多模态时）
        print(f"  [{m.source}] {m.content}")
    print()

    await client.close()


# ===========================================================
# Demo 2 — Tool 调用（最大反差点 vs LangChain）
# ===========================================================
# AutoGen 的 tool 注册方式特别简单：**直接传 Python 函数**给 Agent，
# 不需要 @tool 装饰器、不需要 BaseTool 子类、不需要 Pydantic schema。
# AutoGen 自己用类型注解 + docstring 生成 OpenAI function-calling schema。
#
# 注意：函数可以是普通同步函数 OR async 函数，AutoGen 会自动适配。
# -----------------------------------------------------------
def add(a: int, b: int) -> int:
    """计算 a + b 并返回整数和。"""
    return a + b


def multiply(a: int, b: int) -> int:
    """计算 a * b 并返回整数积。"""
    return a * b


async def demo_2_tools():
    from autogen_agentchat.agents import AssistantAgent

    client = get_model_client()
    agent = AssistantAgent(
        name="calculator",
        model_client=client,
        tools=[add, multiply],     # ← 直接传函数，简到离谱
        system_message=(
            "你是计算助手，必须用工具完成所有算术，不许心算。"
            "调用工具时**必须**通过 OpenAI function calling 协议返回 tool_calls 字段，"
            "**禁止**以任何文本形式（包括 <｜｜DSML｜｜...> 之类的标签）输出工具调用——"
            "那种格式不会被识别，会导致工具不被执行。"
            "拿到结果后用一句话总结。"
        ),
        # AutoGen 默认 max_tool_iterations=1：一轮 tool 调用就停。
        # 调高让 agent 在单 turn 内循环 tool 调用（add → multiply → 总结）。
        # 注意：不要用 reflect_on_tool_use=True —— 它对 reasoning 模型
        # （deepseek-v4-pro / deepseek-reasoner）会触发"工具结果反思"流程，
        # 把上一条 assistant 消息回传给 API，但 AutoGen 序列化时丢掉了
        # `reasoning_content`，DeepSeek 直接 400。
        max_tool_iterations=5,
    )

    result = await agent.run(task="先算 13 加 29，再把结果乘以 7。")

    print("=== Demo 2: Tool 调用 ===")
    for m in result.messages:
        # 工具相关的消息有 ToolCallRequestEvent / ToolCallExecutionEvent / ToolCallSummaryMessage
        msg_type = type(m).__name__
        if isinstance(m.content, list):
            # 工具调用 / 工具结果：content 是 FunctionCall / FunctionExecutionResult 列表
            for c in m.content:
                print(f"  [{m.source}/{msg_type}] {c}")
        else:
            print(f"  [{m.source}/{msg_type}] {m.content}")
    print()

    await client.close()


# ===========================================================
# Demo 3 — 两 Agent 对话：Assistant + UserProxy + RoundRobinGroupChat
# ===========================================================
# 这是 AutoGen 的"经典 Hello World"：
#
#   AssistantAgent  ←──→  UserProxyAgent
#       (LLM)              (代表人；可以是真人 input、也可以自动应答)
#
# 学会三件事：
#   ① UserProxyAgent 是什么、为什么要它（不是真人占位，是"接受/拒绝"角色）
#   ② RoundRobinGroupChat 让两个 Agent 轮流说话（A→B→A→B…）
#   ③ TextMentionTermination("APPROVE") —— 看到约定关键词就停
# -----------------------------------------------------------
async def demo_3_two_agents():
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import (
        TextMentionTermination, MaxMessageTermination,
    )

    client = get_model_client()

    writer = AssistantAgent(
        name="writer",
        model_client=client,
        system_message=(
            "你是文案撰写助手。给出一段广告文案后，等待 reviewer 反馈；"
            "如果 reviewer 说 APPROVE 就停止；否则改写并再次提交。"
        ),
    )

    # 自动版 UserProxy：不需要真人 input，按规则自动回复
    async def auto_reviewer_input(prompt: str, _cancel=None) -> str:
        # 约定：第一次让它修改，第二次直接 APPROVE，避免无限循环
        if "再来一版" not in prompt and "APPROVE" not in prompt:
            return "请把语气改得更活泼一点，再来一版。"
        return "APPROVE"

    reviewer = UserProxyAgent(
        name="reviewer",
        input_func=auto_reviewer_input,
    )

    # 终止：看到 "APPROVE" 就停 OR 最多 6 条消息（保险丝）
    termination = TextMentionTermination("APPROVE") | MaxMessageTermination(6)

    team = RoundRobinGroupChat(
        [writer, reviewer],
        termination_condition=termination,
    )

    result = await team.run(task="写一段 30 字的耳机广告文案。")

    print("=== Demo 3: 两 Agent 对话 (Assistant + UserProxy) ===")
    for m in result.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        print(f"  [{m.source}] {content[:120]}")
    print(f"  ↳ 终止原因: {result.stop_reason}")
    print()

    await client.close()


# ===========================================================
# Demo 4 — 多角色 GroupChat：Planner / Coder / Critic
# ===========================================================
# 真实多 Agent 协同：每个 Agent 有清晰角色，按固定顺序轮流发言。
# 这是 02_集成LlamaIndex_AutoGen_MCP方案.md 里 AutoGen 路径 B 的迷你版本。
#
#   Planner → 拆任务、列方案
#   Coder   → 给出实现思路（不真跑代码）
#   Critic  → 唱反调，挑漏洞，决定是否 DONE
#
# 关键点：
#   ① 用 system_message 把每个 Agent 的"人设"写死
#   ② 用 RoundRobinGroupChat 让三人按 [Planner, Coder, Critic, Planner...]
#      的顺序循环
#   ③ 用 TextMentionTermination("DONE") 让 Critic 觉得满意时主动收尾
# -----------------------------------------------------------
async def demo_4_groupchat():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import (
        TextMentionTermination, MaxMessageTermination,
    )

    client = get_model_client()

    planner = AssistantAgent(
        name="planner",
        model_client=client,
        system_message=(
            "你是 Planner。把用户问题拆成 2-3 个子任务，"
            "明确给出 'Step 1/2/3：xxx' 的结构。不要写代码。"
        ),
    )

    coder = AssistantAgent(
        name="coder",
        model_client=client,
        system_message=(
            "你是 Coder。基于 Planner 的步骤，给出**伪代码 / 简化实现思路**。"
            "保持简短，不超过 30 行；不要执行代码。"
        ),
    )

    critic = AssistantAgent(
        name="critic",
        model_client=client,
        system_message=(
            "你是 Critic。检查 Planner 的步骤是否覆盖完整、Coder 的实现是否完整、是否有漏洞。"
            "若一切 OK，回复 'DONE'（必须大写的这个词，不要带其它前后缀）；"
            "若有问题，写一句话指出来，让前两位再迭代一轮。"
        ),
    )

    termination = TextMentionTermination("DONE") | MaxMessageTermination(20)
    team = RoundRobinGroupChat(
        [planner, coder, critic],
        termination_condition=termination,
    )

    task = "实现一个 LRU 缓存（容量 3，支持 get / put）。"
    result = await team.run(task=task)

    print("=== Demo 4: 多角色 GroupChat (Planner/Coder/Critic) ===")
    for m in result.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        print(f"\n--- [{m.source}] ---\n{content[:10000]}")
    print(f"\n↳ 终止原因: {result.stop_reason}")
    print(f"↳ 总消息数: {len(result.messages)}")
    print()

    await client.close()


# ===========================================================
# Demo 5 — SelectorGroupChat：让 LLM 决定下一个谁发言
# ===========================================================
# RoundRobin 的局限：固定顺序，不管当前局面下谁最该说。
# SelectorGroupChat 让一个 selector LLM 看完已有对话，决定下一个轮到谁。
#
# 经典使用场景：
#   · 路由型任务：用户问"代码题"→应该让 coder 接手；问"史料题"→让 historian
#   · 多专家会诊：每个 Agent 是不同领域专家，按问题动态选
#
# 这里举个简化的"双专家"例子：math_expert + lit_expert，
# 让 selector 自动按问题路由。
# -----------------------------------------------------------
async def demo_5_selector():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import SelectorGroupChat
    from autogen_agentchat.conditions import (
        TextMentionTermination, MaxMessageTermination,
    )

    client = get_model_client()

    math_expert = AssistantAgent(
        name="math_expert",
        model_client=client,
        description="数学专家：擅长解算、证明、概率题",   # 给 selector 看的
        system_message="你是数学专家。简洁解答数学问题；非数学问题请说：我不擅长。",
    )

    lit_expert = AssistantAgent(
        name="lit_expert",
        model_client=client,
        description="文学专家：擅长古诗词、现代文学、修辞分析",
        system_message="你是文学专家。简洁解答文学问题；非文学问题请说：我不擅长。",
    )

    summarizer = AssistantAgent(
        name="summarizer",
        model_client=client,
        description="汇总者：在所有专家答完后给最终结论，结尾说 'TERMINATE'",
        system_message=(
            "你是汇总者。基于前面专家的回答给出最终一句话总结。"
            "结尾必须独立一行写 'TERMINATE'。"
        ),
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(8)

    team = SelectorGroupChat(
        [math_expert, lit_expert, summarizer],
        model_client=client,        # selector 也需要一个 LLM 来做选人决策
        termination_condition=termination,
        # selector_prompt 可自定义；这里用默认
        allow_repeated_speaker=False,   # 不让同一个 Agent 连续发言
    )

    task = "请回答：(1) 圆周率前 3 位是多少？(2) 床前明月光，的下一句是？"
    result = await team.run(task=task)

    print("=== Demo 5: SelectorGroupChat (动态选人) ===")
    for m in result.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        print(f"\n--- [{m.source}] ---\n{content[:10000]}")
    print(f"\n↳ 终止原因: {result.stop_reason}")
    print()

    await client.close()


# ===========================================================
# Demo 6 — Termination 条件组合
# ===========================================================
# 终止条件是 AutoGen 的"保险丝"，至关重要。常用条件：
#
#   · MaxMessageTermination(n)       —— 最多 n 条消息（必备保险丝）
#   · TextMentionTermination(s)      —— 任意 Agent 说了 s 就停
#   · TokenUsageTermination(n)       —— 累计 token 超过 n 就停（费用控制）
#   · TimeoutTermination(seconds)    —— 总耗时超限就停
#   · SourceMatchTermination(name)   —— 指定 Agent 一发言就停
#   · ExternalTermination()          —— 程序外部 .set() 强制停
#
# 组合：
#   `|` 表示"或"（任一满足就停）
#   `&` 表示"与"（全部满足才停，少用）
#
# 推荐组合：业务关键词 | 最大轮数（业务先停就别浪费 token；保险丝兜底）
# -----------------------------------------------------------
async def demo_6_termination():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import (
        TextMentionTermination, MaxMessageTermination, SourceMatchTermination,
    )

    client = get_model_client()

    a = AssistantAgent("a", model_client=client,
                       system_message="你叫 A，回答时说'我是 A：' 然后正经回答。")
    b = AssistantAgent("b", model_client=client,
                       system_message="你叫 B，回答时说'我是 B：' 然后正经回答。")
    c = AssistantAgent("c", model_client=client,
                       system_message=(
                           "你叫 C。你只在前两人都说完后才发言；"
                           "你的发言必须以 'FINAL:' 开头，给出最终结论。"
                       ))

    # 三种终止：FINAL 关键词 / C 一开口就停 / 最大 6 条兜底
    termination = (
        TextMentionTermination("FINAL:")
        | SourceMatchTermination("c")     # c 一发言就停（与上面其实重复，演示组合）
        | MaxMessageTermination(6)
    )

    team = RoundRobinGroupChat([a, b, c], termination_condition=termination)
    result = await team.run(task="一句话讨论：吃饭和睡觉哪个更重要？")

    print("=== Demo 6: Termination 组合 ===")
    for m in result.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        print(f"  [{m.source}] {content[:120]}")
    print(f"\n↳ 终止原因: {result.stop_reason}")
    print()

    await client.close()


# ===========================================================
# Demo 7 — CodeExecutorAgent：让一个 Agent 真的去跑 Python 代码
# ===========================================================
# 这是 AutoGen 的杀手锏：CodeExecutorAgent 不调 LLM，它**只负责执行**前一个
# Agent（通常是 Coder）抛出的代码块，然后把 stdout 返回到对话流。
#
# 工作流：
#   coder    → 写一段 ```python ...``` 代码块
#   executor → 真的跑（本地或 Docker），把结果发回对话
#   coder    → 看结果决定是否继续 / 修 bug
#
# 两种执行器：
#   · LocalCommandLineCodeExecutor —— 直接在当前进程的子进程跑，**不安全**，
#                                      仅 demo 和受信任输入用
#   · DockerCommandLineCodeExecutor —— 在 Docker 容器里跑，需要 docker daemon
#
# 本 demo 用 Local（教程用，不要在生产跑用户输入的代码！）。
# -----------------------------------------------------------
async def demo_7_code_executor():
    try:
        from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_agentchat.conditions import (
            TextMentionTermination, MaxMessageTermination,
        )
        from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
    except ImportError as e:
        print("=== Demo 7: CodeExecutorAgent ===")
        print(f"(skip: 缺少依赖 {e})")
        print()
        return

    client = get_model_client()

    coder = AssistantAgent(
        name="coder",
        model_client=client,
        system_message=(
            "你是 Coder。给出可执行的 Python 代码块（用 ```python 包起来），"
            "等 executor 跑完后查看输出。"
            "拿到正确结果后总结一句并以 'DONE' 结尾。"
        ),
    )

    # 把代码执行的工作目录隔在 _autogen_workspace/code_exec 下
    code_dir = WORK_DIR / "code_exec"
    code_dir.mkdir(exist_ok=True)
    executor = LocalCommandLineCodeExecutor(work_dir=str(code_dir), timeout=15)
    code_runner = CodeExecutorAgent(name="executor", code_executor=executor)

    termination = TextMentionTermination("DONE") | MaxMessageTermination(8)
    team = RoundRobinGroupChat(
        [coder, code_runner],
        termination_condition=termination,
    )

    task = "用 Python 算斐波那契数列前 10 项并打印出来。"
    result = await team.run(task=task)

    print("=== Demo 7: CodeExecutorAgent ===")
    for m in result.messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        print(f"\n--- [{m.source}] ---\n{content[:300]}")
    print(f"\n↳ 终止原因: {result.stop_reason}")
    print()

    await client.close()


# ===========================================================
# Demo 8 — Console 流式 UI：直播多 Agent 对话
# ===========================================================
# 前面的 demo 都用 await team.run(...) 一次拿完整结果，看不到中间过程。
# `Console(team.run_stream(task=...))` 是 AutoGen 自带的流式打印器，
# 跑的时候 Agent 发言会**逐 token 实时**打到 stdout，调试体验非常爽。
#
# Console 还会自动渲染：
#   · 谁在说话（Agent 名字 + 颜色）
#   · 工具调用 (FunctionCall) 与工具结果 (FunctionExecutionResult)
#   · 终止原因
#   · 总耗时 + 总 token 消耗
#
# 这正是 02_集成LlamaIndex_AutoGen_MCP方案.md 里"录 demo GIF"的用法。
# -----------------------------------------------------------
async def demo_8_console_stream():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import (
        TextMentionTermination, MaxMessageTermination,
    )
    from autogen_agentchat.ui import Console

    client = get_model_client()

    optimist = AssistantAgent(
        name="optimist",
        model_client=client,
        system_message="你是乐观派。200 字以内说出对一个话题的积极看法。",
        model_client_stream=True,  # 这个参数一定要加上，控制agent 开启流式打印
    )
    pessimist = AssistantAgent(
        name="pessimist",
        model_client=client,
        system_message="你是悲观派。200 字以内说出对一个话题的消极看法。",
        model_client_stream=True,
    )
    judge = AssistantAgent(
        name="judge",
        model_client=client,
        system_message=(
            "你是裁判。听完两边发言后，给出一段话平衡评论；"
            "结尾独立一行写 'TERMINATE'。"
        ),
        model_client_stream=True,
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(6)
    team = RoundRobinGroupChat(
        [optimist, pessimist, judge],
        termination_condition=termination,
    )

    print("=== Demo 8: Console 流式 UI ===")
    print("(下面是直播；正常情况下你会看到 token 一个个蹦出来)\n")

    # ⭐ 关键就这一行：Console(...) 替代 await team.run(...)
    await Console(team.run_stream(task="讨论：远程办公 vs 现场办公。"))
    print()

    await client.close()


# ===========================================================
# 把 Demo 对应到 CodeLens 改造路径（02_集成方案文档）
# ===========================================================
# 那份方案里 AutoGen 列了 4 条路径：
#
#   路径 A: 5-Agent GroupChat 深度模式 ⇐ 看懂 Demo 4 + Demo 5
#   路径 B: Code Review Crew (3-Agent) ⇐ 看懂 Demo 4
#   路径 C: Bug Hunt with code executor ⇐ 看懂 Demo 7
#   路径 D: 经典两 Agent (UserProxy)   ⇐ 看懂 Demo 3
#
# 推荐起步顺序：
#   Demo 1, 2 → 跑通环境（半天）
#   Demo 3    → 理解 UserProxy 和终止条件（半天，对应路径 D）
#   Demo 4, 6 → 多角色 + 终止组合（一天，对应路径 B）
#   Demo 5    → SelectorGroupChat（一天，对应路径 A 升级版）
#   Demo 7    → CodeExecutor（一天，对应路径 C，可选）
#   Demo 8    → 流式 UI（半天；做演示视频时用）
#
# 看完这 8 个 demo，再去写 codelens/app/teams/ 下的多 Agent 模块，
# 几乎每一行都能对上号。
# -----------------------------------------------------------

DEMOS = {
    # "1": demo_1_hello,
    # "2": demo_2_tools,
    # "3": demo_3_two_agents,
    # "4": demo_4_groupchat,
    # "5": demo_5_selector,
    # "6": demo_6_termination,
    # "7": demo_7_code_executor,
    "8": demo_8_console_stream,
}


async def _run_all():
    for key in sorted(DEMOS):
        await DEMOS[key]()


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which is None:
        asyncio.run(_run_all())
    else:
        asyncio.run(DEMOS[which]())
