# agent-learn

LLM Agent 学习与项目仓库。包含一个完整可演示的中型项目（**CodeLens**）和按官方教程整理的 LangChain / LangGraph / LlamaIndex / AutoGen 笔记。

## 仓库结构

```
agent-learn/
├── codelens/          # ⭐ 主项目：基于 LangGraph + AutoGen 的 Agentic RAG 助手
├── tutorial/          # LangChain / LangGraph / LlamaIndex / AutoGen 学习笔记 + 可跑 demo
└── .env.example       # 环境变量示例；真实 .env 不入库
```

---

# CodeLens — Agentic RAG 助手

给开发者的"代码 + 文档"智能问答助手：喂一个 Markdown 文档目录 + 一个 C++ 代码库（默认 `cpp-httplib`，约 1.2 万行 header-only HTTP 库），Agent 自主调工具检索、推理、反思，给出**带文件:行号引用**的回答。

## 一次完整问答演示

```
用户：对比 ThreadPool 在文档建议和源码实现上的差异。

[mode_router] LLM 决策：Deep（命中"对比"语义）

[GroupChat 启动 — RoundRobin: Retriever → Analyst → Critic → Reporter]

  Retriever  → tool: search_multi_aspect("ThreadPool 文档建议 vs 源码实现")
              ← 命中 docs/threading.md（建议固定大小）+ httplib.h:L1475 (实际实现)

  Analyst    → 引用 [doc] docs/threading.md:L23：建议线程数固定为 CPU 核数
              引用 [code] httplib.h:L1486：用 std::thread::hardware_concurrency() 动态取
              结论：文档建议 = 实际实现的核心策略，但源码额外加了 hardware_concurrency 兜底

  Critic     → "证据不足：只看了构造路径，没看 shutdown / 异常路径。需要补查：
              ThreadPool 析构 + 任务异常处理"

  Retriever  → tool: search_code_only("ThreadPool ~Shutdown ~exception")
              ← 命中 httplib.h:L1521 detail::shutdown 部分

  Analyst    → 补充：shutdown 时 join 所有 worker，未捕获异常会终止进程
  Critic     → "证据充分，可以汇总"

  Reporter   → ## 文档 vs 源码差异
              1. 线程数策略一致 [docs/threading.md:L23][httplib.h:L1486]
              2. 文档未提到异常处理；源码也未捕获 [httplib.h:L1521]
              ...
              DONE  ← 终止
```

简单题（"ThreadPool 怎么实现？"）会被路由到 **Quick** 模式：单 LangGraph ReAct Agent 直接回。

## 三档运行模式

| 模式 | 引擎 | 适用场景 | 代价 |
| --- | --- | --- | --- |
| **Quick** | LangGraph 单 Agent ReAct + reflect | 单一来源、概念解释、API 用法、定义类问题 | 1 个 LLM 多轮（≤50 iter） |
| **Deep** | AutoGen 4-Agent GroupChat（Retriever / Analyst / Critic / Reporter） | 跨域对比、多目标连问、文档 vs 源码一致性、需多视角质疑 | 4 个角色 × 多轮（≤12 msg） |
| **Auto** | LLM 路由器 + 关键词 fallback | 不确定走哪个就用它 | Quick / Deep + 一次轻量决策（≤10 token） |

UI 默认 Quick，侧边栏可手动切到 Deep / Auto。

## 架构总览

```
                    ┌─────────────────────────────────────────────────┐
                    │  入口：scripts/run_*.py  /  ui/app.py (Streamlit) │
                    └─────────────────┬───────────────────────────────┘
                                      │
                            ┌─────────┴─────────┐
                            │  mode_router      │  (Auto 模式专用)
                            │  LLM + 关键词     │
                            └────┬─────────┬────┘
                          Quick  │         │  Deep
                                 ▼         ▼
                ┌────────────────────┐   ┌───────────────────────────────┐
                │   LangGraph        │   │   AutoGen GroupChat           │
                │   StateGraph       │   │   RoundRobin / Selector       │
                │                    │   │                               │
                │  ┌─ agent ─┐       │   │   Retriever ─ tool_calls      │
                │  │   ↑     │       │   │      ↓                        │
                │  │ tools ←─┤       │   │   Analyst                     │
                │  │   reflect       │   │      ↓                        │
                │  └─────────┘       │   │   Critic ── 需要补查 ──┐      │
                │  iter ≤ 50         │   │      ↓                 │      │
                │                    │   │   Reporter ─ DONE      │      │
                └────────┬───────────┘   │  msg ≤ 12              │      │
                         │               └───────────┬────────────┘      │
                         │                           │                   │
                         │  Quick 6 个工具             │  Deep 6 个工具      │
                         ▼                           ▼                   │
        ┌─────────────────────────────────────────────────────────────┐ │
        │  工具层（LangGraph 直接绑定 / AutoGen 通过 Retriever 持有）│ │
        │  ─────────────────────────────────────────────────────────  │ │
        │  search_docs(LlamaIndex Router) │ grep_code(rg)            │ │
        │  read_file(行号 + 400 行截断)    │ list_files │ web_search   │ │
        │  calculator                                                │ │
        └─────────────────────┬───────────────────────────────────────┘ │
                              │                                          │
                              ▼                                          │
        ┌─────────────────────────────────────────────────────────────┐ │
        │  LlamaIndex 检索层（router.py）                              │◀┘
        │  ─────────────────────────────────────────────────────────  │
        │  RouterQueryEngine (LLMSingleSelector + RobustParser)       │
        │     ├── doc_only       → docs_qe                            │
        │     ├── code_only      → code_qe                            │
        │     └── multi_aspect   → SubQuestionQueryEngine             │
        │                            ↓ 拆子问题，分配给 docs_qe/code_qe│
        └─────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  Milvus Lite（本地文件 storage/milvus.db）                   │
        │     ├── codelens_docs   collection (Markdown chunks)        │
        │     └── codelens_code   collection (C++ chunks)             │
        │  Embedding: gte-Qwen2-1.5B (HuggingFace 本地，GPU/CPU 自动)   │
        └─────────────────────────────────────────────────────────────┘
```

## 技术栈

| 层 | 选型 | 备注 |
| --- | --- | --- |
| LLM | **DeepSeek-Chat**（OpenAI 兼容接口） | `model="deepseek-chat"` 别名，不要用真名 v4-pro/flash |
| Embedding | **gte-Qwen2-1.5B-instruct**（HuggingFace） | 本地跑，省 API 额度；`trust_remote_code=True` |
| 向量库 | **Milvus Lite**（单文件 sqlite-like） | 双 collection：`codelens_docs` + `codelens_code` |
| 检索层 | **LlamaIndex** RouterQueryEngine + SubQuestionQueryEngine | 顶层 LLMSingleSelector 三选一 |
| 编排 — Quick | **LangGraph** 1.1+ StateGraph（agent / tools / reflect） | 单 Agent ReAct + 反思 + 硬上限 50 iter |
| 编排 — Deep | **AutoGen** 0.4+ RoundRobinGroupChat / SelectorGroupChat | 4 角色 + DONE/MaxMessage 双终止 |
| Mode 路由 | 自研 `mode_router.py`：LLM + 关键词 | Quick/Deep 二选一，决策开销 ≤10 token |
| 记忆 | LangGraph **SqliteSaver** + `trim_messages` | 多轮 + token 窗口裁剪 |
| UI | **Streamlit** 1.x | 模式切换 / 多 thread / 流式 token |

## 目录结构

```
codelens/
├── app/
│   ├── llm.py                    # ChatOpenAI(DeepSeek) 单例
│   ├── embeddings.py             # gte-Qwen2 本地 embedding，模块级单例
│   ├── prompts.py                # SYSTEM_PROMPT（限定查询 data/，禁查 app/）
│   ├── memory.py                 # trim_messages 滑窗裁剪
│   ├── retriever.py              # 老 LangChain 检索器（保留兼容）
│   ├── vectorstore.py            # 老 LangChain-Milvus 入口
│   ├── ingest/                   # 老 LangChain ingest（splitter）
│   │
│   ├── retrieval/                # ⭐ LlamaIndex 检索层
│   │   ├── settings.py           #   全局 Settings（注入 DeepSeek + gte-Qwen2）
│   │   ├── vector_stores.py      #   双 Milvus collection 句柄
│   │   ├── ingest.py             #   双 collection drop+rebuild
│   │   └── router.py             #   Router + SubQ + RobustParser
│   │
│   ├── tools/                    # LangGraph ReAct 工具
│   │   ├── search_docs.py        #   语义检索（走 LlamaIndex Router）
│   │   ├── grep_code.py          #   ripgrep / grep 字面匹配
│   │   ├── read_file.py          #   带行号 + 400 行硬截断
│   │   ├── list_files.py         #   列目录
│   │   ├── web_search.py         #   DDGS / DuckDuckGo Web 检索（免 key）
│   │   └── calculator.py         #   安全数学表达式计算
│   │
│   ├── graph/                    # ⭐ LangGraph
│   │   ├── state.py              #   CodeLensState (messages/iterations/reflection)
│   │   ├── nodes.py              #   agent_node / tools_node / reflect_node
│   │   └── build.py              #   StateGraph + AGENT_MAX_ITERATIONS=50
│   │
│   └── teams/                    # ⭐ AutoGen GroupChat
│       ├── retrieval_tools.py    #   AutoGen 工具封装（LlamaIndex + Web + Calculator）
│       ├── agents.py             #   Retriever/Analyst/Critic/Reporter 4 角色
│       ├── groupchat.py          #   build_team(use_selector=False)
│       ├── runner.py             #   run_groupchat / stream_groupchat
│       └── mode_router.py        #   Auto 模式 LLM 路由
│
├── scripts/
│   ├── build_index.py            # 老路径（LangChain Chroma / Milvus 单 collection）
│   ├── build_index_v2.py         # ⭐ 当前：LlamaIndex 双 collection drop+rebuild
│   ├── run_cli.py                # Quick：LangGraph 单 Agent
│   ├── run_cli_memory.py         # Quick + 多轮 + checkpoint
│   ├── run_groupchat.py          # Deep：AutoGen 4-Agent 流式打印
│   ├── run_auto.py               # ⭐ Auto：mode_router 自动分发
│   └── run_tools.py              # 工具调试用
│
├── ui/
│   ├── app.py                    # Streamlit 入口
│   ├── sidebar.py                # Mode toggle + Sessions（多 thread）
│   ├── chat.py                   # replay_history / run_turn_streaming
│   ├── render.py                 # 工具调用 / token 流渲染
│   └── runtime.py                # graph + saver 单例
│
├── data/
│   ├── code/cpp-httplib/         # 默认目标代码库（git clone 拉下来）
│   └── docs/docs-src/            # 配套 Markdown 文档
│
├── storage/
│   ├── milvus.db                 # Milvus Lite 数据
│   └── checkpoints.db            # LangGraph SqliteSaver
│
├── tests/                        # 烟测脚本
└── requirements.txt
```

`data/` + `storage/` + `__pycache__/` 已在 `.gitignore` 中。

---

## Quick Start

### 1. 环境

```bash
conda create -n agent python=3.11 -y
conda activate agent
cd codelens
pip install -r requirements.txt
```

> `requirements.txt` 把 LangChain 1.2+ / LangGraph 1.1+ / LlamaIndex 0.14+ / AutoGen 0.4+ / DDGS Web 检索依赖装齐。注意 LlamaIndex 必须用 0.14+，否则跟 `langchain-openai>=1`（强制 `openai>=2.26`）的依赖会对撞——文件里写了详细注释。

### 2. API key

仓库根目录 `.env`（可从 `.env.example` 复制）：

```dotenv
OPENAI_API_KEY="sk-xxxxxxxx"
OPENAI_API_BASE="https://api.deepseek.com/v1"
```

DeepSeek 用 OpenAI 兼容接口；模型名固定 `deepseek-chat`（别名），**不要用 `deepseek-v4-pro` 真名**——真名是 thinking 模型，AutoGen 序列化时丢 `reasoning_content` 字段会 400。

安全提醒：`.env` / `.env.*` 不应入库。如果密钥曾经被提交或贴到日志里，请在 DeepSeek 控制台轮换旧 key，再把新 key 只放在本地 `.env`。

### 3. 准备数据

```bash
# 默认目标代码库：cpp-httplib（首次需要 clone）
git clone --depth 1 https://github.com/yhirose/cpp-httplib data/code/cpp-httplib
# data/docs/ 放任意 Markdown 技术文档（已自带 cpp-httplib 的 docs-src/）
```

想换目标代码库就直接换 `data/code/` 下的内容，重新跑索引即可。

### 4. 建索引

```bash
# 双 collection（codelens_docs + codelens_code），会 drop 重建
python scripts/build_index_v2.py
```

首次跑会下载 gte-Qwen2-1.5B 权重（约 3GB），后续走本地缓存。GPU 上几分钟，纯 CPU 大约 20–30 分钟。

### 5. 跑

```bash
# CLI · Auto 模式（最常用）
python scripts/run_auto.py "对比 ThreadPool 文档建议和源码实现"

# CLI · 强制模式
python scripts/run_auto.py --force quick "ThreadPool 怎么实现？"
python scripts/run_auto.py --force quick "搜索一下 cpp-httplib 最新 release 信息"
python scripts/run_auto.py --force quick "计算 128 * 0.75 + sqrt(16)"
python scripts/run_auto.py --force deep "..."
python scripts/run_groupchat.py --selector "..."     # SelectorGroupChat 替代 RoundRobin

# CLI · 多轮带记忆
python scripts/run_cli_memory.py --thread demo "ThreadPool 实现"
python scripts/run_cli_memory.py --thread demo "那它的锁粒度呢？"   # 同 thread 接续

# Streamlit
streamlit run ui/app.py
```

---

## 三档模式详解

### Quick — LangGraph 单 Agent ReAct

`app/graph/build.py` 定义的状态图：

```
        tool_calls
   ┌──────────────────┐
   │                  │
   ▼                  │
 tools ──────────► agent ◀────────────── 需要补查
                    │
        no tool_calls │
       ┌──────────────┴────────────┐
       │ iter < 3                  │ iter ≥ 3
       ▼                           ▼
    reflect ───── 可以结束 ──────► END
       │
       └── 需要继续检索 ──────► agent
```

**关键设计**：

- **`AGENT_MAX_ITERATIONS=50` 硬上限**：阶段 8.1 改造后 `search_docs` 返回 LlamaIndex 合成答案 + 引用文件列表，模型容易把多个引用 source 误解为"还有别的可查"，陷入反复 search 死循环；硬上限作为兜底，避免 LangGraph recursion_limit 抛错。
- **reflect 节点只在 iter < 3 时触发**：让同一个 LLM 以审稿人身份检查，输出 `需要继续检索：<关键词>` 或 `可以结束`，避免"看起来答完但证据不足"的情况。

入口：`scripts/run_cli.py`、`scripts/run_cli_memory.py`。

### Deep — AutoGen 4-Agent GroupChat

`app/teams/agents.py` 4 个角色：

| Agent | 工具 | 职责 |
| --- | --- | --- |
| **Retriever** | 6 个工具：4 个 LlamaIndex 检索工具 + `web_search` / `calculator` | 唯一持有工具能力；按问题类型挑工具；拿到结果后**一句话**说查到了什么 |
| **Analyst** | — | 读 Retriever 拿到的片段，先声明 `[doc]` / `[code]`，结构化解读，**禁止凭印象编造** |
| **Critic** | — | 唱反调：引用是否支撑结论？有没有漏掉的角度（错误处理路径 / 边界条件）？输出 `需要补查：<具体问题>` 或 `证据充分，可以汇总` |
| **Reporter** | — | 综合所有发言写最终回答，关键论断标 `[文件名]`，**单独一行 DONE** 触发终止 |

**协调与终止**：

- 默认 **RoundRobinGroupChat**（实现简单、行为可预测、调试容易）；带 `--selector` 走 **SelectorGroupChat**（LLM 看局面挑下一个发言者）。
- `TextMentionTermination("DONE") | MaxMessageTermination(12)` 双终止：正常 1 轮 4 条消息，最多 3 轮兜底。

入口：`scripts/run_groupchat.py`。

### Auto — LLM 路由器 + 关键词 fallback

`app/teams/mode_router.py`：

1. 一次轻量 DeepSeek 调用（max_tokens=10），prompt 强约束输出 `Quick` / `Deep`
2. LLM 解析失败 → 关键词 fallback（`对比 / 差异 / vs / compare / ...`）
3. 仍不命中 → 默认 `Quick`（更便宜，对简单题影响最小）

入口：`scripts/run_auto.py`；UI 侧边栏也可切到 Auto。

---

## 工具层（6 个）

| 工具 | 实现 | 设计要点 |
| --- | --- | --- |
| `search_docs` | LlamaIndex Router（语义检索） | 顶层 LLMSingleSelector 三选一 doc_only / code_only / multi_aspect |
| `grep_code` | `rg`（无则降级 `grep`） | 找精确符号名（类名、函数名、宏） |
| `read_file` | 标准库 + 行号 + **400 行硬截断** | "grep 定位 → read_file 查看" 双步闭环；防 `httplib.h` 1.2 万行一次塞爆 context |
| `list_files` | `pathlib.rglob` | 不知道有哪些文件时探目录 |
| `web_search` | DDGS / DuckDuckGo（免 key） | 查最新 / 网上 / 外部资料，返回标题、摘要、URL |
| `calculator` | AST 白名单表达式求值 | 精确计算数字、四则/幂运算和常用 `math` 函数，禁止执行任意 Python |

`read_file` 的设计动机来自实测：只有 `grep_code` 时模型陷入 30+ 次"换 pattern 重 grep"死循环；加上 `read_file(path, start, end)` 后循环立即收敛。

---

## 检索层（LlamaIndex Router + SubQuestion）

`app/retrieval/router.py` 暴露 4 个 engine：

```python
get_docs_query_engine()         # codelens_docs 单查（top_k=5）
get_code_query_engine()         # codelens_code 单查（top_k=5）
get_subquestion_query_engine()  # 拆子问题 → 分配给 docs/code → 合成
get_router()                    # 顶层 LLMSingleSelector 三选一
```

**双 collection 而不是单 collection + metadata 过滤**：查文档时不希望召回里混进代码片段干扰评分，分开查比 metadata 过滤更稳；embedding 同一个，存储成本忽略。

**SubQuestion 不需要第四个 collection**：它是"在 docs/code 之上的元能力"——把复合问题拆成子问题，每个子问题去 `docs_qe` / `code_qe` 查回答再合成，不是新数据源。

---

## AutoGen + DeepSeek 兼容性踩坑

每条都是实测，配置严格按这个走（详见 `app/teams/agents.py` docstring）：

1. **必须用 `model="deepseek-chat"` 别名**，不是 `deepseek-v4-pro` 真名。真名是 thinking 模型，要求下一轮回传 `reasoning_content`，AutoGen 0.7.5 序列化时丢这个字段 → 必 400。
2. **DSML 标签泄漏**：DeepSeek 兼容层偶发把 thinking 内容以 `<｜｜DSML｜｜tool_calls>` 形式漏到 content 冒充 tool 调用 → 工具不被识别。每个 system_message 加 `_PROTOCOL_GUARD` 显式禁止该格式。
3. **`max_tool_iterations` 默认 1**：一次工具就停。多步必须显式调高（Retriever 给 5）。
4. **不要 `reflect_on_tool_use=True`**：触发把上一条 assistant 消息回传 API 的反思流程，对 reasoning 模型必 400。
5. **必须显式传 `model_info`**：否则 AutoGen 走最保守路径直接拒调 tools。

---

## Chunking 策略

| 类型 | Splitter | chunk_size | overlap |
| --- | --- | --- | --- |
| Markdown | `MarkdownHeaderTextSplitter`（h1/h2/h3）→ `RecursiveCharacterTextSplitter` | 800 | 80 |
| C++ | `RecursiveCharacterTextSplitter.from_language(Language.CPP)` | 1200 | 120 |

C++ 用 Language-aware 切分（按花括号 / 函数边界），避免按字符切把函数拦腰切开造成语义丢失。

---

## 多轮记忆

```python
# scripts/run_cli_memory.py 的核心
saver = SqliteSaver.from_conn_string("storage/checkpoints.db")
graph = build_graph(checkpointer=saver)

config = {"configurable": {"thread_id": "demo"}}
graph.invoke({"messages": [...]}, config=config)   # 第一轮
graph.invoke({"messages": [...]}, config=config)   # 第二轮，自动接续
```

`app/memory.py` 提供 `trim()` helper，后续可接到 agent 节点入口做 token 窗口裁剪。当前多轮接续主要依赖 LangGraph checkpoint；Streamlit 侧边栏的 "Sessions" 列出所有 thread_id，可点击切换历史会话。

---

## Streamlit UI 特性

```bash
streamlit run ui/app.py
```

- **侧边栏 Mode toggle**：Quick / Deep / Auto 一键切，带 hover 解释文案
- **多会话管理**：New / Clear 按钮 + 历史 thread 列表，点击切换
- **流式输出**：LangGraph token 实时打印；Deep 模式按 Agent 角色分块渲染
- **静态信息**：当前模型、向量库、索引 chunk 数

---


## 入口速查

```bash
# 索引
python scripts/build_index_v2.py             # 双 collection drop+rebuild

# CLI
python scripts/run_cli.py "..."              # Quick · 单次
python scripts/run_cli_memory.py --thread X "..."   # Quick · 多轮
python scripts/run_groupchat.py "..."        # Deep · 流式
python scripts/run_groupchat.py --selector "..."    # Deep · LLM 选发言者
python scripts/run_auto.py "..."             # Auto · 自动分发
python scripts/run_auto.py --force quick "..." # 跳过 Auto 决策
python scripts/run_auto.py --force quick "搜索一下 cpp-httplib 最新 release 信息"
python scripts/run_auto.py --force quick "计算 128 * 0.75 + sqrt(16)"

# 工具调试
python scripts/run_tools.py

# 测试
python -m unittest discover -s tests -p 'test_tools.py'

# UI
streamlit run ui/app.py
```

---

## 局限与后续

- **没做 Rerank**：Top-K 召回后没走 cross-encoder / bge-reranker 二阶段精排；高召回噪声场景会拖准确率。
- **没做 AST 级代码切分**：用 LangChain `Language.CPP` 按花括号切，跨 namespace / 模板特化的函数边界会被切散。
- **多语言支持有限**：仅在 C++ + Markdown 上验证；其他语言要相应调 chunk_size / Language enum。
- **Deep 模式延时高**：4 角色串行 LLM 调用 + 工具调用，单题 30s+。可优化方向：Critic 上限收紧、Analyst 与 Critic 部分并行。
- **Selector 模式未充分调优**：`SelectorGroupChat` 当前用默认 prompt，挑发言者偶发让 Reporter 提前抢戏。

---

# tutorial/

按官方教程跑通的最小可复现 demo，CodeLens 用到的每一项能力都先在这里验证过：

| 文件 | 内容 |
| --- | --- |
| `01_langchain_tutorial.py` | Runnable / Prompt / OutputParser / 结构化输出（DeepSeek `function_calling` 模式） |
| `02_langgraph_tutorial.py` | StateGraph / ToolNode / checkpointer / 条件边 / 多轮 |
| `03_milvus_tutorial.py` | Milvus Lite 双 collection 建索引 + 检索 |
| `04_llamaindex_tutorial.py` | Router + SubQuestion + Milvus backend（CodeLens 阶段 8.1 的原型） |
| `05_autogen_tutorial.py` | AssistantAgent / RoundRobinGroupChat / SelectorGroupChat（CodeLens 阶段 8.3 的原型） |

写代码遇到概念卡壳就回到 `tutorial/` 写最小复现，验完再塞回 CodeLens——这套流程在踩 LlamaIndex `{{}}` 坑、AutoGen DeepSeek 协议坑时挽救了大量调试时间。
