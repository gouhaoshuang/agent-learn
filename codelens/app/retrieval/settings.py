# -*- coding: utf-8 -*-
"""
LlamaIndex 全局 Settings 初始化。

LlamaIndex 的设计是"全局单例"：所有 Index / QueryEngine / Document 处理都会
从 `llama_index.core.Settings` 拿 `llm` 和 `embed_model`，而不是像 LangChain
那样每个 chain 显式传入。这文件就是那唯一一处装配。

调用方在使用 LlamaIndex 任何能力前，必须先调用一次 `setup_llamaindex()`。
重复调用是幂等的（首次构造，之后直接返回）。

为什么不直接 import 时 setup？
  · embedding 是 gte-Qwen2-1.5B-instruct，加载约 3-4GB 显存，不能在 import
    时无脑触发——只有真要建索引/查询时再加载。
  · LLM 走 DeepSeek HTTP，构造便宜，但放一起统一管理更清楚。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


_initialized = False


def _resolve_local_or_hub_id(hub_id: str) -> str:
    """如果 HF cache 里有该模型的完整 snapshot，返回绝对路径；否则返回原 hub id。

    HF cache 结构：
        $HF_HOME/hub/models--<org>--<name>/refs/main          # 指向当前 revision
        $HF_HOME/hub/models--<org>--<name>/snapshots/<rev>/   # 该 revision 的所有文件（符号链接）
    """
    hf_home = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    cache_dir = Path(hf_home) / "hub" / f"models--{hub_id.replace('/', '--')}"
    refs_main = cache_dir / "refs" / "main"
    if not refs_main.exists():
        return hub_id
    rev = refs_main.read_text().strip()
    snap = cache_dir / "snapshots" / rev
    # 简单校验：snapshot 目录里有 config.json 就认为模型基本完整
    if (snap / "config.json").exists():
        return str(snap)
    return hub_id


def setup_llamaindex() -> None:
    """幂等：首次调用装配 LlamaIndex 全局 Settings；之后直接返回。"""
    global _initialized
    if _initialized:
        return

    # .env 在仓库根目录（codelens 的上一级）
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

    from llama_index.core import Settings
    from llama_index.llms.openai_like import OpenAILike
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import torch

    # ---- LLM ----
    # 模型选 deepseek-chat 别名（不是 deepseek-v4-pro / deepseek-v4-flash 真名）。
    # 真名是 thinking 模型，要求下一轮回传 `reasoning_content` 字段；客户端 SDK
    # 在多轮调用时丢这个字段会被服务端 400 拒绝。别名走兼容层，不要求该字段。
    # （详见 docs/resume/03_LlamaIndex_AutoGen_整合实施计划.md 第 6.1 节）
    Settings.llm = OpenAILike(
        model="deepseek-chat",
        api_base=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        is_chat_model=True,           # DeepSeek 走 /v1/chat/completions，不是 /completions
        is_function_calling_model=True,
        temperature=0.2,
        timeout=60,
    )

    # ---- Embedding ----
    # 沿用 codelens 现有选型（见 app/embeddings.py）：gte-Qwen2-1.5B-instruct，
    # 中英混合 + 1.5B 规模，在 cpp-httplib 这种代码 + 英文文档场景表现稳。
    # device 自适应：有 GPU 走 GPU，否则 CPU（1.5B 在 CPU 上 embed 几千 chunk 也是几分钟级）。
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 优先用本地 HF snapshot 路径，绕开 hub/镜像版本检查的 do_poll 卡死。
    # 现象：HF_ENDPOINT 指向 hf-mirror.com 时，即使 cache 完整，from_pretrained
    # 也会先去镜像确认 modeling_qwen.py 是否有新版本（trust_remote_code 路径），
    # 镜像不稳时会 poll 几十分钟无响应。直接给绝对路径让 transformers 把它当
    # 本地目录，跳过任何 hub 调用。
    model_name = _resolve_local_or_hub_id("Alibaba-NLP/gte-Qwen2-1.5B-instruct")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        device=device,
        trust_remote_code=True,        # gte-Qwen2 仓库带自定义代码
        embed_batch_size=16,           # 和现有 LangChain 版本对齐；OOM 调小到 8/4
    )

    # ---- 切分参数（仅作 Settings 默认值；本项目实际切片走自家 splitter）----
    # LlamaIndex 默认会用 `SentenceSplitter`，对 Markdown 和 C++ 不友好。
    # 我们沿用 langchain 那套 split_markdown / split_cpp（见 app/ingest/splitter.py），
    # 转成 LlamaIndex `Document` 时把 chunk 直接当 Document.text 喂进去——
    # 等于 LlamaIndex 拿到的就已经是切好的小片段，它的 splitter 不会再二次切。
    # 这里把默认 chunk_size 调到一个对"已切好"的片段也安全的值，防止 LlamaIndex
    # 内部某些路径（比如 SubQuestionQueryEngine 合成 prompt）按默认 1024 截断。
    Settings.chunk_size = 1500
    Settings.chunk_overlap = 0       # 我们的片段已经在 splitter 阶段做过 overlap 了

    _initialized = True
