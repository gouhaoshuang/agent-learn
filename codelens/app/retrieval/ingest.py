# -*- coding: utf-8 -*-
"""
LlamaIndex ingest pipeline：扫语料 → 切片 → 写两个 collection。

为什么继续用现有 splitter（split_markdown / split_cpp）而不换 LlamaIndex 自带的？
  · 现有 splitter 是 LangChain 的 MarkdownHeaderTextSplitter（带 h1/h2/h3
    metadata 自动回填）+ Language.CPP RecursiveCharacterTextSplitter，针对
    Markdown 的章节边界和 C++ 的语法边界都已调好。
  · LlamaIndex 默认的 SentenceSplitter 对代码不友好（按句号切，C++ 一刀下去
    切到函数体里）；它的 CodeSplitter 又只对 Python/JS/Go 等少数语言原生支持，
    C++ 是空的。
  · 切完之后把每个 chunk 的 page_content 包成 LlamaIndex `Document`，metadata
    一同传进去——LlamaIndex 不会再二次切（chunk_size 已在 settings 里调高），
    一片就是一片。

输入：data/docs（markdown）+ data/code（C/C++）
输出：codelens_docs + codelens_code 两个 Milvus collection
"""

from __future__ import annotations

import time
from pathlib import Path

from tqdm import tqdm

from app.ingest.splitter import split_markdown, split_cpp
from app.retrieval.settings import setup_llamaindex
from app.retrieval.vector_stores import get_docs_store, get_code_store


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS_DIR = PROJECT_ROOT / "data" / "docs"
DEFAULT_CODE_DIR = PROJECT_ROOT / "data" / "code"
CPP_SUFFIXES = {".cpp", ".h", ".hpp", ".cc"}


def _lc_to_li_documents(lc_docs, default_metadata: dict):
    """LangChain Document → LlamaIndex Document。

    LangChain Document 是 `page_content` + `metadata`；LlamaIndex Document 是
    `text` + `metadata`，结构基本一样，转换是一对一的字段映射。

    `default_metadata` 是该批所有片段共有的 metadata（如 type=doc/code），会和
    每片自己的 metadata（splitter 阶段可能写入的 h1/h2/h3 章节标题）合并。
    """
    from llama_index.core import Document

    out = []
    for lc in lc_docs:
        meta = {**default_metadata, **(lc.metadata or {})}
        out.append(Document(text=lc.page_content, metadata=meta))
    return out


def collect_doc_documents(docs_dir: Path):
    """扫 markdown → split_markdown → 转成 LlamaIndex Documents。"""
    md_files = list(docs_dir.rglob("*.md"))
    print(f"[scan] markdown files under {docs_dir}: {len(md_files)}")
    docs = []
    for md_path in tqdm(md_files, desc="split md", unit="file"):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        lc_chunks = split_markdown(text)
        docs.extend(_lc_to_li_documents(
            lc_chunks,
            default_metadata={"source": str(md_path), "type": "doc"},
        ))
    print(f"[scan] md chunks      : {len(docs)}")
    return docs


def collect_code_documents(code_dir: Path):
    """扫 C/C++ → split_cpp → 转成 LlamaIndex Documents。"""
    code_files = [p for p in code_dir.rglob("*") if p.suffix in CPP_SUFFIXES]
    print(f"[scan] C/C++ files under {code_dir}: {len(code_files)}")
    docs = []
    for code_path in tqdm(code_files, desc="split cpp", unit="file"):
        code = code_path.read_text(encoding="utf-8", errors="ignore")
        lc_chunks = split_cpp(code)
        docs.extend(_lc_to_li_documents(
            lc_chunks,
            default_metadata={"source": str(code_path), "type": "code"},
        ))
    print(f"[scan] cpp chunks     : {len(docs)}")
    return docs


def _build_one(documents, vector_store, label: str):
    """把一批 LlamaIndex Documents 写进指定 vector_store。"""
    from llama_index.core import VectorStoreIndex, StorageContext

    if not documents:
        print(f"[{label}] no documents, skip")
        return

    storage = StorageContext.from_defaults(vector_store=vector_store)
    t0 = time.time()
    # `from_documents` 内部会调 Settings.embed_model 把每篇 Document 切→embed→写库；
    # 因为我们 settings 里把 chunk_size 调高、切片已在 splitter 阶段做完，这一步
    # 实际上是"按片段直接 embed + 入库"，不会再二次切。
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage,
        show_progress=True,
    )
    print(f"[{label}] embed+write done, {len(documents)} docs in {time.time()-t0:.1f}s")


def build(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    code_dir: Path = DEFAULT_CODE_DIR,
    overwrite: bool = True,
) -> None:
    """重建索引主入口。overwrite=True 会清掉同名 collection 重建。"""
    setup_llamaindex()

    t0 = time.time()
    print(f"[init] LlamaIndex Settings ready (LLM=deepseek-chat, embed=gte-Qwen2-1.5B)")

    doc_documents = collect_doc_documents(Path(docs_dir))
    code_documents = collect_code_documents(Path(code_dir))
    total = len(doc_documents) + len(code_documents)
    print(f"[scan] total chunks   : {total}  (elapsed {time.time()-t0:.1f}s)")

    if total == 0:
        print("[done] no chunks found, abort")
        return

    # overwrite=True 在 vector_store 构造时就 drop 同名 collection 再建；
    # 对每个 collection 分别构造一次。
    docs_store = get_docs_store(overwrite=overwrite)
    code_store = get_code_store(overwrite=overwrite)

    _build_one(doc_documents, docs_store, label="docs")
    _build_one(code_documents, code_store, label="code")

    print(f"[done] indexed {total} chunks (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    build()
