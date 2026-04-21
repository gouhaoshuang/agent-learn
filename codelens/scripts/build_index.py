# -*- coding: utf-8 -*-
"""
构建向量索引的入口脚本。

流程：
    1. 扫描 data/docs/ 下的所有 Markdown 文件 → 用 split_markdown 切片
    2. 扫描 data/code/ 下的 C/C++ 源码       → 用 split_cpp     切片
    3. 给每个 chunk 打上 source / type 元数据（检索时用于回溯来源和过滤）
    4. 全部喂给 Chroma，生成持久化到 storage/chroma 的向量库

运行方式（需在 codelens/ 根目录下执行，以保证 app.* / ingest.* 可被导入）：
    python -m ingest.build_index
或：
    cd codelens && python ingest/build_index.py
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.embeddings import get_embeddings
# 原代码写的是 `from app.ingest.splitter import ...`，但 ingest/ 目录位于
# codelens/ 根下、与 app/ 平级，并不是 app 的子模块，所以会报 ModuleNotFoundError。
from ingest.splitter import split_markdown, split_cpp


# Chroma 持久化目录（相对当前工作目录）。
# 重跑时会在同一目录下追加/更新向量，不会自动清空。
PERSIST_DIR = "storage/chroma"

# 视作 C/C++ 源码的文件后缀集合。
CPP_SUFFIXES = {".cpp", ".h", ".hpp", ".cc"}


def build(docs_dir: str = "data/docs", code_dir: str = "data/code") -> None:
    """扫描文档和代码目录，切片后建向量索引。

    参数：
        docs_dir: Markdown 文档根目录
        code_dir: C/C++ 源码根目录
    """
    chunks: list[Document] = []

    # ---- 1. Markdown 文档切片 ----
    # rglob 递归遍历所有 .md 文件；errors="ignore" 是为了防止个别文件
    # 非 UTF-8 编码导致整体失败（对 RAG 来说容错比严格更重要）。
    for md_path in Path(docs_dir).rglob("*.md"):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        for chunk in split_markdown(text):
            # 记录来源路径和类型，检索时可用 metadata 过滤（只搜 doc 或只搜 code）
            chunk.metadata["source"] = str(md_path)
            chunk.metadata["type"] = "doc"
            chunks.append(chunk)

    # ---- 2. C/C++ 源码切片 ----
    # 这里用 rglob("*") 再按后缀过滤，而不是 rglob("*.cpp")，
    # 是为了一次遍历覆盖 .cpp / .h / .hpp / .cc 多种后缀。
    for code_path in Path(code_dir).rglob("*"):
        if code_path.suffix not in CPP_SUFFIXES:
            continue
        code = code_path.read_text(encoding="utf-8", errors="ignore")
        for chunk in split_cpp(code):
            chunk.metadata["source"] = str(code_path)
            chunk.metadata["type"] = "code"
            chunks.append(chunk)

    # ---- 3. 建库 ----
    # Chroma.from_documents 会把 chunks 做 embedding 并写入 persist_directory，
    # 本身带副作用（落盘）。返回的 db 对象在本脚本里无用，所以不再赋值。
    if not chunks:
        print(f"no chunks found under {docs_dir!r} or {code_dir!r}, skip indexing")
        return

    Chroma.from_documents(
        chunks,
        embedding=get_embeddings(),
        
        persist_directory=PERSIST_DIR,
    )
    print(f"indexed {len(chunks)} chunks into {PERSIST_DIR}")


if __name__ == "__main__":
    build()
