# -*- coding: utf-8 -*-
"""
构建向量索引的入口脚本。

流程：
    1. 扫描 data/docs/ 下的所有 Markdown 文件 → 用 split_markdown 切片
    2. 扫描 data/code/ 下的 C/C++ 源码       → 用 split_cpp     切片
    3. 给每个 chunk 打上 source / type 元数据（检索时用于回溯来源和过滤）
    4. 分批喂给 Chroma（带进度条），生成持久化到 storage/chroma 的向量库

运行方式（在任意目录都可以，路径已锚定 PROJECT_ROOT）：
    python scripts/build_index.py
"""

import sys
import time
from pathlib import Path

# 把 codelens/（本脚本的父目录的父目录）加到 sys.path，
# 以便 `from app.*` / `from ingest.*` 在任意工作目录下都能导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from app.embeddings import get_embeddings
from ingest.splitter import split_markdown, split_cpp


PERSIST_DIR = str(PROJECT_ROOT / "storage" / "chroma")
DEFAULT_DOCS_DIR = str(PROJECT_ROOT / "data" / "docs")
DEFAULT_CODE_DIR = str(PROJECT_ROOT / "data" / "code")
CPP_SUFFIXES = {".cpp", ".h", ".hpp", ".cc"}

# embed + 写库的批大小。Qwen2-1.5B 在 L40 上 batch=16 很稳；OOM 就调小到 8/4。
EMBED_BATCH_SIZE = 16


def _collect_md_chunks(docs_dir: str) -> list[Document]:
    chunks: list[Document] = []
    md_files = list(Path(docs_dir).rglob("*.md"))
    print(f"[scan] markdown files under {docs_dir}: {len(md_files)}")
    for md_path in tqdm(md_files, desc="split md", unit="file"):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        for chunk in split_markdown(text):
            chunk.metadata["source"] = str(md_path)
            chunk.metadata["type"] = "doc"
            chunks.append(chunk)
    print(f"[scan] md chunks      : {len(chunks)}")
    return chunks


def _collect_cpp_chunks(code_dir: str) -> list[Document]:
    chunks: list[Document] = []
    code_files = [p for p in Path(code_dir).rglob("*") if p.suffix in CPP_SUFFIXES]
    print(f"[scan] C/C++ files under {code_dir}: {len(code_files)}")
    for code_path in tqdm(code_files, desc="split cpp", unit="file"):
        code = code_path.read_text(encoding="utf-8", errors="ignore")
        for chunk in split_cpp(code):
            chunk.metadata["source"] = str(code_path)
            chunk.metadata["type"] = "code"
            chunks.append(chunk)
    print(f"[scan] cpp chunks     : {len(chunks)}")
    return chunks


def build(docs_dir: str = DEFAULT_DOCS_DIR, code_dir: str = DEFAULT_CODE_DIR) -> None:
    t0 = time.time()

    # ---- 1/2. 扫描 + 切片 ----
    md_chunks = _collect_md_chunks(docs_dir)
    cpp_chunks = _collect_cpp_chunks(code_dir)
    chunks = md_chunks + cpp_chunks
    print(f"[scan] total chunks   : {len(chunks)}  (elapsed {time.time()-t0:.1f}s)")

    if not chunks:
        print(f"[done] no chunks found under {docs_dir!r} or {code_dir!r}")
        return

    # ---- 3. 初始化 embedding + 空库 ----
    print(f"[init] loading embedding model ...")
    emb = get_embeddings()

    print(f"[init] opening chroma at {PERSIST_DIR}")
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)

    # ---- 4. 分批 embed + 写库 ----
    # 不用 Chroma.from_documents 是因为它一把吞所有 chunk，不给进度条。
    # 分批 add_documents 才能让每个 batch 走完时看到进度和每秒处理量。
    t1 = time.time()
    n_total = len(chunks)
    with tqdm(total=n_total, desc="embed+write", unit="chunk") as bar:
        for i in range(0, n_total, EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            db.add_documents(batch)
            bar.update(len(batch))

    t2 = time.time()
    print(
        f"[done] indexed {n_total} chunks into {PERSIST_DIR}  "
        f"(embed {t2 - t1:.1f}s, total {t2 - t0:.1f}s)"
    )


if __name__ == "__main__":
    build()
