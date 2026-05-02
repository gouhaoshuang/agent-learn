# -*- coding: utf-8 -*-
"""
LlamaIndex 双 collection 索引重建入口（阶段 8.1）。

和老的 `scripts/build_index.py` 的关系：
  · 老脚本走 LangChain Chroma / langchain-milvus → 单 collection（codelens）
  · 本脚本走 LlamaIndex MilvusVectorStore → 两个独立 collection
    （codelens_docs / codelens_code）
  · 老的 collection 不会自动清理。如果之前用过，建议 `rm storage/milvus.db`
    后再跑本脚本——文档计划 2.1 节的关键决策就是"重建索引、不迁移数据"。

运行（在任意目录）：
    python scripts/build_index_v2.py
"""

import sys
from pathlib import Path

# 把 codelens/（本脚本父目录的父目录）加到 sys.path，
# 让 `from app.*` 在任意 cwd 下都能 import。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from app.retrieval.ingest import build
    # overwrite=True：drop 同名 collection 再建，避免重复入库累积脏数据
    build(overwrite=True)


if __name__ == "__main__":
    main()
