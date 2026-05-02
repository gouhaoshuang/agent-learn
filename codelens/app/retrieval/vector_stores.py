# -*- coding: utf-8 -*-
"""
两个 Milvus collection 的工厂函数：`codelens_docs` / `codelens_code`。

设计要点：
  · 同一个 Milvus Lite 文件（storage/milvus.db）放两个 collection，按用途隔离。
    Milvus 的 collection 之间是独立 schema，互不干扰。
  · LlamaIndex 自家的 `MilvusVectorStore`（来自 llama-index-vector-stores-milvus）
    不会触碰 langchain-milvus 那条同时踩 MilvusClient + ORM 的 hack 路径，schema
    干净。
  · 模块级单例：每个 collection 一个全局实例，避免重复初始化造成连接泄漏。
  · 索引参数：Milvus Lite 只支持 FLAT / IVF_FLAT / AUTOINDEX，HNSW 是 server 版
    才有；这里走 AUTOINDEX 让引擎自动挑（小规模一般就是 FLAT，精确搜索）。

向量维度（dim）说明：
  · gte-Qwen2-1.5B-instruct 的输出维度是 1536。
  · LlamaIndex 的 `MilvusVectorStore` 在首次写入时如果 collection 不存在，
    会用 `dim` 参数建表；如果不传 dim，会从第一条向量推断——更稳的做法是显式
    给定，避免推断失败时报莫名其妙的 schema 错。
"""

from __future__ import annotations

from pathlib import Path

# 锚定到 codelens/storage/milvus.db
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MILVUS_URI = str(PROJECT_ROOT / "storage" / "milvus.db")

# gte-Qwen2-1.5B-instruct 的 embedding 维度
EMBED_DIM = 1536

DOCS_COLLECTION = "codelens_docs"
CODE_COLLECTION = "codelens_code"


_docs_store = None
_code_store = None


def _make_store(collection_name: str, overwrite: bool = False):
    """构造 LlamaIndex MilvusVectorStore；overwrite=True 会清掉同名 collection。"""
    from llama_index.vector_stores.milvus import MilvusVectorStore

    return MilvusVectorStore(
        uri=MILVUS_URI,
        collection_name=collection_name,
        dim=EMBED_DIM,
        overwrite=overwrite,
        similarity_metric="COSINE",
    )


def get_docs_store(overwrite: bool = False):
    """文档 collection（存 markdown 切片）。overwrite 仅在 ingest 重建时用。"""
    global _docs_store
    if _docs_store is None or overwrite:
        _docs_store = _make_store(DOCS_COLLECTION, overwrite=overwrite)
    return _docs_store


def get_code_store(overwrite: bool = False):
    """代码 collection（存 C++ 切片）。overwrite 仅在 ingest 重建时用。"""
    global _code_store
    if _code_store is None or overwrite:
        _code_store = _make_store(CODE_COLLECTION, overwrite=overwrite)
    return _code_store
