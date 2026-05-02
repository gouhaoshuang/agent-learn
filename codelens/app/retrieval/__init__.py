# -*- coding: utf-8 -*-
"""
LlamaIndex 检索层（阶段 8.1）。

为什么单独立一个 `retrieval/` 模块、而不是直接改 `app/retriever.py`？
  · 旧的 `retriever.py` 是 LangChain `as_retriever()` 一行包装，路径锚定的是 Milvus
    单 collection（langchain-milvus 写入），schema 上踩了 MilvusClient 新 API
    + 老 ORM Collection 两套（见 `vectorstore.py` 兼容性补丁注释）。
  · 本模块用 LlamaIndex 重建一遍：双独立 collection（docs / code）+ Router 路由
    + SubQuestion 子问题分解；底座是 LlamaIndex 自家的 `MilvusVectorStore`，不
    再触碰 langchain-milvus 那条线。
  · 旧的 `retriever.py` 暂时保留（向上兼容现有 LangGraph 调用），后续 8.1 步骤 4
    会把它内部实现替换成 `router.query(...)`。

子模块：
    settings.py      —— LlamaIndex 全局 Settings（LLM + embedding）
    vector_stores.py —— 两个 MilvusVectorStore 工厂（docs / code 各一个 collection）
    ingest.py        —— 扫语料 → 切片 → 写两个 collection
    router.py        —— RouterQueryEngine + SubQuestionQueryEngine 组装
"""
