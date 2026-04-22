# -*- coding: utf-8 -*-
"""
文本切分工具模块。

在 RAG（检索增强生成）流程的 ingest 阶段，需要把大段的原始文本
切成若干较小的 chunk（片段），再做向量化存入向量库。
本模块提供两种切分器：
    1. split_markdown : 针对 Markdown 文档的两级切分
    2. split_cpp      : 针对 C/C++ 源码的按语法结构切分
"""

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,          # 按 Markdown 标题层级切分
    RecursiveCharacterTextSplitter,      # 按字符数递归切分（带层级分隔符）
    Language,                            # 语言枚举，用于选取语言专属的分隔符集合
)


def split_markdown(text: str):
    """把一段 Markdown 文本切成多个带结构信息的小 chunk。

    两阶段切分（先按语义再按长度），兼顾「章节完整性」和「长度可控」：
      阶段 1：按 # / ## / ### 三级标题切，使每个 chunk 的 metadata
              里带上所属章节（h1/h2/h3），供检索时回溯上下文。
      阶段 2：对过长的章节按 800 字符递归切，相邻 chunk 保留 80 字符
              重叠，防止问题 / 答案被拦腰截断而丢失语义。
    """
    # ---- 阶段 1：按标题切 ----
    header_splitter = MarkdownHeaderTextSplitter(
        # 告诉切分器：遇到 # 当作 h1，## 当作 h2，### 当作 h3
        # 切完后每个文档的 metadata 会带上 {"h1": "...", "h2": "...", ...}
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    header_docs = header_splitter.split_text(text)

    # ---- 阶段 2：按长度再切 ----
    # chunk_size=800  ：单片最长 800 字符
    # chunk_overlap=80：相邻片段重叠 80 字符，防止跨段落截断
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

    # split_documents 会保留第 1 步写入的 metadata（h1/h2/h3 章节信息）
    return size_splitter.split_documents(header_docs)


def split_cpp(code: str):
    """把一段 C/C++ 源码按语法结构切成若干 chunk。

    用 Language.CPP 时，底层会自动使用 C++ 专属的分隔符优先级：
        class / struct / 函数定义 / namespace / 空行 / 行 / 字符 ...
    也就是说，它会尽量在「类、函数」这种语义边界处切，
    而不是机械地按字符数切到半个函数体里。

    参数：
        code: 完整的 .h / .cpp 源文件文本

    chunk_size=1200  : 代码的上下文窗口可以开得比 Markdown 更大一些
    chunk_overlap=120: 保留 120 字符重叠，防止函数签名/头被截断

    注意这里用的是 create_documents（而不是 split_documents）——
    输入是纯字符串列表，输出是 Document 列表。
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CPP, chunk_size=1200, chunk_overlap=120
    )
    return splitter.create_documents([code])
