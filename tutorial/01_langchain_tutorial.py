"""
===========================================================
LangChain 入门教程
===========================================================
本教程基于 test.py 扩展，带你一步步了解 LangChain 的核心概念：
  1. 环境准备与 API Key 配置
  2. 初始化 Chat 模型 (ChatOpenAI)
  3. Prompt 模板 (ChatPromptTemplate)
  4. 输出解析器 (Output Parser)
  5. LCEL 链式调用 (|  管道符)
  6. 多轮对话 & 消息历史
  7. 流式输出 (stream)
  8. 结构化输出 (with_structured_output)
  9. 简单的 RAG 示例思路

运行前准备：
    pip install langchain langchain-openai python-dotenv pydantic

在项目根目录创建 .env 文件，并填入：
    OPENAI_API_KEY=sk-xxxxx
    OPENAI_BASE_URL=https://your-endpoint/v1   # 可选，例如使用 DeepSeek/通义等兼容接口
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件，LangChain 的模型会自动从环境变量读取 API Key
# 本目录没有 .env 时，回溯到上一级查找
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# -----------------------------------------------------------
# 1. 初始化 Chat 模型
# -----------------------------------------------------------
# ChatOpenAI 是对 OpenAI 兼容 Chat Completion 接口的封装
# temperature 控制随机性：0 更稳定，1 更发散
# 这里对接 DeepSeek 的 OpenAI 兼容接口，模型名用 deepseek-chat
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    base_url=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


# -----------------------------------------------------------
# 2. Prompt 模板：把"变量 + 模板"组合成对话消息
# -----------------------------------------------------------
# ChatPromptTemplate 可以写成一个由 role/content 组成的列表
# 中文变量用 {name} 占位符，调用时再填入
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的{domain}专家，回答要通俗易懂。"),
    ("human", "请解释一下什么是{topic}？"),
])


# -----------------------------------------------------------
# 3. 输出解析器：把模型返回的 AIMessage 转成纯字符串
# -----------------------------------------------------------
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()


# -----------------------------------------------------------
# 4. LCEL 链式调用
#    LangChain Expression Language 用 `|` 把组件串起来：
#    prompt -> model -> parser
# -----------------------------------------------------------
chain = prompt | model | parser


def demo_basic_chain():
    """最基础的一次性调用"""
    result = chain.invoke({"domain": "人工智能", "topic": "Transformer"})
    print("=== Demo 1: 基础链 ===")
    print(result)
    print()


# -----------------------------------------------------------
# 5. 多轮对话：使用消息列表 + MessagesPlaceholder
# -----------------------------------------------------------
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位耐心的中文助教。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chat_chain = chat_prompt | model | parser


def demo_multi_turn():
    """演示如何手动维护对话历史"""
    history = []

    # 第一轮
    q1 = "什么是向量数据库？"
    a1 = chat_chain.invoke({"history": history, "input": q1})
    history += [HumanMessage(content=q1), AIMessage(content=a1)]

    # 第二轮（依赖上文）
    q2 = "那它和传统关系型数据库最大的区别是？"
    a2 = chat_chain.invoke({"history": history, "input": q2})

    print("=== Demo 2: 多轮对话 ===")
    print("Q1:", q1, "\nA1:", a1)
    print("Q2:", q2, "\nA2:", a2)
    print()


# -----------------------------------------------------------
# 6. 流式输出：边生成边打印，用户体验更好
# -----------------------------------------------------------
def demo_stream():
    print("=== Demo 3: 流式输出 ===")
    for chunk in chain.stream({"domain": "前端", "topic": "虚拟 DOM"}):
        print(chunk, end="", flush=True)
    print("\n")


# -----------------------------------------------------------
# 7. 结构化输出：让模型按 Pydantic 模型返回 JSON
# -----------------------------------------------------------
from pydantic import BaseModel, Field


class Concept(BaseModel):
    """一个技术概念的结构化描述"""
    name: str = Field(description="概念名称")
    one_liner: str = Field(description="一句话解释")
    key_points: list[str] = Field(description="3-5 个关键点")
    analogy: str = Field(description="生活化的类比")


def demo_structured_output():
    # DeepSeek 等兼容接口未实现 json_schema 的 response_format，
    # 改用 function_calling 模式（通过工具调用拿到结构化参数）
    structured_model = model.with_structured_output(Concept, method="function_calling")
    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是技术科普作者，用 JSON 结构化输出。"),
        ("human", "请解释：{topic}"),
    ])
    structured_chain = explain_prompt | structured_model

    result: Concept = structured_chain.invoke({"topic": "反向传播"})
    print("=== Demo 4: 结构化输出 ===")
    print("name        :", result.name)
    print("one_liner   :", result.one_liner)
    print("key_points  :", result.key_points)
    print("analogy     :", result.analogy)
    print()


# -----------------------------------------------------------
# 8. 简单 RAG 思路（伪代码说明）
# -----------------------------------------------------------
# 真正的 RAG 需要 embedding + 向量库，这里只给出最小示意：
#   1) 把知识切片 -> 嵌入 -> 写入向量库 (Chroma / FAISS)
#   2) 用户提问时，先检索 top-k 相关片段
#   3) 把检索结果拼进 prompt 的 {context}
#
# rag_prompt = ChatPromptTemplate.from_messages([
#     ("system", "仅依据以下资料回答：\n{context}"),
#     ("human", "{question}"),
# ])
# rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | model | parser


# -----------------------------------------------------------
# 入口
# -----------------------------------------------------------
if __name__ == "__main__":
    demo_basic_chain()
    demo_multi_turn()
    demo_stream()
    demo_structured_output()
