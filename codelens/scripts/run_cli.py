from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.llm import get_llm
from app.retriever import get_retriever

retriever = get_retriever()
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是代码/文档问答助手。仅依据下面的【资料】回答，若不足请如实说。\n\n【资料】\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(f"[{i}] {d.metadata.get('source')}\n{d.page_content}"
                       for i, d in enumerate(docs))

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | get_llm() | StrOutputParser()
)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "ThreadPool 如何实现？"
    print(chain.invoke(q))