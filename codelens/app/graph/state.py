from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class CodeLensState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iterations: int          # ReAct 已循环几次
    reflection: str          # 反思节点输出