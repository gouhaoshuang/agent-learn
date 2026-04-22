from langchain_core.messages import trim_messages
from app.llm import get_llm

def trim(messages, max_tokens: int = 3000):
    return trim_messages(
        messages, max_tokens=max_tokens,
        token_counter=get_llm(), strategy="last",
        include_system=True, allow_partial=False,
    )