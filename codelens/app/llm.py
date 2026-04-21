import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        temperature=temperature,
        base_url=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
