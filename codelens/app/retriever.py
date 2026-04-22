from pathlib import Path

from langchain_chroma import Chroma

from app.embeddings import get_embeddings

# 锚定到 codelens/storage/chroma，避免 cwd 不同时找不到向量库
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(PROJECT_ROOT / "storage" / "chroma")


def get_retriever(k: int = 5):
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())
    return db.as_retriever(search_kwargs={"k": k})
