from pathlib import Path

from langchain_chroma import Chroma

from app.embeddings import get_embeddings
from app.vectorstore import get_milvus

# 锚定到 codelens/storage/chroma，避免 cwd 不同时找不到向量库
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(PROJECT_ROOT / "storage" / "chroma")


# 模块级单例：Chroma 客户端会打开 sqlite + 把向量库加载进内存，每次 new 都重做
# 很浪费。和 embeddings 一样用 lazy init 复用一个实例。
_chroma = None


def _get_chroma():
    global _chroma
    if _chroma is None:
        _chroma = Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())
    return _chroma


def get_retriever(k: int = 5):
    # as_retriever 本身是轻量包装，search_kwargs 指定 k；复用同一个 Chroma 实例即可。
    # return _get_chroma().as_retriever(search_kwargs={"k": k})
    return get_milvus().as_retriever(search_kwargs={"k": k})
