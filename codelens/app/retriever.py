from langchain_chroma import Chroma
from app.embeddings import get_embeddings

def get_retriever(k: int = 5):
    db = Chroma(persist_directory="storage/chroma", embedding_function=get_embeddings())
    return db.as_retriever(search_kwargs={"k": k})