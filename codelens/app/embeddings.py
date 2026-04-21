from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-multilingual-gemma2",  # 中英混合 && 小，跑得快
        encode_kwargs={"normalize_embeddings": True},
    )