import torch
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    # 显式选 device：有 GPU 就走 GPU，没有就回退 CPU。
    # 不写这行时，sentence-transformers 某些版本默认 CPU，在 1.5B 模型上会慢几十倍。
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        model_kwargs={
            "device": device,
            "trust_remote_code": True,  # gte-Qwen2 仓库带自定义代码
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 16,  # GPU 上跑 1.5B 模型的合理 batch，OOM 就调小到 8/4
        },
    )
