import torch
# 用 langchain-huggingface 包（官方新位置）而不是 langchain_community.embeddings；
# 后者自 LangChain 0.2.2 起已 Deprecation，1.0 里会删。
from langchain_huggingface import HuggingFaceEmbeddings


# 模块级单例：1.5B 的 gte-Qwen2 加载一次约占 3~4GB 显存，每次 new 一个会反复
# 上传 GPU（日志里反映为 "Loading checkpoint shards" 重复出现）。用 None 哨兵
# 做 lazy init，进程内只构造一次。
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    # 显式选 device：有 GPU 就走 GPU，没有就回退 CPU。
    # 不写这行时，sentence-transformers 某些版本默认 CPU，在 1.5B 模型上会慢几十倍。
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _embeddings = HuggingFaceEmbeddings(
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
    return _embeddings
