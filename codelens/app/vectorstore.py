from pathlib import Path

from langchain_milvus import Milvus
from pymilvus import connections

from app.embeddings import get_embeddings


# 锚定到 codelens/storage/milvus.db（Milvus Lite 本地文件），
# 避免 cwd 不同时找不到库。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MILVUS_URI = str(PROJECT_ROOT / "storage" / "milvus.db")

# 模块级单例（和 embeddings / chroma 同样思路）。
_milvus = None


def get_milvus():
    """返回一个单例 Milvus Lite vectorstore。

    关键参数解释：
      · connection_args.uri 是本地文件路径 → pymilvus 自动走 milvus-lite 嵌入式后端
      · collection_name="codelens"       集合名，相当于一个命名空间
      · index_params                       HNSW 索引参数；M/efConstruction 越大召回率越高、内存越多
      · auto_id=True                       让 Milvus 自动生成主键 id。
         **重要**：不加 auto_id 的话，langchain-milvus 会尝试推断 schema，
         途中掉进 pymilvus 旧的 ORM `Collection` 接口，要求先 `connections.connect()`，
         而我们这个 MilvusClient 初始化路径没建 ORM 连接 → 报
         `ConnectionNotExistException: should create connection first.`。
         auto_id=True 绕开那段代码，schema 由 langchain-milvus 自己兜。
    """
    global _milvus
    if _milvus is None:
        _milvus = Milvus(
            embedding_function=get_embeddings(),
            connection_args={"uri": MILVUS_URI},
            collection_name="codelens",
            # Milvus Lite 只支持 FLAT / IVF_FLAT / AUTOINDEX；HNSW 是完整 server 版才有。
            # AUTOINDEX 让引擎自动挑（小规模一般就是 FLAT，精确搜索、没召回损失）。
            # 未来上 Milvus server 时，改回 {"index_type": "HNSW",
            #   "params": {"M": 16, "efConstruction": 200}} 即可。
            index_params={
                "metric_type": "COSINE",
                "index_type": "AUTOINDEX",
            },
            auto_id=True,
        )

        # --- 兼容性补丁：同时注册 ORM 连接 ---
        # 背景：langchain-milvus 同时踩两套 pymilvus API：
        #   1) 新 MilvusClient（由上面的 connection_args 初始化，self.alias = client._using）
        #   2) 老 ORM Collection(name, using=alias)（add_documents 触发 _extract_fields 时用到）
        # MilvusClient 不会自动给 ORM 注册连接，于是 add 时会爆
        #     ConnectionNotExistException: should create connection first.
        # 这里拿到 Milvus 实际生成的随机 alias，用相同 uri 注册一把，两套就都能用了。
        alias = _milvus.alias
        if alias and alias not in connections.list_connections():
            connections.connect(alias=alias, uri=MILVUS_URI)
    return _milvus
