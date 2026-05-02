"""
===========================================================
Milvus 入门教程
===========================================================
本教程承接 01_langchain_tutorial.py / 02_langgraph_tutorial.py。
前两份教你怎么调模型、怎么用图编排 Agent；本教程教你 Agent 背后那块
"装着 embedding 的箱子"——向量数据库 Milvus。

为什么需要 Milvus？
  · 关系数据库（MySQL）按"等值/范围"查；倒排索引（ES）按"关键词"查；
    向量数据库按"语义相似度"查 —— 在 N 个高维向量里找跟 query 最近的 top-k。
  · sklearn 的 NearestNeighbors 也能算 top-k，但它一次性把全部向量加载到内存，
    换数据要重建、换机器要重传、上千万条就跑不动；
  · Milvus 把"持久化 + 索引算法（HNSW/IVF/...）+ 增删改查 + 标量字段过滤"
    工程化地封了一层，是 RAG/检索/推荐场景的标配存储。

Milvus 三种部署形态：
  · Milvus Lite      —— 一个本地文件 (*.db)，零依赖，pip 装 milvus-lite 即用，
                        最适合学习与小规模 demo（CodeLens 现在用的就是它）。
  · Milvus Standalone —— 单机 docker，包含 etcd / minio / milvus 三件套。
  · Milvus Cluster    —— k8s 分布式集群，亿级向量 + 高可用。
本教程**全部用 Milvus Lite**，跑通后切到 Standalone/Cluster 改一行 URI 即可。

⚠️ 平台限制：Milvus Lite 只支持 Linux/macOS。Windows 用户请走 Docker Standalone。

本文件按 Demo 顺序看完，你能：
  Demo 1  连接 Milvus Lite + 用"快速模式"建一个 collection 并插数据
  Demo 2  显式 Schema：PrimaryKey / Vector / 标量字段 + 索引参数
          （HNSW / IVF / FLAT 三种索引、COSINE / L2 / IP 三种距离）
  Demo 3  Top-K 向量搜索（这是向量库的看家本领）
  Demo 4  标量字段过滤：`category == "code"` 这类 boolean expr
  Demo 5  端到端语义检索：用 sentence-transformers 把文本变 embedding，再做检索
  Demo 6  Upsert / Delete / Get by id：增量维护 collection
  Demo 7  持久化与重连：进程重启数据还在
  Demo 8  把 Milvus 接进 LangChain 检索链（这就是 CodeLens 阶段 5 的做法）

运行前准备（agent 环境）：
    pip install pymilvus>=2.4 milvus-lite
    pip install sentence-transformers      # Demo 5/8 才需要

可以单独跑某一个 demo：
    python 03_milvus_tutorial.py 1     # 只跑 Demo 1
    python 03_milvus_tutorial.py       # 全跑
"""

import os
import shutil
from pathlib import Path

# 教程产物（向量库文件）放到 tutorial/ 同级的 storage/ 里，跑完不污染主项目。
TUTORIAL_DIR = Path(__file__).resolve().parent
STORAGE_DIR = TUTORIAL_DIR / "_milvus_storage"
STORAGE_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------
# 公共：一个返回 MilvusClient 的工厂
# -----------------------------------------------------------
# 用 MilvusClient（pymilvus 2.4+ 推荐）。它统一了 Lite/Standalone/Cluster 三种连接：
#   · Lite      : MilvusClient(uri="./xxx.db")      —— uri 是本地文件路径
#   · Standalone: MilvusClient(uri="http://host:19530")
#   · Cluster   : MilvusClient(uri="https://xxx.zillizcloud.com", token="...")
# API 完全一致，本教程的代码改一行 uri 就能直接上集群。
# -----------------------------------------------------------
from pymilvus import MilvusClient, DataType


def get_client(db_name: str = "tutorial.db") -> MilvusClient:
    """每个 demo 用独立的 .db 文件，互不干扰。"""
    uri = str(STORAGE_DIR / db_name)
    return MilvusClient(uri=uri)


# ===========================================================
# Demo 1 — 连接 + 快速模式建 collection + 插入 + 搜索
# ===========================================================
# 核心对象：
#   · MilvusClient                       —— 客户端
#   · client.create_collection(name, dim) —— "快速模式"：只填名字和维度，
#                                            其它字段（id / vector / metric / 索引）走默认
#   · client.insert(name, [{...}, ...])   —— 插入；data 是 dict 列表
#   · client.search(name, data, limit)    —— top-k 搜索；data 是 query 向量列表
#
# 快速模式默认：
#   · 主键字段名 "id"，自动生成 / 也可手动指定
#   · 向量字段名 "vector"，类型 FLOAT_VECTOR
#   · metric_type=COSINE，建 AUTOINDEX（Lite 上自动选合适算法）
# 学完 Demo 1 你会发现 30 行代码就能跑通整条向量库流水线。
# -----------------------------------------------------------
import random


def demo_1_quick_start():
    client = get_client("01_quick.db")
    name = "demo1"

    # 重跑 demo 时先清掉旧 collection（生产环境不要这么写！）
    if client.has_collection(name):
        client.drop_collection(name)

    # 1) 建 collection：只声明维度，其它走默认
    client.create_collection(collection_name=name, dimension=8)

    # 2) 准备 5 条假数据。注意 vector 字段必须是 List[float]，长度 == dimension
    rng = random.Random(42)
    data = [
        {"id": i, "vector": [rng.random() for _ in range(8)], "text": f"item-{i}"}
        for i in range(5)
    ]
    client.insert(collection_name=name, data=data)

    # 3) 用一个新向量去查最近的 3 个
    query_vec = [rng.random() for _ in range(8)]
    results = client.search(
        collection_name=name,
        data=[query_vec],          # 注意是"向量列表"，可以一次查多个 query
        limit=3,
        output_fields=["text"],    # 想让结果带哪些标量字段就在这里写
    )

    print("=== Demo 1: Milvus Lite 快速上手 ===")
    print(f"collection 已建，插了 {len(data)} 条；现在搜索 top-3：")
    # results 是 List[List[hit]] —— 外层每个 query 一个列表
    for hits in results:
        for h in hits:
            # h 是 dict：{"id": ..., "distance": ..., "entity": {...output_fields...}}
            print(f"  id={h['id']}  distance={h['distance']:.4f}  text={h['entity']['text']}")
    print()


# ===========================================================
# Demo 2 — 显式 Schema + 索引参数（HNSW / COSINE）
# ===========================================================
# 快速模式适合 demo，真项目都得显式定义 schema 与索引：
#
#   schema 决定：有哪些字段、哪个是主键、向量维度多少、有没有标量字段；
#   index  决定：向量字段用什么算法 + 距离度量 + 算法参数。
#
# 三种常用向量索引（重点）：
#   · FLAT  —— 暴力扫；100% 召回，慢；调试和小数据集用。
#   · IVF_FLAT / IVF_SQ8 —— 先粗聚类（nlist 个桶），查询时只扫 nprobe 个桶；
#                            适合百万级；nlist↑ 召回↓ 速度↑、nprobe↑ 召回↑ 速度↓。
#   · HNSW  —— 分层小世界图；构建慢、查询快、召回高；适合"读多写少"。
#              关键参数：M (每节点邻居数)、efConstruction (建图时探索宽度)、
#              ef (查询时探索宽度，越大越准越慢)。
#
# 三种常用距离度量：
#   · COSINE  —— 越大越像（内积过 normalize）。文本 embedding 几乎都用这个。
#   · IP      —— 内积；如果向量已经 L2 normalize，IP == COSINE。
#   · L2      —— 欧氏距离；越小越像（注意 distance 解释方向相反）。
# -----------------------------------------------------------
def demo_2_explicit_schema():
    client = get_client("02_schema.db")
    name = "demo2"
    if client.has_collection(name):
        client.drop_collection(name)

    # 1) 用 client.create_schema() 拿一个空 schema，再加字段
    schema = client.create_schema(
        auto_id=False,            # 主键自己给
        enable_dynamic_field=True,  # 允许插入 schema 没声明的字段（像个"额外 JSON")
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=8)
    schema.add_field("category", DataType.VARCHAR, max_length=32)

    # 2) 索引参数：必须先 prepare，再在 create_collection 时一起传
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",
        # index_type="HNSW",  Milvus Lite 只能使用HNSW

        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    # 标量字段也可以建索引（加速过滤）：
    index_params.add_index(field_name="category")

    # 3) 一次性建 collection + 索引
    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )

    # 4) 插数据
    rng = random.Random(7)
    cats = ["code", "doc", "code", "doc", "code"]
    data = [
        {"id": i, "vector": [rng.random() for _ in range(8)], "category": cats[i]}
        for i in range(5)
    ]
    client.insert(collection_name=name, data=data)

    # 5) 看一眼 collection 的元信息
    desc = client.describe_collection(name)
    print("=== Demo 2: 显式 Schema + HNSW 索引 ===")
    print(f"collection: {desc['collection_name']}")
    print(f"fields    : {[f['name'] + ':' + f['type'].name for f in desc['fields']]}")
    print(f"row count : {client.get_collection_stats(name)['row_count']}")
    print()


# ===========================================================
# Demo 3 — Top-K 向量搜索（看家本领）
# ===========================================================
# search() 关键参数：
#   · data         —— query 向量（list of vectors，可批量）
#   · limit (top_k) —— 返回前 K 个最相似
#   · search_params —— 算法运行时参数；HNSW 常调 ef，IVF 常调 nprobe
#   · output_fields —— 返回结果里要带哪些标量字段
#
# distance 的含义跟 metric_type 绑定：
#   · COSINE / IP : distance 越大越像（已经是相似度，不是距离）
#   · L2          : distance 越小越像
# 不要看到字段名叫 distance 就一律觉得"越小越好"，这是新手最常踩的坑。
# -----------------------------------------------------------
def demo_3_search():
    client = get_client("03_search.db")
    name = "demo3"
    if client.has_collection(name):
        client.drop_collection(name)
    client.create_collection(collection_name=name, dimension=8, metric_type="COSINE")

    rng = random.Random(0)
    data = [
        {"id": i, "vector": [rng.random() for _ in range(8)], "label": chr(ord("A") + i)}
        for i in range(20)
    ]
    client.insert(collection_name=name, data=data)

    # 用其中第 0 条的 vector 当 query：理论上自己跟自己最像，distance 接近 1.0
    query_vec = data[0]["vector"]

    results = client.search(
        collection_name=name,
        data=[query_vec],
        limit=5,
        search_params={"params": {"ef": 64}},   # HNSW 查询参数
        output_fields=["label"],
    )

    print("=== Demo 3: Top-K 向量搜索 ===")
    print("metric=COSINE → distance 越接近 1 越像；自己应当排第一：")
    for h in results[0]:
        print(f"  id={h['id']}  distance={h['distance']:.4f}  label={h['entity']['label']}")
    print()


# ===========================================================
# Demo 4 — 标量字段过滤（boolean expression）
# ===========================================================
# 真实场景几乎一定要"先过滤再查 top-k"：比如
#   · 只在 category=="code" 的向量里找最相似
#   · 只查 created_at > 2025-01-01 的
#
# Milvus 的 filter 是字符串 boolean expression，类似 SQL where：
#   "category == 'code'"
#   "category in ['code', 'doc']"
#   "rating >= 4 and category != 'archived'"
#   "id > 100"
# 标量字段如果建了索引，过滤会快很多（见 Demo 2 里的 add_index("category")）。
# -----------------------------------------------------------
def demo_4_filter():
    client = get_client("04_filter.db")
    name = "demo4"
    if client.has_collection(name):
        client.drop_collection(name)

    # 这次让 schema 带个 category 字段
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=8)
    schema.add_field("category", DataType.VARCHAR, max_length=32)
    schema.add_field("rating", DataType.INT64)

    idx = client.prepare_index_params()
    idx.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    idx.add_index(field_name="category")          # 加速 category 过滤

    client.create_collection(name, schema=schema, index_params=idx)

    rng = random.Random(1)
    cats = ["code", "doc", "code", "doc", "code", "doc"]
    data = [
        {"id": i, "vector": [rng.random() for _ in range(8)],
         "category": cats[i], "rating": (i % 5) + 1}
        for i in range(6)
    ]
    client.insert(name, data)

    query_vec = [rng.random() for _ in range(8)]

    # 1) 只在 code 类里找
    r1 = client.search(name, data=[query_vec], limit=3,
                       filter="category == 'code'",
                       output_fields=["category", "rating"])
    # 2) 在 rating>=3 的所有类里找
    r2 = client.search(name, data=[query_vec], limit=3,
                       filter="rating >= 3",
                       output_fields=["category", "rating"])

    print("=== Demo 4: 标量字段过滤 ===")
    print("filter: category == 'code'")
    for h in r1[0]:
        print(f"  id={h['id']}  cat={h['entity']['category']}  rating={h['entity']['rating']}  d={h['distance']:.3f}")
    print("filter: rating >= 3")
    for h in r2[0]:
        print(f"  id={h['id']}  cat={h['entity']['category']}  rating={h['entity']['rating']}  d={h['distance']:.3f}")
    print()


# ===========================================================
# Demo 5 — 端到端语义检索（真·embedding，不是随机数）
# ===========================================================
# 前面 4 个 demo 用 random 向量是为了把 API 讲清楚；真正能用的检索系统一定是：
#   text  ──[Embedding 模型]──▶  List[float]  ──[Milvus]──▶  最近邻 text
#
# 这里用 sentence-transformers 跑一个最小的中英文 embedding 模型：
#   "BAAI/bge-small-zh-v1.5"  ——  512 维, ~100MB, CPU 也能跑
# CodeLens 用的也是这个家族（默认 bge-small-zh-v1.5）。
#
# 注意：模型第一次会从 HuggingFace 下载，慢的话提前 export HF_ENDPOINT 用镜像。
# -----------------------------------------------------------
def demo_5_semantic():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("=== Demo 5: 端到端语义检索 ===")
        print("(skip: 需要 `pip install sentence-transformers`)")
        print()
        return

    client = get_client("05_semantic.db")
    name = "demo5"

    print("=== Demo 5: 端到端语义检索 ===")
    print("加载 embedding 模型（首次会下载几百 MB，请耐心）...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    dim = model.get_sentence_embedding_dimension()  # 512
    print(f"embedding 维度: {dim}")

    if client.has_collection(name):
        client.drop_collection(name)
    client.create_collection(collection_name=name, dimension=dim, metric_type="COSINE")

    # 一个迷你"知识库"
    docs = [
        "Milvus 是一个开源的向量数据库，支持十亿级向量检索。",
        "HNSW 是一种基于图的近似最近邻搜索算法，构建慢但查询快。",
        "Python 的 list 和 tuple 区别在于前者可变后者不可变。",
        "Transformer 架构通过自注意力机制处理序列，是 BERT/GPT 的基础。",
        "ReAct 是一种让大模型边推理边调用工具的提示范式。",
        "向量的余弦相似度衡量两个向量方向是否一致，常用于语义匹配。",
        "Docker 容器与虚拟机相比启动更快、资源占用更低。",
    ]

    # 1) 把所有文本批量转成 embedding（normalize_embeddings=True 是 bge 的最佳实践）
    vectors = model.encode(docs, normalize_embeddings=True).tolist()

    # 2) 写入
    client.insert(
        collection_name=name,
        data=[{"id": i, "vector": v, "text": t}
              for i, (v, t) in enumerate(zip(vectors, docs))],
    )

    # 3) 拿用户问题去查
    queries = ["什么数据库适合做语义搜索？", "怎么让 LLM 自己调工具？"]
    q_vecs = model.encode(queries, normalize_embeddings=True).tolist()

    results = client.search(
        collection_name=name, data=q_vecs, limit=2,
        output_fields=["text"],
    )

    for q, hits in zip(queries, results):
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  [{h['distance']:.3f}] {h['entity']['text']}")
    print()


# ===========================================================
# Demo 6 — Upsert / Delete / Get by id
# ===========================================================
# 一个长期运行的 RAG 系统，知识库一定会变：
#   · 新文档进来       → insert
#   · 旧文档内容更新   → upsert (主键存在则覆盖、不存在则插入)
#   · 文档下线         → delete
#   · 调试某个文档     → get(id) 直接看
# -----------------------------------------------------------
def demo_6_crud():
    client = get_client("06_crud.db")
    name = "demo6"
    if client.has_collection(name):
        client.drop_collection(name)
    client.create_collection(collection_name=name, dimension=4)

    rng = random.Random(0)
    init = [{"id": i, "vector": [rng.random() for _ in range(4)], "title": f"v0-{i}"}
            for i in range(3)]
    client.insert(name, init)

    print("=== Demo 6: Upsert / Delete / Get ===")
    print("初始 3 条：")
    for r in client.get(name, ids=[0, 1, 2]):
        print(f"  id={r['id']}  title={r['title']}")

    # upsert：把 id=1 的 title 改掉，再插一条 id=99 的新数据
    client.upsert(name, [
        {"id": 1, "vector": [rng.random() for _ in range(4)], "title": "v1-updated"},
        {"id": 99, "vector": [rng.random() for _ in range(4)], "title": "brand-new"},
    ])
    print("\nupsert(id=1 改名, id=99 新增) 之后：")
    for r in client.get(name, ids=[0, 1, 2, 99]):
        print(f"  id={r['id']}  title={r['title']}")

    # delete：按主键 / 也支持按 filter 表达式
    client.delete(name, ids=[0])
    print("\ndelete(id=0) 之后：")
    for r in client.get(name, ids=[0, 1, 2, 99]):
        print(f"  id={r['id']}  title={r['title']}")
    # 不存在的 id 在 get 结果里就不会出现（不会报错）
    print()


# ===========================================================
# Demo 7 — 持久化与重连
# ===========================================================
# Milvus Lite 把整个 collection 存成一个 .db 文件（其实是 SQLite + 一些自管文件）。
# 进程重启、机器迁移时把这个 .db 拷走就行；连接时 uri 指向同一个文件即可。
#
# 这个 demo 故意分两段：
#   · 先建 client A，写入数据，close
#   · 再建 client B 连同一个 .db，能查到刚才写入的数据
# -----------------------------------------------------------
def demo_7_persistence():
    db_file = STORAGE_DIR / "07_persist.db"
    if db_file.exists():
        db_file.unlink()              # 干净起步

    print("=== Demo 7: 持久化与重连 ===")

    # ---- 第一次连接：写入 ----
    client_a = MilvusClient(uri=str(db_file))
    name = "demo7"
    client_a.create_collection(collection_name=name, dimension=4)
    rng = random.Random(123)
    client_a.insert(name, [
        {"id": i, "vector": [rng.random() for _ in range(4)], "tag": f"t{i}"}
        for i in range(5)
    ])
    print(f"client A 写入 5 条；文件大小 {db_file.stat().st_size} bytes")
    client_a.close()

    # ---- 模拟"重启"：新建一个 client 连同一个文件 ----
    client_b = MilvusClient(uri=str(db_file))
    rows = client_b.query(
        collection_name=name,
        filter="id >= 0",       # query 不走向量、按表达式直接拉行
        output_fields=["id", "tag"],
        limit=10,
    )
    print(f"client B 重连后读到 {len(rows)} 条：")
    for r in rows:
        print(f"  id={r['id']}  tag={r['tag']}")
    client_b.close()
    print()


# ===========================================================
# Demo 8 — Milvus + LangChain：CodeLens 阶段 5 同款用法
# ===========================================================
# 上面 7 个 demo 让你能直接拿 pymilvus API 写检索；
# 但在 RAG 项目里通常会再裹一层 langchain：
#   · MilvusVectorStore.from_documents(docs, embedding, ...) 一行入库
#   · vector_store.as_retriever().invoke(query) 一行检索
#
# 这正是 CodeLens 阶段 5 的写法（codelens/app/vectorstore.py）。
# 关系是「LangChain 提供 Document 抽象 + Retriever 接口；底下还是 pymilvus」。
# -----------------------------------------------------------
def demo_8_langchain_milvus():
    try:
        from langchain_milvus import Milvus
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("=== Demo 8: LangChain × Milvus ===")
        print("(skip: 需要 `pip install langchain-milvus langchain-huggingface sentence-transformers`)")
        print()
        return

    print("=== Demo 8: LangChain × Milvus ===")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    docs = [
        Document(page_content="Milvus 支持 HNSW、IVF、FLAT 多种索引类型。",
                 metadata={"source": "milvus_doc"}),
        Document(page_content="LangChain 提供统一的 Retriever 抽象。",
                 metadata={"source": "langchain_doc"}),
        Document(page_content="ReAct 让模型在思考与行动间循环。",
                 metadata={"source": "agent_doc"}),
        Document(page_content="向量的余弦相似度常用于语义匹配。",
                 metadata={"source": "math_note"}),
    ]

    # 注意 connection_args 的 uri 指向 Lite 文件路径
    db_file = STORAGE_DIR / "08_langchain.db"
    if db_file.exists():
        db_file.unlink()

    # ⚠️ langchain-milvus 0.3.x + Milvus Lite 的兼容坑：
    # `MilvusClient(uri="*.db")` 会创建一个内部 alias（如 "cm-xxx"），但**不会**
    # 注册到全局 `pymilvus.connections`。而 langchain_milvus 内部又用这个 alias
    # 去 ORM `Collection(name, using=alias)` 读 schema —— 必然抛
    # `ConnectionNotExistException("should create connection first.")`。
    #
    # workaround：分两步——先构造 Milvus 实例，再把它内部 alias 桥接到
    # 全局 connections，最后调 add_documents。等 langchain-milvus 在
    # _extract_fields 里完全切到 MilvusClient 路径后这段就能删掉。
    from pymilvus import connections

    vs = Milvus(
        embedding_function=embeddings,
        collection_name="demo8",
        connection_args={"uri": str(db_file)},
        index_params={"index_type": "FLAT", "metric_type": "COSINE"},
        auto_id=True,
        drop_old=True,
    )
    # 把 MilvusClient 内部 alias 也注册到全局，让后续 ORM Collection 能找到
    bridge_alias = vs._milvus_client._using
    if bridge_alias not in connections.list_connections():
        connections.connect(alias=bridge_alias, uri=str(db_file))
    vs.add_documents(docs)

    retriever = vs.as_retriever(search_kwargs={"k": 2})

    for q in ["哪种索引适合读多写少？", "怎么让 Agent 思考?"]:
        print(f"\nQ: {q}")
        for d in retriever.invoke(q):
            print(f"  - [{d.metadata['source']}] {d.page_content}")
    print()


# ===========================================================
# 把 Demo 对应到 CodeLens 阶段
# ===========================================================
# · Demo 1~3  ⇒  理解 Milvus 的"建表 / 插入 / top-k"基础三件套
#                ；阶段 1 的 build_index 本质就是这三步循环跑全部 chunk。
# · Demo 4    ⇒  CodeLens 里 metadata.type == "code" / "doc" 这种过滤的底层。
# · Demo 5    ⇒  把 sentence-transformers 接 Milvus，正是 CodeLens
#                 app/embeddings.py + app/ingest/build_index.py 在做的事。
# · Demo 6    ⇒  增量重建索引（项目里加新文档而不重建全量）的正确姿势。
# · Demo 7    ⇒  CodeLens 重启后能读回旧索引就是靠这个机制。
# · Demo 8    ⇒  CodeLens 阶段 5 (codelens/app/vectorstore.py) 的封装方式
#                 ；LangChain 包了一层但底层还是这些 API。
#
# 看完 8 个 demo，再去翻 codelens/app/ 下任何一处用到 Milvus 的代码都不会怕。
# -----------------------------------------------------------

DEMOS = {
    # "1": demo_1_quick_start,
    # "2": demo_2_explicit_schema,
    # "3": demo_3_search,
    # "4": demo_4_filter,
    "5": demo_5_semantic,
    # "6": demo_6_crud,
    # "7": demo_7_persistence,
    # "8": demo_8_langchain_milvus,
}


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which is None:
        for key in sorted(DEMOS):
            DEMOS[key]()
    else:
        DEMOS[which]()
