# -*- coding: utf-8 -*-
"""
检索层冒烟测试（阶段 8.1 验收脚本）。

不是 pytest 单元测试——是一个"跑一遍打印结果让人肉眼对一遍"的脚本，因为：
  · Router 选哪个分支本身就是 LLM 的判断，没有确定的正确答案
  · 检索召回的"对错"也是模糊的，需要人看片段是否相关

所以这文件是一份**对照表**：3 类问题（纯文档 / 纯代码 / 跨类型对比）各 5 题，
跑一遍打印 router 选了哪个分支 + 拿回了哪些片段，肉眼确认：
  ① 文档题被路由到 doc_only，代码题被路由到 code_only，对比题被路由到
     multi_aspect（至少大致符合）
  ② 召回的 source 路径在合理范围（文档题来自 data/docs/，代码题来自 data/code/）

运行：
    python tests/test_retrieval.py
"""

import sys
from pathlib import Path

# 把 codelens/（本脚本父目录的父目录）加到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 三类基准问题。每类 5 题；后面跑评测（阶段 8.2 / 8.5）时也用同一份基准。
DOC_ONLY_QUESTIONS = [
    "cpp-httplib 的 README 介绍这是个什么样的库？",
    "cpp-httplib 文档里说怎么启用 SSL？",
    "Server 类有哪些主要的配置选项？文档里是怎么列的？",
    "cpp-httplib 文档里推荐的 thread pool 大小是怎么决定的？",
    "cpp-httplib 在文档里如何说明它的依赖关系？",
]

CODE_ONLY_QUESTIONS = [
    "ThreadPool 类的 enqueue 方法是怎么实现的？",
    "Request 结构体里都有哪些字段？",
    "httplib.h 里 detail::handle_EINTR 函数做了什么？",
    "Server::listen 内部是怎么 accept 连接的？",
    "Stream 类的 read 在什么情况下返回 0？",
]

MULTI_ASPECT_QUESTIONS = [
    "ThreadPool 在文档里的使用建议和源码实现是否一致？",
    "对比一下 Server 和 Client 类在错误处理上的差异。",
    "Request 和 Response 两个结构体在字段设计上有什么对称性？",
    "SSL 相关功能在文档里是怎么介绍的、源码里又是怎么实现的？",
    "Stream 与 SocketStream 在读写超时处理上有什么区别？",
]


def _truncate(s: str, n: int = 200) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")


def _run_one(label: str, question: str, router):
    """跑一题，打印 router 决策 + 合成答案前若干字符 + 引用片段 source。"""
    print(f"\n[{label}] Q: {question}")
    response = router.query(question)
    body = str(response).strip()
    print(f"  A: {_truncate(body, 240)}")
    nodes = getattr(response, "source_nodes", []) or []
    if nodes:
        print(f"  refs ({len(nodes)}):")
        for n in nodes[:5]:
            src = (n.node.metadata or {}).get("source", "(unknown)")
            t = (n.node.metadata or {}).get("type", "?")
            print(f"    - [{t}] {src}")
    else:
        print("  refs: (none — router 可能选了只走 LLM 的分支？)")


def main():
    from app.retrieval.router import get_router
    router = get_router()
    print("=" * 60)
    print("CodeLens 检索层冒烟测试 (阶段 8.1)")
    print("=" * 60)

    for q in DOC_ONLY_QUESTIONS:
        _run_one("DOC", q, router)
    for q in CODE_ONLY_QUESTIONS:
        _run_one("CODE", q, router)
    for q in MULTI_ASPECT_QUESTIONS:
        _run_one("MULTI", q, router)

    print("\n" + "=" * 60)
    print("Done. 人肉对照表：")
    print("  [DOC]   应路由到 doc_only，refs 来自 data/docs/")
    print("  [CODE]  应路由到 code_only，refs 来自 data/code/")
    print("  [MULTI] 应路由到 multi_aspect，refs 来自 doc + code 混合")
    print("=" * 60)


if __name__ == "__main__":
    main()
