import os
import time
from pathlib import Path

import lazyllm
from lazyllm import Document, Retriever, OnlineEmbeddingModule
from get_api import APIKEY


def main():
    # 确保持久化目录存在（向量库与分片库会落在这里）
    # 注意：将持久化目录放在数据集目录之外，避免被文档读取器误当作原始文件解析
    store_dir = Path("kb_store")
    store_dir.mkdir(parents=True, exist_ok=True)

    embed_model = OnlineEmbeddingModule(source='glm', embed_model_name='embedding-2', api_key=APIKEY)

    # 配置持久化存储：
    # - 向量库：Chroma -> data/rag_store/chroma.sqlite3
    # - 分片内容与元数据：SQLite -> data/rag_store/segments.db
    store_conf = {
        "vector_store": {
            "type": "ChromadbStore",
            "kwargs": {"dir": str(store_dir)}
        },
        "segment_store": {
            "type": "MapStore",
            "kwargs": {"uri": str(store_dir / "segments.db")}
        }
    }

    t0 = time.time()
    # 第一次运行会解析/切分/入库；之后将直接复用本地索引，显著加速
    doc = Document(dataset_path="data", embed=embed_model, store_conf=store_conf)
    retriever = Retriever(doc, group_name=Document.CoarseChunk, similarity="cosine", topk=3)
    t1 = time.time()
    print(f"Document init (with persistent store) time: {t1 - t0:.2f}s")

    # 进行一次简单查询验证读取是否正常
    llm = lazyllm.OnlineChatModule(source='glm', model='glm-4.5', api_key=APIKEY)
    prompt = 'You are an assistant. Answer using provided context.'
    llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

    query = "介绍一下小龙女"
    nodes = retriever(query=query)
    ctx = "".join([n.get_content() for n in nodes])
    t2 = time.time()
    print(f"Retrieve time: {t2 - t1:.2f}s, nodes={len(nodes)}")

    res = llm({"query": query, "context_str": ctx})
    t3 = time.time()
    print(f"Answer: {res}")
    print(f"LLM time: {t3 - t2:.2f}s")


if __name__ == "__main__":
    main()


