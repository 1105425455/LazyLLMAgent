import shutil
from pathlib import Path

import lazyllm
from lazyllm import Document, Retriever, OnlineEmbeddingModule
from get_api import APIKEY


def main():
    # 重建与 jinyong_rag_optimized.py 对应的本地库（kb_store），包含 Medium/Fine 分组
    store_dir = Path("kb_store_optimized")
    if store_dir.exists():
        print(f"[Reset] Removing store dir: {store_dir}")
        shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)

    embed_model = OnlineEmbeddingModule(source='glm', embed_model_name='embedding-2', api_key=APIKEY)

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

    # 初始化 Document 并按 Medium/Fine 两种粒度建立索引
    doc = Document(dataset_path="data", embed=embed_model, store_conf=store_conf)
    retriever_med = Retriever(doc, group_name=Document.MediumChunk, similarity="cosine", topk=3)
    retriever_fine = Retriever(doc, group_name=Document.FineChunk, similarity="cosine", topk=3)

    # 触发一次检索以确保集合创建完成
    _ = retriever_med(query="初始化索引")
    _ = retriever_fine(query="初始化索引")

    print("[Build] kb_store initialized with MediumChunk and FineChunk indices.")


if __name__ == "__main__":
    main()


