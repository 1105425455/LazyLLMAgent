import lazyllm
import time
from lazyllm import (
    fc_register, Document, Retriever,
    OnlineEmbeddingModule, OnlineChatModule, WebModule,
    ReactAgent
)
from get_api import APIKEY
# 模型与数据
embed_model = OnlineEmbeddingModule(source='glm', embed_model_name='embedding-2', api_key=APIKEY)

# 与本地持久化配置对齐（kb_store_optimized/）
store_conf = {
    "vector_store": {
        "type": "ChromadbStore",
        "kwargs": {"dir": "kb_store_optimized"}
    },
    "segment_store": {
        "type": "MapStore",
        "kwargs": {"uri": "kb_store_optimized/segments.db"}
    }
}

doc = Document(dataset_path="data", embed=embed_model, store_conf=store_conf)


# 多粒度召回：中粒度 + 细粒度
retriever_med = Retriever(doc, group_name=Document.MediumChunk, similarity="cosine", topk=8)
retriever_fine = Retriever(doc, group_name=Document.FineChunk, similarity="cosine", topk=12)


def _fuse_nodes(nodes_a, nodes_b, limit=10):
    # 简单去重融合：按分数降序，基于 uid 去重
    all_nodes = []
    seen = set()
    for n in list(nodes_a) + list(nodes_b):
        uid = getattr(n, "_uid", None) or getattr(n, "uid", None)
        if uid in seen:
            continue
        seen.add(uid)
        all_nodes.append(n)
    # 如果节点自带 score，按 score 排序；否则保持顺序
    try:
        all_nodes.sort(key=lambda x: getattr(x, "_score", None) or getattr(x, "score", 0), reverse=True)
    except Exception:
        pass
    return all_nodes[:limit]


# 注册优化版 RAG 工具
@fc_register("tool")
def search_knowledge_base_optimized(query: str):
    """
    使用多粒度检索（中粒度+细粒度）并融合结果，从知识库中返回相关内容。

    Args:
        query (str): 用户的问题或检索词，用于在知识库中查找相关片段。
    """
    nodes_med = retriever_med(query=query)
    nodes_fine = retriever_fine(query=query)
    fused = _fuse_nodes(nodes_med, nodes_fine, limit=10)
    context_str = "".join([node.get_content() for node in fused])
    return context_str


# prompt 设计
prompt = 'You will act as an AI question-answering assistant and complete a dialogue task. In this task, you need to provide your answers based on the given context and questions. You can use the search_knowledge_base_optimized tool to find relevant information from the knowledge base.'

# 创建 ReactAgent（沿用 GLM 聊天）
# 终端耗时打印：封装一个带计时的 LLM 包装器
class TimedLLM:
    def __init__(self, inner):
        self._inner = inner

    def prompt(self, *args, **kwargs):
        return self._inner.prompt(*args, **kwargs)

    def share(self, *args, **kwargs):
        self._inner = self._inner.share(*args, **kwargs)
        return self

    def used_by(self, *args, **kwargs):
        self._inner = self._inner.used_by(*args, **kwargs)
        return self

    def __call__(self, *args, **kwargs):
        t0 = time.time()
        res = self._inner(*args, **kwargs)
        t1 = time.time()
        print(f"[Timing] answer: {t1 - t0:.2f}s")
        return res

    def __getattr__(self, name):
        return getattr(self._inner, name)


timed_llm = TimedLLM(OnlineChatModule(source='glm', model='glm-4.5', api_key=APIKEY, stream=False))

agent = ReactAgent(
    llm=timed_llm,
    tools=['search_knowledge_base_optimized'],
    prompt=prompt,
    stream=False
)

# Web 服务启动
w = WebModule(agent, stream=False)
w.start().wait()


