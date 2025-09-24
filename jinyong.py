import lazyllm
import time
from lazyllm import (
    fc_register, Document, Retriever, 
    OnlineEmbeddingModule, OnlineChatModule, WebModule,
    ReactAgent
)
from get_api import APIKEY
embed_model = OnlineEmbeddingModule(source='glm', embed_model_name='embedding-2', api_key='8df34f9ec1894e5f9fdf863c9f1aef6a.aSipXmwYch69ymaD')

# 与本地持久化配置对齐（kb_store/）
store_conf = {
    "vector_store": {
        "type": "ChromadbStore",
        "kwargs": {"dir": "kb_store"}
    },
    "segment_store": {
        "type": "MapStore",
        "kwargs": {"uri": "kb_store/segments.db"}
    }
}

doc = Document(dataset_path="data", embed=embed_model, store_conf=store_conf)
retriever = Retriever(doc, group_name='CoarseChunk', similarity="cosine", topk=3)

# 注册RAG工具
@fc_register("tool")
def search_knowledge_base(query: str):
    """
    搜索知识库并返回相关文档内容
    
    Args:
        query (str): 搜索查询字符串
    """
    # 将Retriever组件召回的节点全部存储到列表doc_node_list中
    doc_node_list = retriever(query=query)
    # 将召回节点中的内容组合成字符串
    context_str = "".join([node.get_content() for node in doc_node_list])
    return context_str

# prompt 设计
prompt = 'You will act as an AI question-answering assistant and complete a dialogue task. In this task, you need to provide your answers based on the given context and questions. You can use the search_knowledge_base tool to find relevant information from the knowledge base.'

# 创建ReactAgent
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
        # Delegate other attributes/methods to inner module
        return getattr(self._inner, name)


timed_llm = TimedLLM(OnlineChatModule(source='glm', model='glm-4.5', api_key=APIKEY, stream=False))

agent = ReactAgent(
    llm=timed_llm,
    tools=['search_knowledge_base'],
    prompt=prompt,
    stream=False
)

# 创建Web模块并启动
w = WebModule(agent, stream=False)
w.start().wait()
