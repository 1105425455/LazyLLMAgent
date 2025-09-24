import lazyllm
from lazyllm import Retriever, Document, OnlineEmbeddingModule
from get_api import APIKEY
import os
import time
embed_model = OnlineEmbeddingModule(source='glm', embed_model_name='embedding-2', api_key=APIKEY)
start_time = time.time()
doc = Document(dataset_path="data", embed=embed_model)
retriever = Retriever(doc, group_name='CoarseChunk', similarity="cosine", topk=3)
end_time = time.time()
print(f'Document time: {end_time - start_time} seconds')
llm = lazyllm.OnlineChatModule(source='glm',model='glm-4.5',api_key=APIKEY)
print(llm)
# prompt 设计
prompt = 'You will act as an AI question-answering assistant and complete a dialogue task. In this task, you need to provide your answers based on the given context and questions.'
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))
start_time = time.time()
query = "介绍一下杨过"
# 将Retriever组件召回的节点全部存储到列表doc_node_list中
doc_node_list = retriever(query=query)
# 将query和召回节点中的内容组成dict，作为大模型的输入
res = llm({"query": query, "context_str": "".join([node.get_content() for node in doc_node_list])})
end_time = time.time()
print(f'With RAG Answer: {res}')
print(f'With RAG Answer time: {end_time - start_time} seconds')