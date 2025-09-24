import time
import threading
import tkinter as tk
from tkinter import ttk

import lazyllm
from lazyllm import Document, Retriever, OnlineEmbeddingModule, OnlineChatModule
from get_api import APIKEY


# 初始化模型与持久化文档（与 jinyong_rag_optimized.py 保持一致）
embed_model = OnlineEmbeddingModule(source='glm', embed_model_name='embedding-2', api_key=APIKEY)

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

retriever_med = Retriever(doc, group_name=Document.MediumChunk, similarity="cosine", topk=8)
retriever_fine = Retriever(doc, group_name=Document.FineChunk, similarity="cosine", topk=12)


def _fuse_nodes(nodes_a, nodes_b, limit=10):
    all_nodes = []
    seen = set()
    for n in list(nodes_a) + list(nodes_b):
        uid = getattr(n, "_uid", None) or getattr(n, "uid", None)
        if uid in seen: continue
        seen.add(uid)
        all_nodes.append(n)
    try:
        all_nodes.sort(key=lambda x: getattr(x, "_score", None) or getattr(x, "score", 0), reverse=True)
    except Exception:
        pass
    return all_nodes[:limit]


chat = OnlineChatModule(source='glm', model='glm-4.5', api_key=APIKEY, stream=False)
base_instruction = 'You will act as an AI question-answering assistant and complete a dialogue task. In this task, you need to provide your answers based on the given context and questions.'
chat.prompt(lazyllm.ChatPrompter(instruction=base_instruction, extra_keys=['context_str']))

rewrite_chat = OnlineChatModule(source='glm', model='glm-4.5', api_key=APIKEY, stream=False)
rewrite_instruction = (
    '你是查询改写助手。已知本系统的知识库主要包含金庸小说（如神雕侠侣）原文片段、人物、情节、武学与设定等内容。'
    '请根据用户的问题，改写出一个更贴近上述知识库、便于检索命中的中文查询，保持语义一致但更具体、更聚焦。'
    '只输出改写后的中文查询，不要输出其他说明。'
)

summarize_chat = OnlineChatModule(source='glm', model='glm-4.5', api_key=APIKEY, stream=False)
summarize_instruction = (
    '你是总结助手。现在有用户的原始问题与其回答，以及一个改写后的问题与其回答。'
    '请依据两份回答中的证据，面向原始问题给出简洁、直接、可信的中文答案；若有引用冲突，以与原文更一致者为准。'
)


def ask_query(query: str) -> str:
    t0 = time.time()
    nodes_med = retriever_med(query=query)
    nodes_fine = retriever_fine(query=query)
    fused = _fuse_nodes(nodes_med, nodes_fine, limit=10)
    ctx = "".join([n.get_content() for n in fused])
    t1 = time.time()
    ans = chat({"query": query, "context_str": ctx})
    t2 = time.time()
    print(f"[Timing] retrieve(med+fine+fuse): {t1 - t0:.2f}s, answer: {t2 - t1:.2f}s, total: {t2 - t0:.2f}s")
    return str(ans)


# Tk 界面
root = tk.Tk()
root.title("金庸小说辅助阅读器 - Tk")
root.geometry("800x600")

frm_top = ttk.Frame(root)
frm_top.pack(fill=tk.X, padx=10, pady=10)

lbl = ttk.Label(frm_top, text="输入问题：")
lbl.pack(side=tk.LEFT)

query_var = tk.StringVar()
entry = ttk.Entry(frm_top, textvariable=query_var, width=80)
entry.pack(side=tk.LEFT, padx=8)

btn = ttk.Button(frm_top, text="提问")
btn.pack(side=tk.LEFT)

txt = tk.Text(root, wrap=tk.WORD)
txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


def on_ask_clicked():
    q = query_var.get().strip()
    if not q:
        return
    txt.delete(1.0, tk.END)
    txt.insert(tk.END, "思考中，请稍候...\n")

    def worker():
        try:
            res = ask_query(q)
            # 直接以字符串形式调用，避免 provider 对参数键的要求
            rewrite_input = (
                f"{rewrite_instruction}\n"
                f"用户问题: {q}\n"
                f"只输出改写后的查询。"
            )
            rewrite_query = str(rewrite_chat(rewrite_input)).strip()

            res_rewrite = ask_query(rewrite_query or q)

            summarize_input = (
                f"{summarize_instruction}\n"
                f"原始问题: {q}\n"
                f"原始回答: {str(res)}\n"
                f"改写问题: {rewrite_query}\n"
                f"改写回答: {str(res_rewrite)}\n"
                f"请基于证据给出最终答案："
            )
            res_summarize = summarize_chat(summarize_input)
            print("res",res)
            print("rewrite_query",rewrite_query)
            print("res_rewrite",res_rewrite)
            print("res_summarize",res_summarize)
        except Exception as e:
            res_summarize = f"发生错误：{e}"
        def update():
            txt.delete(1.0, tk.END)
            txt.insert(tk.END, res_summarize)
        root.after(0, update)

    threading.Thread(target=worker, daemon=True).start()


btn.configure(command=on_ask_clicked)
entry.bind('<Return>', lambda e: on_ask_clicked())

root.mainloop()


