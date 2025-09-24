# 金庸阅读辅助器
目录下新建api_key.env 放入智谱的apikey，因为我用的是智谱的，可以根据需要改动。

1.将需要的小说pdf放入到data下

2.运行jinyong_build_kb_store.py文件进行rag并将数据存储到目录下的kb_store_optimized下

3.运行jinyong_rag_optimized.py即可在线提问关于小说的具体内容。
