import lazyllm

chat = lazyllm.OnlineChatModule(source='glm',model='glm-4.5',api_key='8df34f9ec1894e5f9fdf863c9f1aef6a.aSipXmwYch69ymaD')
while True:
    query = input("query(enter 'quit' to exit): ")
    if query == "quit":
        break
    res = chat.forward(query)
    print(f"answer: {res}")
