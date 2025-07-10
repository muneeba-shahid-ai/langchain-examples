from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "gemma3:4b",
    temperature = 0.5,
    num_predict = 100,
    # other params ...
)
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
response =llm.invoke(messages)
print(response)
