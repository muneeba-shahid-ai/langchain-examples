from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)
chat_history = []
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message) 

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query)) 
    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message

    print(f'AI: {response}')


print("---- Message History ----")
print(chat_history)