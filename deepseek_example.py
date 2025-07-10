from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get the API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg.content


