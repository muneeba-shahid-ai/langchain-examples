import os
from uuid import uuid4
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Index setup
index_name = "books-index"
dimension = 1536

if not pc.has_index(index_name):
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "langchain_book")

if not os.path.exists(books_dir):
    raise FileNotFoundError(f"Directory does not exist: {books_dir}")

print(f"Books directory: {books_dir}")

# Embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index, embedding_model, "text")


# Load, split, embed, and upload each document one-by-one
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

for file in os.listdir(books_dir):
    if file.endswith(".txt"):
        print(f"Processing file: {file}")
        loader = TextLoader(os.path.join(books_dir, file), encoding="utf-8")
        docs = loader.load()
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            try:
                chunk_id = str(uuid4())
                vector_store.add_documents(documents=[chunk], ids=[chunk_id])
                print(f"✅ Uploaded chunk: {chunk_id}")
            except Exception as e:
                print(f"❌ Failed to upload chunk: {e}")
