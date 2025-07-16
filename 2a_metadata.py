import os
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

from pinecone import ServerlessSpec
index_name = "new-index"  # change if desired
dimension = 1536
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print("Vector store already exists. No need to initialize.")

index = pc.Index(index_name)

    # Create embeddings

print("\n--- Creating embeddings ---")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
print("\n--- Finished creating and persisting vector store ---")

# Setup directories
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "myBook")

print(f"Books directory: {books_dir}")
    # Ensure the books directory exists
if not os.path.exists(books_dir):
    raise FileNotFoundError(f"The directory {books_dir} does not exist. Please check the path.")

print(f"Books directory: {books_dir}")

    # List all text files in the directory

for file in os.listdir(books_dir):
    if file.endswith(".txt"):
        print(f"\n Processing file: {file}")
        loader = TextLoader(os.path.join(books_dir, file), encoding="utf-8")
        docs = loader.load()
        
        for doc in docs:
            doc.metadata = {"source": file}

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(docs)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(chunks)}")
    for chunk in chunks:
            try:
                chunk_id = str(uuid4())
                vector_store.add_documents(documents=[chunk], ids=[chunk_id])
                print(f"✅ Uploaded chunk: {chunk_id}")
            except Exception as e:
                print(f"❌ Failed to upload chunk: {e}")

else:
    print("Vector store already exists. No need to initialize.")