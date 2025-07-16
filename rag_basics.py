import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore  # Updated import
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

# Set Pinecone configuration
index_name = "my-first-db"
pinecone_region = "us-east-1"  # or your chosen region

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # 384 for MiniLM
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_region)
    )

# Use Hugging Face free embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vectorstore - CORRECTED APPROACH
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_model,  # Pass embedding object directly
    text_key="text"  # Metadata field containing text
)

# Define the query
query = "Who is Odysseus' wife?"

# Retrieve relevant documents
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.9}
)
relevant_docs = retriever.invoke(query)

# Display the results
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")