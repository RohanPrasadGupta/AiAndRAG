# from chromadb.api.types import Document
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import shutil
import os

if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("Cleared existing ChromaDB")
else:
    print("No existing ChromaDB found")


with open("documents.txt", "r", encoding="utf-8") as file:
    text = file.read()

docs = [Document(page_content=text, metadata={"source": "documents.txt"})]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    )

chunks = splitter.split_documents(docs)


for chunk in chunks:
    content = chunk.page_content.lower()
    
    if "deep learning" in content or "neural" in content or "machine learning" in content:
        chunk.metadata["category"] = "ml"
    elif "python" in content or "programming" in content:
        chunk.metadata["category"] = "programming"
    elif "cloud" in content or "aws" in content:
        chunk.metadata["category"] = "cloud"
    elif "data" in content or "analysis" in content:
        chunk.metadata["category"] = "data"
    else:
        chunk.metadata["category"] = "fundamentals"
    
    # Print so you can see what got assigned
    print(f"Category: {chunk.metadata['category']} | {chunk.page_content[:60]}...")

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
test_vector = embedding_model.embed_query("What is Artificial Intelligence?")
print(f"Embedding dimension: {len(test_vector)}")
print(f"First 5 numbers of vector: {test_vector[:5]}")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db",
)

# See what metadata is stored
all_data = vectorstore.get(include=["metadatas"])
for i, meta in enumerate(all_data["metadatas"]):
    print(f"Chunk {i}: {meta}")


count = vectorstore._collection.count()
print(f"Total vectors stored in ChromaDB: {count}")
# print("ChromaDB saved to ./chroma_db folder")

print("\n--- Quick similarity search test ---")
query = "What programming language is used for AI?"
results = vectorstore.similarity_search(
    query, 
    k=2,
    filter= {"category": "programming"}
    ) # k stands for number of results to return

print(f"Query: '{query}'")
print(f"Top 2 most relevant chunks:\n")
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(result.page_content)
    print("---")


# Test search for other categories
print("\n--- Filter: ml ---")
results_ml = vectorstore.similarity_search(
    query,
    k=2,
    filter={"category": "ml"}
)
for i, result in enumerate(results_ml):
    print(f"Result {i+1}: {result.page_content[:100]}...")
 
print("\n--- Filter: cloud ---")
results_cloud = vectorstore.similarity_search(
    query,
    k=2,
    filter={"category": "cloud"}
)
for i, result in enumerate(results_cloud):
    print(f"Result {i+1}: {result.page_content[:100]}...")

print("\n--- Filter: elephant ---")
results_cloud = vectorstore.similarity_search(
    query,
    k=2,
    filter={"category": "elephant"}
)
for i, result in enumerate(results_cloud):
    print(f"Result {i+1}: {result.page_content[:100]}...")