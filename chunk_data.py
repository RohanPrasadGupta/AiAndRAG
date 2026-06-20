# from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

with open("documents.txt", "r", encoding="utf-8") as file:
    text = file.read()

docs = [Document(page_content=text,metadata={"source": "documents.txt"})]

splitter =  RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    length_function = len,
)

chunks = splitter.split_documents(docs)

print(f"Total chunks: {len(chunks)} | chunk_size: 500 | overlap: 50")
# print("All Chunks :=>",chunks)
print(f"\n---List of Chunks ---")
for chunk in chunks:
    print(chunk.page_content)
    print("--------------------------------")





# =========== Old Code ====================

# loader = TextLoader("documents.txt")
# docs = loader.load()

# print("=================Before Chunking===============================")
# print(f"Number of documnets loaded",{len(docs)})
# print(f"Number of first itmm: ",{len(docs)})
# print(f"Total characters in document: {len(docs[0].page_content)}")
# # print(f"\n--- First 200 characters ---")
# # print(docs[0].page_content[:200])

# splitter =  RecursiveCharacterTextSplitter(
#     chunk_size = 500,
#     chunk_overlap = 50,
#     length_function = len,
# )

# chunks = splitter.split_documents(docs)

# print("=================After Chunking===============================")
# # print(f"\n\nTotal chunks created: {chunks}")
# print(f"\n\nTotal chunks created: {len(chunks)}")
# # print(f"\n--- Chunk 1 ---")
# # print(chunks[0].page_content)

# print(f"\n---List of Chunks ---")
# for chunk in chunks:
#     print(chunk.page_content)
#     print("--------------------------------")

