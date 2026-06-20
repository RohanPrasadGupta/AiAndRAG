from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(embedding_function=embedding_model, persist_directory="./chroma_db")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate(
template="""You are a helpful assistant. 
Answer the question below using ONLY the context provided.
If the answer is not in the context, say "I don't know based on the provided context."

Context:\n{context}

Question:\n{question}

Answer:""",
input_variables=["context", "question"],
)

# Step 4: Create a function format_docs(docs) that takes a list
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Step 5: Create the LLM
llm = OllamaLLM(model="qwen3.5:4b")
chain = (
    {
        "context": retriever | format_docs, 
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)
result = chain.invoke("What programming language is used for AI and what is it used for?")
result2 = chain.invoke("What is the capital of France and what is AI?")
print("Result:", result)
print("Result2:", result2)