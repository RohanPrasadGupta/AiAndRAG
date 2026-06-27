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

{chat_history}

Context:\n{context}

Question:\n{question}

Answer:""",
input_variables=["context", "question", "chat_history"],
)

# Create a function format_docs(docs) that takes a list
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Create the LLM
# llm = OllamaLLM(model="qwen2.5:1.5b")
llm = OllamaLLM(model="qwen3.5:4b")
chain = (
    {
        "context":(lambda x : x['question']) | retriever | format_docs, 
        "question": lambda x : x['question'],
        "chat_history": lambda x : x['chat_history']
    }
    | prompt
    | llm
)
# Turn 1 — ask about Python
result = chain.invoke({
    "question": "What programming language is used for AI?",
    "chat_history": ""
})

# Turn 2 — deliberately vague, no mention of Python
result2 = chain.invoke({
    "question": "Can you tell me more about it?",
    "chat_history": f"Human: What programming language is used for AI?\nAI: {result}"
})

print("Turn 1:", result)
print("Turn 2:", result2)