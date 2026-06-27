from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

vectorstore = Chroma(
    embedding_function=embedding_model, 
    persist_directory="./chroma_db"
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model="qwen3.5:4b")

compression_prompt = ChatPromptTemplate.from_messages(
    [("system", """ Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history.
Do NOT answer the question — just reformulate it if needed,
otherwise return it as is. """),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{input}"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Answer using ONLY the context below.
If the answer is not in the context, say 'I don't know based on the provided context.'

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    compression_prompt
)

answer_chain = create_stuff_documents_chain(
    llm,
    answer_prompt
)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    answer_chain
)

chat_history = []  # empty list — no history yet

result1 = rag_chain.invoke({
    "input": "What programming language is used for AI?",
    "chat_history": chat_history
})

print("Turn 1:", result1["answer"])

chat_history.append(HumanMessage(content="What programming language is used for AI?"))
chat_history.append(AIMessage(content=result1["answer"]))


print("chat_history after turn 1:", chat_history)

result2 = rag_chain.invoke({
    "input": "Can you tell me more about it?",
    "chat_history": chat_history
})

print("\nTurn 2:", result2["answer"])
chat_history.append(HumanMessage(content="What programming language is used for AI?"))
chat_history.append(AIMessage(content=result2["answer"]))


print("chat_history after turn 2:", chat_history)