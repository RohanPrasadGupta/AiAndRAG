from multiprocessing import context
from typing import Literal
from unicodedata import category
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    answer:str = Field(description="Answer to the quetion")
    confidence:Literal["high","medium","low"]
    category:str
    source_used:list[str]


comporession_llm = OllamaLLM(model="qwen2.5:1.5b")
answer_llm = OllamaLLM(model="qwen3.5:4b")

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})


compression_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a chat history and the latest user question,
formulate a standalone question that can be understood
without the chat history.
Do NOT answer — just reformulate if needed, otherwise return as is."""),

MessagesPlaceholder(variable_name="chat_history"),

("human", "{input}"),
])

answer_prompt = ChatPromptTemplate.from_messages([(
    "system", """You are a helpful assistant.
Answer using ONLY the context provided below.
If the answer is not in the context, say you don't know.

You MUST respond with ONLY a valid JSON object.
No big explanation for answer just 100 words, no markdown, no code fences — just raw JSON.

Use this exact structure:
{{
    "answer": "your answer here in 100 words",
    "confidence": "high or medium or low",
    "category": "the topic category",
    "sources_used": ["source1"]
}}

Context:{context}"""),

MessagesPlaceholder(variable_name="chat_history"),

("human","{input}")

])

# def formatDocs(docs):
#     combineContext = ""

#     for doc in docs:
#         combineContext = combineContext + doc.page_content

#     return(combineContext)

def formatDocs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_pipline5(input_dict):
    question = input_dict["input"]
    chat_history = input_dict.get("chat_history",[])


    if chat_history:
        # if hostory present and asked more on questions
        reWrittern = compression_chain.invoke({
            "input":question,
            "chat_history":chat_history
        })
    else:
        reWrittern = question

    docs = retriever.invoke(reWrittern)
    context = formatDocs(docs)

    raw_response = answer_chain.invoke({
        "input":question,
        "chat_history":chat_history,
        "context":context
    })

    return raw_response

compression_chain = compression_prompt | comporession_llm | StrOutputParser()
answer_chain = answer_prompt | answer_llm | StrOutputParser()


chat_history5 = []

userQuestion = "what programming language is used as AI ?"

result1 = rag_pipline5({
    "input":userQuestion,
    "chat_history": chat_history5
})

print("================== Result1 ==================")
print("Result1:=>", result1)
chat_history5.append(HumanMessage(content=userQuestion))
chat_history5.append(AIMessage(
    content= result1.answer if isinstance(result1,RAGResponse) else str(result1)
    ))

# ==================================================================

userQuestion2 = "Can you Explaim more on AI and cloud Computing"
result2 = rag_pipline5({
    "input":userQuestion2,
    "chat_history": chat_history5
})


print("================== Result2 ================== ")
print("Result2:=>", result2)
chat_history5.append(HumanMessage(content=userQuestion))
chat_history5.append(AIMessage(
    content= result2.answer if isinstance(result2,RAGResponse) else str(result2)
    ))

print("================== Chat History ==========================")
print("chat_history5=>",chat_history5)
