import json
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Literal

# ── Models ─────────────────────────────────────────────────────────
compression_llm = OllamaLLM(model="qwen3.5:4b")
answer_llm = OllamaLLM(model="qwen3.5:4b")

# ── Pydantic schema (validation only, no .with_structured_output) ──
class RAGResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    confidence: Literal["high", "medium", "low"]
    category: str
    sources_used: list[str]

# ── Embeddings + retriever ─────────────────────────────────────────
embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Compression prompt (same as before) ───────────────────────────
compression_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a chat history and the latest user question,
formulate a standalone question that can be understood
without the chat history.
Do NOT answer — just reformulate if needed, otherwise return as is."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# ── Answer prompt — instructs LLM to return JSON ──────────────────
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Answer using ONLY the context provided below.
If the answer is not in the context, say you don't know.

You MUST respond with ONLY a valid JSON object.
No explanation, no markdown, no code fences — just raw JSON.

Use this exact structure:
{{
    "answer": "your answer here",
    "confidence": "high or medium or low",
    "category": "the topic category",
    "sources_used": ["source1"]
}}

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# ── Helpers ────────────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def parse_response(raw: str) -> RAGResponse:
    # Strip markdown fences if model wraps output anyway
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    return RAGResponse(**data)

# ── Chains ─────────────────────────────────────────────────────────
compression_chain = compression_prompt | compression_llm | StrOutputParser()
answer_chain = answer_prompt | answer_llm | StrOutputParser()

# ── Pipeline ───────────────────────────────────────────────────────
def rag_pipeline(input_dict):
    question = input_dict["input"]
    chat_history = input_dict.get("chat_history", [])

    # Step 1: rewrite question
    rewritten = compression_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    print(f"\n[Rewritten]: {rewritten}")

    # Step 2: retrieve chunks
    docs = retriever.invoke(rewritten)
    print("docs chunks...",docs)
    context = format_docs(docs)

    # Step 3: generate raw JSON string
    raw_response = answer_chain.invoke({
        "input": question,
        "chat_history": chat_history,
        "context": context
    })
    print(f"\n[Raw LLM output]: {raw_response}")

    # Step 4: parse into Pydantic object
    try:
        structured = parse_response(raw_response)
        return structured
    except (json.JSONDecodeError, Exception) as e:
        print(f"[Parse error]: {e}")
        print("[Falling back to raw response]")
        return raw_response

# ── Run ────────────────────────────────────────────────────────────
chat_history = []

result1 = rag_pipeline({
    "input": "What programming language is used for AI?",
    "chat_history": chat_history
})

print(f"\n── Turn 1 ──")
if isinstance(result1, RAGResponse):
    print(f"Answer:     {result1.answer}")
    print(f"Confidence: {result1.confidence}")
    print(f"Category:   {result1.category}")
    print(f"Sources:    {result1.sources_used}")
else:
    print(f"Raw: {result1}")

chat_history.append(HumanMessage(
    content="What programming language is used for AI?"
))
chat_history.append(AIMessage(
    content=result1.answer if isinstance(result1, RAGResponse) else str(result1)
))