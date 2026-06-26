from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import json

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password123",
)

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

answer_llm = OllamaLLM(model="qwen3.5:4b")
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",  """You are a helpful assistant.

Analyze the question and extract the topic mentioned in the question
for the Neo4j query.

Return ONLY valid JSON.
Do not use markdown or code fences.

Use this exact structure:
{{
    "topic": ["topic1"]
}}

If the question is not about skills, return:
{{
    "topic": []
}}

Question:{context}"""),
])

final_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Answer the question using ONLY the context provided.
If context is from a graph database, it contains precise facts — 
explain them clearly in natural language.
If context is from documents, summarize the relevant information.

Context:
{context}"""),
    ("human", "{question}")
])

answer_chain = answer_prompt | answer_llm | StrOutputParser()
final_chain = final_prompt | answer_llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def route_question(question: str) -> str:
    if "what is" in question.lower() or "explain" in question.lower():
        return "vector"
    elif "skills" in question.lower() or "required" in question.lower() or "connect" in question.lower() or "path" in question.lower():
        return "graph"
    else:
        return "vector"

def vector_search(question: str) -> str:
    context = retriever.invoke(question)
    formated_context = format_docs(context)
    return formated_context

def graph_search(question: str) -> str:
    skills = answer_chain.invoke({"context": question})
    json_skills = json.loads(skills)
    topic = json_skills["topic"]
    topic_name = topic[0] if topic else None
    
    if not topic_name:
        return "No topic found in the question."

    print("topic_name: ", topic_name)

    query = """
    MATCH (skill)-[:REQUIRED_FOR*1..3]->(target {name: $topic_name})
    RETURN 
        skill.name AS required_skill,
        target.name AS required_for
    """

    results = graph.query(query, params={"topic_name": topic_name})
    print("results: ", results)
    lines = []
    for row in results:
        lines.append(f"- {row['required_skill']} is required for {row['required_for']}")

    return "\n".join(lines)

def hybrid_retrieve(question: str) -> dict:
    route = route_question(question)
    if route == "vector":
        context = vector_search(question)
    else:
        context = graph_search(question)
    
    return {"route": route, "context": context}

question_a = "Explain what Machine Learning is"
question_b = "What skills do I need before learning Deep Learning?"

result_a = hybrid_retrieve(question_a)
result_b = hybrid_retrieve(question_b)

print("result_a: ", result_a)
print("=================== \n")
print("result_b: ", result_b)

print("\n========= FINAL ANSWERS =========\n")

answer_a = final_chain.invoke({
    "context": result_a["context"],
    "question": question_a
})
print(f"Q: {question_a}")
print(f"Source: {result_a['route']}")
print(f"A: {answer_a}\n")

answer_b = final_chain.invoke({
    "context": result_b["context"],
    "question": question_b
})
print(f"Q: {question_b}")
print(f"Source: {result_b['route']}")
print(f"A: {answer_b}\n")
