# Imports
import json
import stat
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── State definition ─────────────────────────────────────────────
class AgentState(TypedDict):
    question: str
    route: str
    context: str
    answer: str
    retry_count: int
    tools_tried: list[str]

# ── Setup (Ollama, Chroma, Neo4j, Claude — same as before) ──────

claude_client = anthropic.Anthropic() 

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

GRAPH_SCHEMA = """
Node labels and properties:
- Topic {name: string}    examples: "Artificial Intelligence", 
                                    "Machine Learning", "Deep Learning"
- Skill {name: string}    examples: "Python", "Mathematics", 
                                    "Data Analysis", "Cloud Computing"
- Performance {name: string}  examples: "AI Model Performance"

Relationships:
- (Topic)-[:IS_BRANCH_OF]->(Topic)
- (Topic)-[:REQUIRES_SKILL]->(Skill)
- (Skill)-[:USED_FOR]->(Topic)
- (Topic)-[:CORE_COMPONENT_OF]->(Topic)
- (Skill)-[:IMPROVES]->(Performance)
- (Skill)-[:REQUIRED_FOR]->(Topic)
- (Topic)-[:ADVANCED_AREA_OF]->(Topic)
- (Skill)-[:REQUIRED_FOR]->(Topic)
- (Topic)-[:EXTENDS]->(Topic)
- (Skill)-[:SUPPORTS]->(Topic)
"""

cypher_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Neo4j Cypher expert.
Given the graph result explain the result in a natural language with the question.


Result:
{result}

Question:
{question}
""")
])

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a routing assistant.
Decide whether this question needs:
- "vector" search (for explanations, definitions, descriptions)
- "graph" search (for relationships, prerequisites, connections,
                  paths between concepts, what skills are needed)

Return ONLY one word: vector or graph
No explanation."""),
    ("human", "{question}")
])

question_route_chain = router_prompt | answer_llm | StrOutputParser()
cypher_chain = cypher_prompt | answer_llm | StrOutputParser()

def generate_cypher(question: str) -> str:
    prompt = f"""You are a Neo4j Cypher expert.

Given the graph schema and a question, write a Cypher query.

Return ONLY the Cypher query.
No explanation, no markdown, no code fences.

Important Neo4j syntax rule:
When using multiple alternative relationship types, use a colon only
before the first relationship type.

Correct:
[:REQUIRES_SKILL|REQUIRED_FOR|SUPPORTS]

Incorrect:
[:REQUIRES_SKILL|:REQUIRED_FOR|:SUPPORTS]

Graph Schema:
{GRAPH_SCHEMA}

Question:
{question}
"""

    message = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user", 
            "content": prompt
        }]
    )
    
    return message.content[0].text.strip()


# ── Node 1: router_node ──────────────────────────────────────────
def router_node(state:AgentState)->dict:
    question  = state["question"]
    route = question_route_chain.invoke({"question" : question})

    return {"route":route.strip().lower()}
    
# ── Node 2: vector_search_node ────────────────────────────────────
def vector_search_node(state:AgentState)->dict:
    quesiton = state["question"]
    docs = retriever.invoke(quesiton)
    result = "\n\n".join(doc.page_content for doc in docs)

    return {
        "context" : result,
        "tools_tried": state["tools_tried"] + ["vector"]
    }


# ── Node 3: graph_search_node ─────────────────────────────────────
def graph_search_node(state:AgentState)->dict:
    quesiton = state["question"]

    generate_cypher_query = generate_cypher(quesiton)

    try:
        result = graph.query(generate_cypher_query)
        context = str(result) if result else "No result Found"
    except Exception as e:
        context = "No result Found"

    return{
        "context":context,
        "tools_tried":state["tools_tried"] + ["graph"]
    }

# ── Node 4: answer_node ───────────────────────────────────────────
def answer_node(state:AgentState)->dict:
    prompt_text = f"""Answer the question using ONLY the context below.
    
Context: {state["context"]}
Question: {state["question"]}

Answer:"""
    answer = answer_llm.invoke(prompt_text)
    return {"answer": answer}

# ── Conditional edge functions ───────────────────────────────────
def route_decision(state:AgentState) -> dict:
    # Returns the route value to follow next
    return state["route"]

def handle_fallback_node(state:AgentState) -> dict:
    return {"retry_count": state["retry_count"] + 1}

def graph_fallback_decision(state:AgentState) -> str:
    if state["context"] == "No result Found" and "vector" not in state["tools_tried"]:
        return "fallback"
    
    return "answer"

# ── Build the graph ───────────────────────────────────────────────

builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("vector_search", vector_search_node)
builder.add_node("graph_search", graph_search_node)
builder.add_node("answer", answer_node)
builder.add_node("handle_fallback", handle_fallback_node)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "vector":"vector_search",
        "graph":"graph_search"
    }
)

# builder.add_conditional_edges(
#     "graph_search",
#     graph_fallback_decision,
#     {
#         "retry_vector":"vector_search",
#         "answer":"answer"
#     }
# )

builder.add_conditional_edges(
    "graph_search",
    graph_fallback_decision,
    {
        "fallback":"handle_fallback",
        "answer":"answer"
    }
)

builder.add_edge("handle_fallback","vector_search")

builder.add_edge("vector_search","answer")

builder.add_edge("answer",END)

agent = builder.compile()

# ── Test with 3 questions ─────────────────────────────────────────
questions = [
    "Explain what Machine Learning is",              # → vector path
    "What skills do I need for Deep Learning?",     # → graph path, success
    "What is the connection between X and Y?"        # → graph path may fail → fallback
]


allResult = []

for question in questions:
    initial_state = {
        "question": question,
        "route": "",
        "context": "",
        "answer": "",
        "retry_count": 0,
        "tools_tried": []
    }
    final_state = agent.invoke(initial_state)
    print("\n\n")
    print("======================")
    print("question: ", final_state["question"])
    print("route: ", final_state["route"])
    print("context: ", final_state["context"])
    print("answer: ", final_state["answer"])
    print("retry_count: ", final_state["retry_count"])
    print("tools_tried: ", final_state["tools_tried"])
    print("======================")
    allResult.append(final_state)

# for result in allResult:
#     print("\n\n")
#     print("======================")
#     print("question: ", result["question"])
#     print("route: ", result["route"])
#     print("context: ", result["context"])
#     print("answer: ", result["answer"])
#     print("retry_count: ", result["retry_count"])
#     print("tools_tried: ", result["tools_tried"])

