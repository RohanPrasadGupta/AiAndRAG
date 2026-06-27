import json
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

claude_client = anthropic.Anthropic() 

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

graph_cypher_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Neo4j Cypher expert.
Given the graph result explain the result in a natural language with the question.


Result:
{result}

Question:
{question}
""")
])

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

class AgentState(TypedDict):
    question: str
    sub_questions: list[dict]  # [{"question": "...", "tool": "vector/graph"}]
    vector_context: str        # result from vector search
    graph_context: str         # result from graph search
    combined_context: str      # merged before final answer
    answer: str
    tools_tried: list[str]


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


decomposer_prompt = ChatPromptTemplate.from_messages([
    (
    "system", """
        You are a question decomposer.
        Break the question into sub-questions.
        For each sub-question decide which tool to use:
        - "vector" for definitions and explanations
        - "graph" for skills, prerequisites, relationships

        Return ONLY valid JSON. No markdown.

        Structure:
        {{
            "sub_questions": [
                {{"question": "What is Machine Learning?", "tool": "vector"}},
                {{"question": "What skills do I need for ML?", "tool": "graph"}}
            ]
        }}

        Question: {question}
        """
    )
])

decomposer_chain = decomposer_prompt | answer_llm | StrOutputParser()
graph_result_chain = graph_cypher_prompt | answer_llm | StrOutputParser()

def vector_search_node(state: AgentState) -> dict:
    vector_question = [
        sq for sq in state["sub_questions"]
        if sq["tool"] == "vector"
        ]
    
    if not vector_question:
        return {"vector_context":""}
    
    all_vector_context = []
    for sq in vector_question:
        docs = retriever.invoke(sq["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        all_vector_context.append(context)
    
    return {"vector_context" : "\n\n".join(all_vector_context),"tools_tried":state["tools_tried"] + ["vector"]}
    
def graph_search_node(state:AgentState)->dict:
    graph_question = [
        sq for sq in state["sub_questions"]
        if sq["tool"] == "graph"
    ]

    if not graph_question:
        return {"graph_context":""}
    
    all_graph_context = []
    for sq in graph_question:
        generate_cypher_query = generate_cypher(sq["question"])
        try:
            result = graph.query(generate_cypher_query)
            context = graph_result_chain.invoke({"question" : sq["question"], "result":str(result)}) if result else "No result Found"
            
        except Exception as e:
            context = "No result Found" 
        all_graph_context.append(context)

    return {"graph_context" : "\n\n".join(all_graph_context),"tools_tried":state["tools_tried"] + ["graph"]}

def combiner_node(state:AgentState)->dict:
    parts = []

    if state["vector_context"]:
        parts.append(f"=== Vector Search === \n{state['vector_context']}")
    if state["graph_context"]:
        parts.append(f"=== Graph Search === \n{state['graph_context']}")

    return {
        "combined_context":"\n\n".join(parts) if parts else "No Context Found."
    }


def decomposer_node(state: AgentState) -> dict:
    full_question = state["question"]

    sub_context = decomposer_chain.invoke({
        "question":full_question
    })
    parsed = json.loads(sub_context)
    result = parsed["sub_questions"]

    return {"sub_questions":result}

def answer_node(state: AgentState) -> dict:
    prompt_text = f"""Answer the question using ONLY the context below.

Context: {state['combined_context']}
Question: {state['question']}

Answer:"""
    answer = answer_llm.invoke(prompt_text)
    return {"answer": answer}


builder = StateGraph(AgentState)

builder.add_node("decomposer", decomposer_node)
builder.add_node("vector_search", vector_search_node)
builder.add_node("graph_search", graph_search_node)
builder.add_node("combiner", combiner_node)
builder.add_node("answer", answer_node)

builder.set_entry_point("decomposer")
builder.add_edge("decomposer","vector_search")
builder.add_edge("vector_search","graph_search")
builder.add_edge("graph_search","combiner")
builder.add_edge("combiner","answer")
builder.add_edge("answer",END)

agent = builder.compile()

# questions = ["What is Artificial Intelligence? and What skills do I need for ML?",
#             "Explain what Machine Learning is?"]
# questions = ["What is Artificial Intelligence? and What skills do I need for ML?"]

# for question in questions:
#     initial_state = {
#         "question": question,
#         "sub_questions": [],
#         "vector_context": "",
#         "graph_context": "",
#         "combined_context": "",
#         "answer": "",
#         "tools_tried": []
#     }

#     final_result = agent.invoke(initial_state)
    
#     print("\n======================")
#     print("Q:", final_result["question"])
#     print("Sub-questions:", final_result["sub_questions"])
#     print("Tools tried:", final_result["tools_tried"])
#     print("Answer:", final_result["answer"])


# CORRECT ✅ — only runs when executed directly
if __name__ == "__main__":
    questions = ["What is AI? and What skills for ML?"]
    for question in questions:
        initial_state = {
            "question": question,
            "sub_questions": [],
            "vector_context": "",
            "graph_context": "",
            "combined_context": "",
            "answer": "",
            "tools_tried": []
        }
        final_result = agent.invoke(initial_state)
        print("final result:", final_result["answer"])