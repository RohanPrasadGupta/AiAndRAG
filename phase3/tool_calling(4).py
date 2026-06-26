from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

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

def route_question(questions: str) -> str:
    result = []
    for question in questions:
        type = question_route_chain.invoke({
        "question":question
        })
        result.append({
            "route": type,
            "question":question
        })

    return result

def generate_cypher(question: str) -> str:
    prompt = f"""You are a Neo4j Cypher expert.
Given the graph schema and a question, write a Cypher query that answers the question.

Return ONLY the Cypher query.
No explanation, no markdown, no code fences.

Graph Schema:
{GRAPH_SCHEMA}

Question: {question}"""

    message = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user", 
            "content": prompt
        }]
    )
    
    return message.content[0].text.strip()

def graph_search(question: str) -> str:
    # Step 1: generate Cypher from question
    generate_cypher_query = generate_cypher(question)
    # Step 2: execute against Neo4j
    query_result = graph.query(generate_cypher_query)
    result = cypher_chain.invoke({"result": json.dumps(query_result), "question": question})
    # Step 3: format results as string
    return result.strip()

def format_vector_result(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def vector_search(question: str) -> str:
    vector_result = retriever.invoke(question)
    formated_result = format_vector_result(vector_result)

    return formated_result

def hybrid_retrieve(questions: str) -> dict:
    result_routes = route_question(questions)
    hybrid_results = []

    for result_route in result_routes:
        if result_route["route"] == "vector":
            context = vector_search(result_route["question"])
        else:
            context = graph_search(result_route["question"])

        hybrid_results.append({
            "route":result_route["route"],
            "question":result_route["question"],
            "context":context
        })

    return hybrid_results

questions = [
    "Explain what Machine Learning is",
    "What skills do I need before learning Deep Learning?",
    "Can you tell me what prerequisites exist for Machine Learning?",
    "What connects Python to Artificial Intelligence?"
]

results = hybrid_retrieve(questions)
print("========================")
for result in results:
    print(result["question"])
    print(result["route"])
    print(result["context"])
    print("========================")