from multiprocessing import context
from typing import Literal
from unicodedata import category
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from pydantic import BaseModel, Field
import json

# test_dataset = [
#     {
#         "question": "What is Artificial Intelligence?",
#         "ground_truth": "Artificial Intelligence is a branch of computer science that focuses on creating machines and software capable of performing tasks that normally require human intelligence, such as learning, reasoning, problem-solving, decision-making, language understanding, and image recognition."
#     },
#     {
#         "question": "What programming language is most widely used in AI?",
#         "ground_truth": "Python is the most widely used programming language in AI because of its simplicity and the large number of AI libraries available."
#     },
#     {
#         "question": "Why is mathematics important for learning AI?",
#         "ground_truth": "Mathematics is important for learning AI because concepts such as algebra, calculus, probability, and statistics help learners understand how AI algorithms work and how machine learning models are built and improved."
#     },
#     {
#         "question": "What is Machine Learning in AI?",
#         "ground_truth": "Machine Learning is a core component of AI that enables systems to learn patterns from data and make predictions without being explicitly programmed for every task."
#     },
#     {
#         "question": "Why are cloud computing skills valuable in AI development?",
#         "ground_truth": "Cloud computing skills are valuable in AI development because platforms such as AWS, Microsoft Azure, and Google Cloud provide tools for training, deploying, and managing AI models at scale."
#     }
# ]

test_dataset = [
    {
        "question": "What is Artificial Intelligence?",
        "ground_truth": "Artificial Intelligence is a branch of computer science that focuses on creating machines and software capable of performing tasks that normally require human intelligence, such as learning, reasoning, problem-solving, decision-making, language understanding, and image recognition."
    },
    {
        "question": "What programming language is most widely used in AI?",
        "ground_truth": "Python is the most widely used programming language in AI because of its simplicity and the large number of AI libraries available."
    }
]
evaluater_llm = OllamaLLM(model="qwen2.5:1.5b")
answer_llm = OllamaLLM(model="qwen3.5:4b")

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",""" 
    You are a helpful assistant.
Answer using ONLY the context provided below.
If the answer is not in the context, say you don't know.

You MUST respond with ONLY a valid JSON object.
No big explanation for answer just 100 words, no markdown, no code fences — just raw JSON.

Use this exact structure:
{{
    "answer": "your answer here in 50 words",
}}

Context:{retriver_context}"""),
    (
        "human","{input_question}"
    )
])


evaluater_prompt = ChatPromptTemplate.from_messages([
    ("system", """ 
You are a helpful evaluator assistant.

You need to evaluate the AI answer using:
1. The retrieved context
2. The ground truth answer
3. The AI result

Return scores from 0 to 5.

Score meaning:
0 = very bad
1 = poor
2 = partially correct
3 = acceptable
4 = good
5 = excellent

Evaluate these:
- correctness: Does the AI result match the ground truth?
- groundedness: Is the AI result supported by the retrieved context?
- retrieval_relevance: Is the retrieved context relevant to the question and ground truth?

You MUST respond with ONLY a valid JSON object.
No markdown.
No code fences.
No explanation outside JSON.

Use this exact structure:
{{
    "raw_context_ai_res": "copy the AI result here",
    "ground_truth": "copy the ground truth here",
    "correctness": 0,
    "groundedness": 0,
    "retrieval_relevance": 0,
    "reasoning": "short reason for the scores"
}}

Retrieved context:
{retriver_context}

AI result:
{raw_context_ai_res}

Ground truth:
{ground_truth}

"""),
])


llm_answer_chain = answer_prompt | answer_llm | StrOutputParser()
llm_score_chain = evaluater_prompt | evaluater_llm | StrOutputParser()

result = []

def formatDocs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_pipeline(input_data):
    question = input_data["question"]
    ground_truth = input_data["ground_truth"]

    docs = retriever.invoke(question)
    retriver_context = formatDocs(docs)

    # print("retriver_context==>",retriver_context)
    # print("\n")

    raw_context_ai_res = llm_answer_chain.invoke(
        {
            "input_question":question,
            "retriver_context":retriver_context
        }
    )

    print(f"[Q: {question[:40]}]")
    print(f"[Raw answer]: {raw_context_ai_res}")

    # print("raw_context_ai_res =>",raw_context_ai_res)
    # print("\n")

    score_ai_res = llm_score_chain.invoke({
            "retriver_context":retriver_context,
            "raw_context_ai_res":raw_context_ai_res,
            "ground_truth":ground_truth,
    })
    
    # print("score_ai_res =>",score_ai_res)
    # print("\n")

    result.append(score_ai_res)

for data in test_dataset:
    rag_pipeline(data)




print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)

total_correctness = 0
total_groundedness = 0
total_retrieval = 0
valid_count = 0

for i, raw in enumerate(result):
    try:
        # Clean markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        scored = json.loads(cleaned)
        
        c = scored.get("correctness", 0)
        g = scored.get("groundedness", 0)
        r = scored.get("retrieval_relevance", 0)
        
        total_correctness += c
        total_groundedness += g
        total_retrieval += r
        valid_count += 1
        
        print(f"\nQ{i+1}: {test_dataset[i]['question'][:50]}...")
        print(f"  Correctness:         {c}/5")
        print(f"  Groundedness:        {g}/5")
        print(f"  Retrieval Relevance: {r}/5")
        print(f"  Reasoning: {scored.get('reasoning', '')[:80]}...")
        
    except json.JSONDecodeError as e:
        print(f"\nQ{i+1}: Parse error — {e}")

print("\n" + "="*60)
print("AVERAGE SCORES")
print("="*60)
if valid_count > 0:
    print(f"Correctness:         {total_correctness/valid_count:.1f}/5")
    print(f"Groundedness:        {total_groundedness/valid_count:.1f}/5")
    print(f"Retrieval Relevance: {total_retrieval/valid_count:.1f}/5")
    print(f"Overall RAG Score:   {(total_correctness+total_groundedness+total_retrieval)/(valid_count*3*5)*100:.1f}%")