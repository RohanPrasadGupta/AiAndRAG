from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# llm = OllamaLLM(model="qwen3.5:4b")
llm = OllamaLLM(model="qwen2.5-coder:14b")

prompt = PromptTemplate.from_template("Explain {topic} in one simple sentence, using an analogy a 10-year-old would understand.")
chain = prompt | llm
result = chain.invoke({"topic": "LLM models"})

print("result :",result)