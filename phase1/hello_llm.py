from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen3.5:4b")
# model = OllamaLLM(model="qwen2.5-coder:14b")

response = model.invoke("Hello, who are you and what can you help me with?")

print(response)