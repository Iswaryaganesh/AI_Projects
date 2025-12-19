from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model = "gemma2:2b",
    temperature = 0
)

response = llm.invoke("What is machine learning")
print(response)