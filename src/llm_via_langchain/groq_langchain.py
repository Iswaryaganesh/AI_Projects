import os
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0
)

response = llm.invoke("What is machine learning")
print(response.content)