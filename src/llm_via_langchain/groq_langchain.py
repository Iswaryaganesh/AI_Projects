import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0
)

response = llm.invoke("What is machine learning")
print(response.content)