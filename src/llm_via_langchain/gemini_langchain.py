import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0
)

response = llm.invoke("What is machine learning. Explain in 30 words")
print(response)