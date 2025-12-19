import os
from langchain_google_genai import GoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyB9Ro3oONqFDG3YkWiD5iS8OFkZV0C1txU"
llm = GoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0
)

response = llm.invoke("What is machine learning")
print(response)