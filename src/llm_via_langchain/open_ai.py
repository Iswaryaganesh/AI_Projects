import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model = "gpt-3.5-turbo-0125",  # less temperature value -> more similar replies when asked multiple times
    temperature=0
)

response = llm.invoke("What is machine learning. Explain in 30 words")
print(response.content)