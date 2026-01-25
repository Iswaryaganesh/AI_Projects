import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def ask_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = GoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature = 0
    )
    return llm

def ask_groq():
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )
    return llm

def ask_ollama():
    llm = OllamaLLM(
        model="gemma2:2b",
        temperature=0
    )
    return llm

def ask_open_ai():
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-0125",  # less temperature value -> more similar replies when asked multiple times
        temperature=0
    )
    return llm

# llm = ask_gemini()
llm = ask_ollama()
# llm = ask_groq()
# llm = ask_open_ai()
response = llm.invoke("What is machine learning. Explain in 30 words")
print(response)