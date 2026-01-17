import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

# 1. answer question
load_dotenv()
def ask_groq():
    api_key = os.getenv("GROQ_API_KEY")
    llm = Groq(
        model="llama-3.1-8b-instant",
        temperature=0
    )
    return llm

def ask_open_ai():
    api_key = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(
        model="gpt-3.5-turbo-0125",  # less temperature value -> more similar replies when asked multiple times
        temperature=0
    )
    return llm

def ask_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    return llm

def ask_ollama():
    # local model
    llm = Ollama(
        model="gemma2:2b",
        temperature=0,
        request_timeout=120.0
    )
    return llm

def call_llm(llm):
    response = llm.stream_complete("What is AI. Explain in 30 words")
    # print(response)
    for response in response:
        print(response.delta, end="", flush=True)  # Token-by-token
    print()

def call_llm_chat(llm, messages):
    chat_response = llm.chat(messages)
    for message in messages:
        print(message.role.value, ": ", message.content)
    print(chat_response.message)

# llm = ask_groq()
llm = ask_ollama()
# llm = ask_gemini()
# llm = ask_open_ai()
call_llm(llm)

# 2. chat message
# messages = [
#     ChatMessage(
#         role="system", content="You are a pirate with colorful personality"
#     ),
#     ChatMessage(
#         role="user", content="What is your name"
#     )
# ]
#
# call_llm_chat(llm, messages)
