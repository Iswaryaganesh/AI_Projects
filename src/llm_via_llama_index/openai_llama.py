import os
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(
    model = "gpt-3.5-turbo-0125",  # less temperature value -> more similar replies when asked multiple times
    temperature=0
)

response = llm.complete("machine learning is")
print(response)


messages = [
    ChatMessage(
        role="system", content="You are a pirate with colorful personality"
    ),
    ChatMessage(
        role="user", content="What is your name"
    )
]
chat_response = llm.chat(messages)
print("Chat response :",chat_response)