from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenAI(
    model="gemini-2.5-flash",
    temperature=0
)

response = llm.complete("Write about corona virus in 20 words")
print(response)

chat_message = [
    ChatMessage(
        role="system", content="You are a pirate with colorful personality"
    ),
    ChatMessage(
        role="user", content="What is your name"
    )
]
chat_response = llm.chat(chat_message)

print("Chat message: ", chat_response)