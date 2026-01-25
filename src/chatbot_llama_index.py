import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from  llama_index.core.llms import ChatMessage, MessageRole

load_dotenv()
llm = Groq(
    model="llama-3.3-70b-versatile",
    temperature=0.1
)

def chat():
    history = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful chatbot assistant. Be concise and accurate"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Ask anything")
        ]
    print("Type exit/quit to quit")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        history.append(ChatMessage(role=MessageRole.USER, content=user_input))

        res = llm.chat(messages=history)
        answer = res.message.content
        print("Bot :", answer)

        history.append(ChatMessage(role=MessageRole.USER, content=answer))

        print("-"*50)

chat()

