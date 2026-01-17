import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)

parser = StrOutputParser()

def chat():
    # chat history is list of messages. each message has a role and a content i.e system, user and assistant
    chat_history = [
        ("system", "You are a helpful chatbot. Be concise and accurate")
    ]

    print("Langchain chatbot. type 'exit' to quit")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        chat_history.append(("user", user_input)) # list of tuple

        # build prompt dynamically
        prompt = ChatPromptTemplate.from_messages(chat_history)

        # form a chain using prompt then apply llm then parse
        chain = prompt | llm | parser

        response = chain.invoke({})

        print(f"Bot: {response}\n")

        chat_history.append(("assistant", response))
        print("-"*80)

chat()