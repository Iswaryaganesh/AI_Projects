from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq

load_dotenv()

# streamlit page setup
# emojidb site for emojis
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered"
)
st.title("🗪 Generative AI Chatbot")

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.0
)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    st.chat_message(message["role"]).markdown(message["content"])

user_input = st.chat_input("Ask anything...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    response = llm.invoke(
        input=[{"role": "system", "content": "you are a helpful assistant"}, *st.session_state.chat_history]
    )
    st.chat_message("assistant").markdown(response.content)
    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
