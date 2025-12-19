import ollama

response = ollama.chat(
    model="gemma2:2b",
    messages=[
        {"role":"system", "content":"You are a helpful assistant"},
        {"role": "user", "content": "What is machine learning? Explain briefly"},
    ],
    stream=True
)
for chunk in response:
    print(chunk["message"]["content"], end="", flush=True)
