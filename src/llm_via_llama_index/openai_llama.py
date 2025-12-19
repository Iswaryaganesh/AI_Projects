import os
from llama_index.llms.openai import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(
    model = "gpt-3.5-turbo-0125",  # less temperature value -> more similar replies when asked multiple times
    temperature=0
)

response = llm.complete("machine learning is")
print(response)