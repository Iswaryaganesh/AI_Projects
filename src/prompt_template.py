import  os
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from dotenv import load_dotenv

load_dotenv()

def run_prompt_ollama():
    print("This is an ollama model")
    llm = Ollama(
        model="gemma2:2b",
        temperature=0.0
    )
    return llm

def run_prompt_groq():
    print("This is a Groq model")
    api_key = os.getenv("GROQ_API_KEY")
    llm = Groq(
        model="llama-3.1-8b-instant",
        temperature=0
    )
    return llm

template = PromptTemplate("You are an expert AI assistant. Give response only from the data given. Don't generate your own answers. If answer is not sufficient respond as Not sufficient. Context: {context}. Question: {question}")
sample_context = """
Blockchain is a decentralized digital ledger that records transactions across many computers. 
Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data.

Key components:
- **Decentralized**: No single authority controls the network
- **Immutable**: Once data is written, it cannot be altered retroactively
- **Consensus**: Nodes agree on ledger state via Proof-of-Work (Bitcoin) or Proof-of-Stake (Ethereum 2.0)
- **Smart Contracts**: Self-executing code with predefined rules (Ethereum)

Bitcoin (2009): First cryptocurrency, Proof-of-Work mining
Ethereum (2015): Smart contracts, now transitioning to Proof-of-Stake
Use cases: Cryptocurrency, supply chain tracking, digital identity, DeFi

Challenges: Scalability (15 TPS vs Visa's 24K), energy consumption, regulation.
"""
sample_question = "What makes blockchain different from traditional databases?"

prompt = template.format(context=sample_context, question=sample_question)

## llm calls for different models
llm = run_prompt_ollama()
# llm = run_prompt_groq()


stream_response = llm.stream_complete(prompt)
for response in stream_response:
    print(response.delta, end="", flush=True)
print()

