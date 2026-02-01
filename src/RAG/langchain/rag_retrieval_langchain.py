from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

vector_db_path = "/src/assets/vector_db"
collection_name = "document_collection"

embedding = HuggingFaceEmbeddings()

llm = OllamaLLM(
        model="gemma2:2b",
        temperature=0
    )

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding,
    persist_directory=vector_db_path
)

retriever = vector_store.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

query = "What does the document say about kingdom monera?"
response = qa_chain.invoke({"query": query})
print(response["result"])
print("-"*50)
for source in response["source_documents"]:
    print(source.metadata)