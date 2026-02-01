from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
print("working dir: ", working_dir)

embedding = HuggingFaceEmbeddings()

# load embedding model
llm = OllamaLLM(
        model="gemma2:2b",
        temperature=0
)

def process_document_to_chroma_db(file_name):
    loader = UnstructuredFileLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    # split text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300
    )
    texts = text_splitter.split_documents(documents)

    # store document chunks in chroma vector database
    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vector_store"
    )
    return 0

def answer_question(user_question):
    vector_db = Chroma(
        persist_directory=f"{working_dir}/doc_vector_store",
        embedding_function=embedding
    )

    # create a retriever for document search
    retriever = vector_db.as_retriever()

    # create a retrievalQA chain to answer user question
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer