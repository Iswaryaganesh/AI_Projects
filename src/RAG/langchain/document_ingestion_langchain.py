from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import nltk
nltk.download("punkt_tab")

# configuration
docs_dir_path = "E:/PycharmProjects/AI_Projects/GenAI_practice/src/assets/docs_dir"
vector_db_path = "E:/PycharmProjects/AI_Projects/GenAI_practice/src/assets/vector_db"
collection_name = "document_collection"

# loading the embedding model
embedding = HuggingFaceEmbeddings()  # use any hugging face model

# instance for directory loader
loader = DirectoryLoader(
    path=docs_dir_path,
    glob="./*.pdf",
    loader_cls=UnstructuredFileLoader
)

# load the document
documents = loader.load()
print(type(documents))

# initializing text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=300
)

text_chunks = text_splitter.split_documents(documents)
print("chunk", text_chunks)

# create vector store
vector_store = Chroma.from_documents(
    documents=text_chunks,
    embedding=embedding,
    persist_directory=vector_db_path,
    collection_name=collection_name
)
