from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

import nltk

nltk.download("punkt_tab")
nltk.download("stopwords")

# configuration
docs_dir_path = "E:/PycharmProjects/AI_Projects/GenAI_practice/src/assets/docs_dir"
vector_db_path = "E:/PycharmProjects/AI_Projects/GenAI_practice/src/assets/vector_db"
collection_name = "documents_collection"

# embedding function
embed_model = HuggingFaceEmbedding()

# directory loader
loader = SimpleDirectoryReader(input_dir=docs_dir_path)

# load the document
documents = loader.load_data()
print(len(documents))

# create parser with chunking strategy
parser = SimpleNodeParser.from_defaults(chunk_size=1500, chunk_overlap=300)

# convert documents to chunks (nodes)
nodes = parser.get_nodes_from_documents(documents)
print(len(nodes))

# define persistent db location
db = chromadb.PersistentClient(path=vector_db_path)

# create or retrieve the vector collection
chroma_collection = db.get_or_create_collection(collection_name)

# connect to vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#  load index from chroma
index = VectorStoreIndex(
    nodes,
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model
)
