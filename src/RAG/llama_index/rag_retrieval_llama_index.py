from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb

Settings.llm = Ollama(
        model="gemma2:2b",
        temperature=0
    )

embed_model = HuggingFaceEmbedding()

vector_db_path = "E:/PycharmProjects/AI_Projects/GenAI_practice/src/assets/vector_db"
collection_name = "documents_collection"

# define persistent db location
db = chromadb.PersistentClient(path=vector_db_path)
chroma_collection = db.get_or_create_collection(collection_name)

# connect to vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#  load index from chroma
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model
)

# create query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# ask questionA
response = query_engine.query("What does the document say about adaptive radiation and energy flow")
print(response)
print("Response: ",response.response)
print("Sources: ")
for node in response.source_nodes:
    print(f"- {node.metadata}")
