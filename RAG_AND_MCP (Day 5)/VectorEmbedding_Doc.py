from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

vector_store = Chroma.from_documents(documents = chunks,
                                    embedding=embeddings,
                                    persist_directory = "./chroma_db")

print("Vector database ")