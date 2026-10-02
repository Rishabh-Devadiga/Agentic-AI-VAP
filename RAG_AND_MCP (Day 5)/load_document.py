from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# loader = TextLoader("sample.")
pdf_files = ["Introduction_to_Statistical_Learning.pdf",
            "Practical_Statistics_for_Data_Scientists.pdf"]

documents = []

for pdf_file in pdf_files:
    loader = PyMuPDFLoader(pdf_file)
    documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")

# Split document into overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
    )
chunks = text_splitter.split_documents(documents)
print(f"Total document chunks created: {len(chunks)}")

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

vector_store = Chroma.from_documents(documents = chunks,
                                    embedding=embeddings,
                                    persist_directory = "./chroma_db")

print("Vector database created successfully.")