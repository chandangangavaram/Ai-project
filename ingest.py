import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

def load_documents(folder_path="sample_docs"):
    documents = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

def create_vector_store():
    print("[INFO] Loading documents...")
    documents = load_documents()
    print(f"[INFO] Loaded {len(documents)} documents.")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    print("[INFO] Generating embeddings using Ollama...")
    embeddings = OllamaEmbeddings(model="mistral")

    print("[INFO] Creating ChromaDB...")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vectorstore.persist()
    print("[SUCCESS] Vector DB created at:", PERSIST_DIR)

if __name__ == "__main__":
    create_vector_store()

