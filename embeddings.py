import os
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec, Pinecone
from dotenv import load_dotenv
load_dotenv()

index_name = "rag-health-docs"

def load_documents_from_folder(folder_path):
    docs = []
    folder = Path(folder_path)

    for file_path in folder.glob("*"):
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            continue  # skip other files

        file_docs = loader.load()
        for doc in file_docs:
            doc.metadata["source"] = file_path.name
        docs.extend(file_docs)
    
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)

# ========== STEP 3: Create Pinecone Index ==========
def create_pinecone_index(index_name):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    if index_name in pc.list_indexes().names():
        print(f"🗑️ Deleting existing index '{index_name}'...")
        pc.delete_index(index_name)
    print(f"📦 Creating new Pinecone index '{index_name}'...")
    pc.create_index(
        index_name,
        dimension=768,  # text-embedding-3-large → 768-d
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    print(f"✅ Index '{index_name}' is ready.")
    return pc

def index_documents(index_name, documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=os.environ.get("GOOGLE_API_KEY"))
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )
    print(f"✅ Successfully stored {len(documents)} chunks.")
    return vectorstore.as_retriever()

if __name__ == "__main__":
    raw_docs = load_documents_from_folder(r"D:\91824\Downloads\Insurance PDFs\Insurance PDFs")  # your folder with PDFs + DOCX
    split_docs = split_documents(raw_docs)
    create_pinecone_index(index_name)
    retriever = index_documents(index_name, split_docs)