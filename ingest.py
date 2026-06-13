import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma  # Updated import
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    WebBaseLoader
)

load_dotenv()

# Step 1: Load all the Documents from /docs Folder
print("Loading Documents...")
pdf_loader = DirectoryLoader(
    "./docs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)
documents = pdf_loader.load()

# Also load .txt files if any
txt_loader = DirectoryLoader(
    "./docs",
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents += txt_loader.load()

# Load .md files
md_loader = DirectoryLoader("./docs", glob="**/*.md",
                             loader_cls=UnstructuredMarkdownLoader)
documents += md_loader.load()

# Load a web page
web_loader = WebBaseLoader("https://example.com/article")
documents += web_loader.load()

print(f"Loaded {len(documents)} document pages/files")

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # ~1000 Characters per chunk
    chunk_overlap=200 # 200 chars overlap (so context isn't lost)
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} Chunks")

# Step 3: Create embeddings and store
print("Creating Embeddings (this may take a minute)...")
# Uses local Ollama model, ensure Ollama is running and model is pulled
embeddings = OllamaEmbeddings(model="nomic-embed-text") 

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db" # saves locally
)

print("Done! Documents indexed into chroma_db/")