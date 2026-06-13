import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
# Remove: from langchain.chains import RetrievalQA
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- Load the vector store we created ---
print("Loading knowledge base...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# --- Create a retriever (finds top 4 similar chunks) ---
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# --- Custom prompt: tells GPT to only use provided context ---
prompt_template = """You are a helpful assistant answering questions
based ONLY on the documents provided below. If the answer is not
in the documents, say "I don't have that information in my knowledge base."

Context from your documents:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# --- Set up the QA chain ---
llm = OllamaLLM(model="phi3", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# --- Chat loop ---
print("\n✅ Knowledge base ready! Ask questions (type 'quit' to exit)\n")
while True:
    question = input("You: ").strip()
    if question.lower() in ["quit", "exit", "q"]:
        break
    if not question:
        continue

    result = qa_chain.invoke({"query": question})
    print(f"\nAI: {result['result']}")

    # Show which documents were used
    print("\n📄 Sources:")
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        print(f"  - {src}" + (f" (page {page})" if page else ""))
    print()