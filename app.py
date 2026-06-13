import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
load_dotenv()

st.title("My Knowledge Base")

@st.cache_resource
def load_qa():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstores = Chroma(persist_directory="./chroma_db",
                         embedding_function=embeddings)
    llm = OllamaLLM(model="phi3", temperature=0)
    return RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstores.as_retriever(
        search_kwargs={"k": 4}),
        return_source_documents=True
    )

qa = load_qa()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask Anything About Your documents"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching knowledge base.."):
        result = qa.invoke({"query": prompt})

    answer = result["result"]
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})