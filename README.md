# Local RAG Chatbot

A fully local, privacy-friendly chatbot that answers questions based on your own documents (PDFs and text files) using Retrieval-Augmented Generation (RAG). Runs entirely on your machine — no API keys, no cloud calls, no cost.

## Features

- 💬 Chat interface built with Streamlit
- 📄 Ask questions about your own PDFs / text files
- 🔍 Retrieves relevant document chunks before answering (with source citations)
- 🖥️ Runs 100% locally using [Ollama](https://ollama.com) — no internet required after setup
- 🔒 Your documents never leave your computer

## Tech Stack

- [LangChain](https://www.langchain.com/) (`langchain-classic` for chains)
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [Ollama](https://ollama.com) for local embeddings + LLM
- [Streamlit](https://streamlit.io/) for the chat UI

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/local-rag-chatbot.git
cd local-rag-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the required Ollama models

```bash
ollama pull nomic-embed-text
ollama pull phi3
```

> You can swap `phi3` for `llama3` or `mistral` in `app.py` if your machine can handle a larger model.

### 5. Add your documents

Place any PDF or `.txt` files you want to chat with inside the `docs/` folder.

### 6. Build the knowledge base

```bash
python ingest.py
```

This reads your documents, splits them into chunks, generates embeddings, and stores them in a local ChromaDB database (`chroma_db/`).

### 7. Run the app

```bash
streamlit run app.py
```

This opens a chat interface in your browser where you can ask questions about your documents.

## Project Structure

```
local-rag-chatbot/
├── docs/            # your PDFs / text files (not tracked by git)
├── ingest.py        # loads, chunks, and embeds your documents
├── app.py           # Streamlit chat interface
├── requirements.txt
└── README.md
```

## Notes

- If you add new documents to `docs/`, delete `chroma_db/` and re-run `python ingest.py` to rebuild the index.
- If you switch embedding models, you must delete `chroma_db/` and re-ingest — embeddings from different models are not compatible with each other.

## License

MIT
