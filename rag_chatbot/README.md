# Genetics RAG Chatbot

An AI-powered genetics tutor that uses Retrieval Augmented Generation (RAG) to answer questions based on open-source genetics textbooks.

## Setup

1. **Activate the virtual environment:**
   ```bash
   cd rag_chatbot
   source venv/bin/activate
   ```

2. **Set your Anthropic API key:**
   ```bash
   export ANTHROPIC_API_KEY='your-key-here'
   ```

3. **Build the vector store (first time only):**
   ```bash
   python ingest.py
   ```

4. **Start chatting:**
   ```bash
   python chat.py
   ```

## How It Works

1. **Ingestion (`ingest.py`):**
   - Loads all extracted text files from the genetics knowledge base
   - Chunks text into ~1000 character segments with overlap
   - Creates embeddings using `all-MiniLM-L6-v2` (runs locally on Apple Silicon)
   - Stores vectors in ChromaDB

2. **Chat (`chat.py`):**
   - Takes your question and finds relevant textbook passages
   - Sends the context + question to Claude
   - Returns an educational response grounded in the textbook content

## Knowledge Sources

- OpenStax Biology 2e (CC BY 4.0)
- Online Open Genetics - LibreTexts (CC BY-SA 3.0)
- NIGMS "The New Genetics" (Public Domain)
- Schleif "Genetics and Molecular Biology"

## Requirements

- Python 3.10+
- Anthropic API key
- ~2GB disk space for embeddings
