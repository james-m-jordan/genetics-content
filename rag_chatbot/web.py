#!/usr/bin/env python3
"""
Genetics Tutor - Web Interface
A Gradio-based chat UI for students to ask genetics questions.
"""

import os
import platform
from pathlib import Path

import gradio as gr
from anthropic import Anthropic
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
ROOT_DIR = Path(__file__).parent.parent
VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"
ENV_FILE = ROOT_DIR / ".env"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# System prompt
SYSTEM_PROMPT = """You are a knowledgeable genetics tutor helping students understand genetics concepts.

You have access to content from several genetics textbooks including:
- OpenStax Biology 2e
- Online Open Genetics (Nickle & Barrette-Ng)
- NIGMS "The New Genetics"
- Schleif's "Genetics and Molecular Biology"

When answering questions:
1. Use the provided context from the textbooks to give accurate, educational responses
2. Explain concepts clearly, suitable for undergraduate students
3. If the context doesn't contain relevant information, say so honestly
4. Use examples and analogies when helpful
5. Cite which textbook source the information comes from when possible

Format your responses using markdown for readability. Be encouraging and supportive!"""


def get_device() -> str:
    """Detect the best available device for embeddings."""
    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "mps"
    return "cpu"


def load_vector_store() -> Chroma:
    """Load the ChromaDB vector store."""
    if not VECTOR_STORE_DIR.exists():
        raise RuntimeError(
            "Vector store not found! Run: uv run python rag_chatbot/ingest.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": get_device()},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embeddings,
        collection_name="genetics_knowledge",
    )


def retrieve_context(vector_store: Chroma, query: str, k: int = 5) -> str:
    """Retrieve relevant context from textbooks."""
    results = vector_store.similarity_search(query, k=k)
    context_parts = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(context_parts)


# Load environment and initialize
load_dotenv(ENV_FILE)
api_key = os.environ.get("ANTHROPIC_API_KEY")

if not api_key:
    print("\n" + "=" * 60)
    print("ANTHROPIC_API_KEY not found!")
    print("Set it in .env file or environment variable.")
    print("Get a key at: https://console.anthropic.com/api-keys")
    print("=" * 60 + "\n")

# Initialize vector store (lazy load on first query if needed)
vector_store = None
client = None


def initialize():
    """Initialize vector store and client on first use."""
    global vector_store, client
    if vector_store is None:
        print("Loading knowledge base...")
        vector_store = load_vector_store()
        print("Knowledge base loaded!")
    if client is None and api_key:
        client = Anthropic(api_key=api_key)


def respond(message: str, history: list) -> str:
    """Generate a response to the user's question."""
    initialize()

    if not api_key:
        return "**Error:** No API key configured. Please set ANTHROPIC_API_KEY in your .env file."

    if not message.strip():
        return "Please ask a genetics question!"

    # Retrieve relevant context
    context = retrieve_context(vector_store, message)

    # Build conversation history for Claude
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    # Add current question with context
    user_message = f"""Based on the following textbook content, please answer the student's question.

TEXTBOOK CONTENT:
{context}

STUDENT'S QUESTION:
{message}

Please provide a clear, educational response based on the textbook content above."""

    messages.append({"role": "user", "content": user_message})

    # Get response from Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return response.content[0].text


# Example questions for students
EXAMPLES = [
    "What is DNA and what is its structure?",
    "Explain Mendel's laws of inheritance",
    "How does mRNA transcription work?",
    "What is the difference between mitosis and meiosis?",
    "What causes genetic mutations?",
    "How do dominant and recessive alleles work?",
]

# Create the Gradio interface
with gr.Blocks(
    title="Genetics Tutor",
    theme=gr.themes.Soft(primary_hue="emerald"),
) as demo:
    gr.Markdown(
        """
        # 🧬 Genetics Tutor

        Ask me anything about genetics! I have access to multiple textbooks including
        OpenStax Biology, Online Open Genetics, and more.

        *Powered by RAG + Claude*
        """
    )

    chatbot = gr.ChatInterface(
        fn=respond,
        examples=EXAMPLES,
        cache_examples=False,
        chatbot=gr.Chatbot(height=500, placeholder="Ask a genetics question..."),
        textbox=gr.Textbox(
            placeholder="What would you like to know about genetics?",
            container=False,
            scale=7,
        ),
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear",
    )

    gr.Markdown(
        """
        ---
        **Sources:** OpenStax Biology 2e (CC BY 4.0) • Online Open Genetics (CC BY-SA 3.0) •
        NIGMS "The New Genetics" (Public Domain) • Schleif "Genetics and Molecular Biology"
        """
    )


def main():
    """Launch the web interface."""
    initialize()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
