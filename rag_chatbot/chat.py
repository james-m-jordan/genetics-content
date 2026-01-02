#!/usr/bin/env python3
"""
Genetics RAG Chatbot - Interactive Q&A with genetics textbook knowledge.
Uses Claude API for responses and local embeddings for retrieval.
"""

import os
import platform
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()

# Paths
ROOT_DIR = Path(__file__).parent.parent
VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"
ENV_FILE = ROOT_DIR / ".env"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# System prompt for the genetics tutor
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

Be encouraging and supportive - you're helping students learn!"""


def get_device() -> str:
    """Detect the best available device for embeddings."""
    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "mps"  # Apple Silicon
    # Could add CUDA detection here
    return "cpu"


def get_api_key() -> str:
    """Get API key from environment, .env file, or prompt user."""
    # Load from .env if exists
    load_dotenv(ENV_FILE)

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        return api_key

    # Prompt user for API key
    console.print("[yellow]No Anthropic API key found.[/]")
    console.print("Get one at: [link=https://console.anthropic.com/api-keys]https://console.anthropic.com/api-keys[/]\n")

    api_key = Prompt.ask("[bold]Enter your Anthropic API key[/]", password=True)

    if not api_key:
        console.print("[red]API key is required.[/]")
        raise SystemExit(1)

    # Offer to save it
    if Confirm.ask("Save API key to .env file for future use?", default=True):
        ENV_FILE.write_text(f"ANTHROPIC_API_KEY={api_key}\n")
        console.print(f"[green]✓ Saved to {ENV_FILE}[/]\n")

    return api_key


def load_vector_store() -> Chroma:
    """Load the existing vector store."""
    if not VECTOR_STORE_DIR.exists():
        console.print("[red]Vector store not found![/]")
        console.print("Run: [bold]uv run genetics-ingest[/] to build it first.")
        raise SystemExit(1)

    device = get_device()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embeddings,
        collection_name="genetics_knowledge",
    )


def retrieve_context(vector_store: Chroma, query: str, k: int = 5) -> str:
    """Retrieve relevant context from the vector store."""
    results = vector_store.similarity_search(query, k=k)

    context_parts = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)


def chat_with_claude(client: Anthropic, messages: list, context: str, query: str) -> str:
    """Send query to Claude with retrieved context."""
    user_message = f"""Based on the following textbook content, please answer the student's question.

TEXTBOOK CONTENT:
{context}

STUDENT'S QUESTION:
{query}

Please provide a clear, educational response based on the textbook content above."""

    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    assistant_message = response.content[0].text
    messages.append({"role": "assistant", "content": assistant_message})

    return assistant_message


def main():
    console.print(Panel.fit(
        "[bold magenta]Genetics Tutor Chatbot[/]\n"
        "[dim]Powered by RAG + Claude[/]",
        border_style="magenta",
    ))

    # Get API key (prompts if not found)
    api_key = get_api_key()

    # Load vector store
    console.print("[dim]Loading knowledge base...[/]")
    vector_store = load_vector_store()
    console.print("[green]✓ Knowledge base loaded[/]\n")

    # Initialize Claude client
    client = Anthropic(api_key=api_key)
    messages = []

    console.print("Ask me anything about genetics! Type 'quit' to exit.\n")

    while True:
        try:
            query = Prompt.ask("[bold cyan]You[/]")

            if query.lower() in ("quit", "exit", "q"):
                console.print("[dim]Goodbye![/]")
                break

            if not query.strip():
                continue

            # Retrieve context
            with console.status("[dim]Searching textbooks...[/]"):
                context = retrieve_context(vector_store, query)

            # Get response from Claude
            with console.status("[dim]Thinking...[/]"):
                response = chat_with_claude(client, messages, context, query)

            # Display response
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold green]Genetics Tutor[/]",
                border_style="green",
            ))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/]")
            break


if __name__ == "__main__":
    main()
