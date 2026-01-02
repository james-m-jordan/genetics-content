#!/usr/bin/env python3
"""
Ingest genetics textbook content into ChromaDB vector store.
Uses sentence-transformers for local embeddings (no API key needed).
"""

import platform
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Paths
CONTENT_DIR = Path(__file__).parent.parent
EXTRACTED_TEXT_DIR = CONTENT_DIR / "extracted_text"
VECTOR_STORE_DIR = Path(__file__).parent / "vector_store"

# Embedding model - runs locally, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_device() -> str:
    """Detect the best available device for embeddings."""
    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "mps"  # Apple Silicon
    return "cpu"

def load_documents() -> list[Document]:
    """Load all text files from extracted_text directory."""
    documents = []

    console.print("[bold blue]Loading documents...[/]")

    for txt_file in EXTRACTED_TEXT_DIR.glob("*.txt"):
        console.print(f"  Loading: {txt_file.name}")
        content = txt_file.read_text(encoding="utf-8", errors="ignore")

        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "source": txt_file.name,
                "file_path": str(txt_file),
            }
        )
        documents.append(doc)

    console.print(f"[green]Loaded {len(documents)} documents[/]")
    return documents

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for better retrieval."""
    console.print("[bold blue]Chunking documents...[/]")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    console.print(f"[green]Created {len(chunks)} chunks[/]")
    return chunks

def create_vector_store(chunks: list[Document]) -> Chroma:
    """Create ChromaDB vector store with embeddings."""
    console.print("[bold blue]Creating embeddings and vector store...[/]")
    console.print(f"  Using model: {EMBEDDING_MODEL}")
    console.print("  (This may take a few minutes on first run)")

    # Use local embeddings - no API key needed
    device = get_device()
    console.print(f"  Using device: {device}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Create vector store
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Embedding chunks...", total=None)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name="genetics_knowledge",
        )

    console.print(f"[green]Vector store created at: {VECTOR_STORE_DIR}[/]")
    return vector_store

def main():
    console.print("[bold magenta]═══ Genetics RAG Ingestion ═══[/]\n")

    # Load documents
    documents = load_documents()

    if not documents:
        console.print("[red]No documents found![/]")
        return

    # Chunk documents
    chunks = chunk_documents(documents)

    # Create vector store
    vector_store = create_vector_store(chunks)

    # Test query
    console.print("\n[bold blue]Testing retrieval...[/]")
    results = vector_store.similarity_search("What is DNA?", k=2)
    for i, doc in enumerate(results):
        console.print(f"\n[yellow]Result {i+1}:[/] (from {doc.metadata['source']})")
        console.print(doc.page_content[:300] + "...")

    console.print("\n[bold green]✓ Ingestion complete! Run chat.py to start chatting.[/]")

if __name__ == "__main__":
    main()
