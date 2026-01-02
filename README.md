# Genetics Tutor

An AI-powered genetics tutor that uses RAG (Retrieval Augmented Generation) to answer questions based on open-source genetics textbooks.

## Quick Start

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/james-m-jordan/genetics-content.git
cd genetics-content

# Set your API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# Build the vector store (one-time, ~2 min)
uv run python rag_chatbot/ingest.py

# Launch web interface for students
uv run python rag_chatbot/web.py
```

Then open **http://localhost:7860** in your browser.

Get an API key at [console.anthropic.com](https://console.anthropic.com/api-keys).

## Interfaces

### Web UI (for students)
```bash
uv run python rag_chatbot/web.py
```
Opens a Gradio chat interface at http://localhost:7860 with:
- Clean, mobile-friendly design
- Example questions to get started
- Conversation history
- Source citations

### Terminal (for power users)
```bash
uv run python rag_chatbot/chat.py
```
Rich terminal UI with markdown rendering.

## What's Included

### Knowledge Base (9MB, 142k lines)
Extracted text from 4 open-source genetics textbooks:

| Source | License | Content |
|--------|---------|---------|
| [OpenStax Biology 2e](https://openstax.org/books/biology-2e) | CC BY 4.0 | Full biology textbook + genetics unit |
| [Online Open Genetics](https://bio.libretexts.org/Bookshelves/Genetics/Online_Open_Genetics) | CC BY-SA 3.0 | Nickle & Barrette-Ng textbook |
| [NIGMS "The New Genetics"](https://www.nigms.nih.gov/) | Public Domain | Beginner-friendly overview |
| Schleif "Genetics and Molecular Biology" | Educational | Advanced molecular genetics |

### RAG Pipeline
- **Embeddings**: `all-MiniLM-L6-v2` (runs locally, no API needed)
- **Vector Store**: ChromaDB (persistent, ~120MB)
- **LLM**: Claude Sonnet 4 via Anthropic API
- **Web UI**: Gradio

## Commands

```bash
# Build/rebuild the vector store
uv run python rag_chatbot/ingest.py

# Launch web interface
uv run python rag_chatbot/web.py

# Terminal chatbot
uv run python rag_chatbot/chat.py
```

## Configuration

### API Key
Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-...
```

Or set as environment variable:
```bash
export ANTHROPIC_API_KEY=your-key
```

### Device
Automatically detects:
- Apple Silicon → MPS (GPU acceleration)
- Other → CPU

## Project Structure

```
genetics-content/
├── extracted_text/          # Source textbook content
├── rag_chatbot/
│   ├── chat.py              # Terminal chatbot
│   ├── web.py               # Web interface (Gradio)
│   ├── ingest.py            # Vector store builder
│   └── vector_store/        # ChromaDB (generated)
├── pyproject.toml           # Dependencies
└── README.md
```

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Anthropic API key
- ~500MB disk space (embeddings + dependencies)

## License

- **Code**: MIT
- **Content**: See individual textbook licenses above (all OER/open-source)
