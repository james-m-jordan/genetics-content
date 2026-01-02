# Genetics Tutor

An AI-powered genetics tutor that uses RAG (Retrieval Augmented Generation) to answer questions based on open-source genetics textbooks.

## Quick Start

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/james-m-jordan/genetics-content.git
cd genetics-content

# Build the vector store (one-time, ~2 min)
uv run python rag_chatbot/ingest.py

# Start chatting!
uv run python rag_chatbot/chat.py
```

On first run, you'll be prompted to enter your Anthropic API key. Get one at [console.anthropic.com](https://console.anthropic.com/api-keys).

## What's Included

### Knowledge Base (9MB, 142k lines)
Extracted text from 4 open-source genetics textbooks:

| Source | License | Content |
|--------|---------|---------|
| [OpenStax Biology 2e](https://openstax.org/books/biology-2e) | CC BY 4.0 | Full biology textbook + genetics unit |
| [Online Open Genetics](https://bio.libretexts.org/Bookshelves/Genetics/Online_Open_Genetics) | CC BY-SA 3.0 | Nickle & Barrette-Ng textbook |
| [NIGMS "The New Genetics"](https://www.nigms.nih.gov/) | Public Domain | Beginner-friendly overview |
| Schleif "Genetics and Molecular Biology" | Educational | Advanced molecular genetics |

### RAG Chatbot
- **Embeddings**: `all-MiniLM-L6-v2` (runs locally, no API needed)
- **Vector Store**: ChromaDB (persistent, ~120MB)
- **LLM**: Claude Sonnet 4 via Anthropic API
- **Interface**: Rich terminal UI with markdown

## Commands

```bash
# Build/rebuild the vector store
uv run python rag_chatbot/ingest.py

# Start the chatbot
uv run python rag_chatbot/chat.py
```

## Configuration

### API Key
Three ways to provide your Anthropic API key:

1. **Interactive prompt** (easiest) - just run the chatbot and enter when prompted
2. **Environment variable**: `export ANTHROPIC_API_KEY=your-key`
3. **`.env` file**: Create `.env` with `ANTHROPIC_API_KEY=your-key`

### Device
Automatically detects:
- Apple Silicon → MPS (GPU acceleration)
- Other → CPU

## Project Structure

```
genetics-content/
├── extracted_text/          # Source textbook content
├── rag_chatbot/
│   ├── chat.py              # Main chatbot
│   ├── ingest.py            # Vector store builder
│   └── vector_store/        # ChromaDB (generated)
├── pyproject.toml           # uv/pip dependencies
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
