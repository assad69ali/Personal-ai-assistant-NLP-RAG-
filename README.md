# Personal Knowledge RAG Assistant

A Retrieval-Augmented Generation (RAG) system that indexes your personal files and answers natural language queries against them via a CLI. Uses semantic chunking, ChromaDB for vector storage, HuggingFace embeddings, and a cross-encoder for result reranking.

## Use Case

Knowledge workers, researchers, and students who want to query their own documents (PDFs, Word files, text) using natural language — getting precise, sourced answers instead of manually searching through files.

## Features

- Indexes any local files into a ChromaDB vector store
- Semantic chunking (splits by meaning, not just fixed character windows)
- Embeds chunks using `all-mpnet-base-v2` (HuggingFace Sentence Transformers)
- Retrieves top-K candidates via cosine similarity
- Reranks results with `cross-encoder/ms-marco-MiniLM-L-12-v2` for higher precision
- Generates a final natural language answer via LLM (OpenAI)
- Interactive CLI — ask questions, get sourced answers in the terminal

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.9+ |
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace `all-mpnet-base-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| LLM | OpenAI API (GPT-4 / GPT-3.5) |
| Hardware | CUDA GPU recommended; CPU fallback supported |

## Prerequisites

- Python 3.9 or higher
- An **OpenAI API key** — get one at [platform.openai.com](https://platform.openai.com)
- CUDA-compatible GPU (recommended for faster embedding; CPU works but is slower)

## Installation

```bash
pip install langchain chromadb sentence-transformers torch openai langchain-community
```

## Configuration

Set your OpenAI API key as an environment variable:

```bash
# Windows
set OPENAI_API_KEY=your_key_here

# Linux / macOS
export OPENAI_API_KEY=your_key_here
```

## Running

**Step 1 — Index your files** (run once, or when files change):
```bash
python "0010/project code/file_indexer.py"
```

**Step 2 — Start the CLI assistant:**
```bash
python "0010/project code/cli_rag_assistant.py"
```

Type a question at the prompt and press Enter to get an answer sourced from your indexed files.

## How It Works

| Step | Module | Description |
|---|---|---|
| 1. Index | `file_indexer.py` | Scans local files, extracts text content |
| 2. Chunk | `semantic_chunker.py` | Splits documents into meaningful semantic segments |
| 3. Embed | `semantic_chroma_indexer.py` | Embeds chunks with `all-mpnet-base-v2`, stores in ChromaDB |
| 4. Retrieve | `search_chroma.py` | Cosine similarity search for top-K candidates |
| 5. Rerank | `rerank_results.py` | Cross-encoder reranks candidates by query relevance |
| 6. Generate | `rag_generator.py` | LLM generates final answer from reranked context |
| 7. CLI | `cli_rag_assistant.py` | Ties all steps together as an interactive terminal app |

## Project Structure

```
├── 0010/
│   └── project code/
│       ├── cli_rag_assistant.py        # Main CLI interface (run this)
│       ├── content_extractor_v2.py     # Extracts text from various file types
│       ├── file_indexer.py             # Indexes files into ChromaDB (run first)
│       ├── rag_generator.py            # LLM generation module
│       ├── rerank_results.py           # Cross-encoder reranking
│       ├── search_chroma.py            # ChromaDB search interface
│       ├── semantic_chroma_indexer.py  # Semantic chunking + indexing
│       └── semantic_chunker.py         # Splits documents into semantic chunks
├── 0010/
│   ├── Personal_Knowledge_Retrieval_Assistant - 0010.pdf
│   └── project presention.pptx
├── Personal_Knowledge_Retrieval_Assistant - 0010.docx
└── Personal_Knowledge_Retrieval_Assistant - 0010.pdf
```

## Output & Results

- CLI prompt returns a natural language answer with the context passages it used
- ChromaDB index is persisted to disk — subsequent queries don't require re-indexing

## Notes

- The OpenAI API key must be set as an environment variable — do **not** hardcode it in source files
- First-time embedding downloads the `all-mpnet-base-v2` model (~420 MB) from HuggingFace
- GPU acceleration significantly speeds up the embedding step for large document collections
