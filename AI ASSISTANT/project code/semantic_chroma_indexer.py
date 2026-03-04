import json
import uuid
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from semantic_chunker import semantic_chunk

# === Paths ===
INPUT_PATH = Path(r"D:\project ir\enriched_with_text.json")
CHROMA_DIR = Path(r"D:\project ir\chroma_db_semantic")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# === Load data ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    files = json.load(f)

# === Setup ChromaDB and embedder ===
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(name="rag-files-semantic")
embedder = SentenceTransformer("all-mpnet-base-v2", device="cuda")


chunk_total = 0

for file in files:
    text = file.get("text", "")
    if not text.strip():
        continue

    try:
        chunks = semantic_chunk(text)
    except Exception as e:
        print(f" Skipped {file['path']} due to chunking error: {e}")
        continue

    if not chunks:
        continue

    embeddings = embedder.encode(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]

    metadatas = [{
        "filename": file["filename"],
        "path": file["path"],
        "source": file["source"],
        "category": file["category"],
        "extension": file["extension"],
        "file_type": file["file_type"],
        "chunk_index": i
    } for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Indexed: {file['path']} ({len(chunks)} semantic chunks)")
    chunk_total += len(chunks)

print(f"\n Total semantic chunks indexed: {chunk_total}")
print(f" ChromaDB saved to: {CHROMA_DIR}")
