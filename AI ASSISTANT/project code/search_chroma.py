import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

# ChromaDB path
CHROMA_DIR = r"D:\project ir\chroma_db_semantic"
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name="rag-files-semantic")

# Load embedding model on GPU
embedder = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2",
    model_kwargs={"device": "cuda"}
)

def search(query: str, top_k=30, filters=None):
    embedding = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=filters if filters else None
    )
    return results

def prompt_filters():
    print("\n Optional: Apply Filters")
    apply = input("Do you want to apply filters? (y/n): ").strip().lower()
    if apply != 'y':
        return None

    filters = {}
    filename = input("Filter by filename (leave blank to skip): ").strip()
    if filename:
        filters["filename"] = filename

    source = input("Filter by source (leave blank to skip): ").strip()
    if source:
        filters["source"] = source

    category = input("Filter by category (leave blank to skip): ").strip()
    if category:
        filters["category"] = category

    extension = input("Filter by file extension (e.g., .pdf) (leave blank to skip): ").strip()
    if extension:
        filters["extension"] = extension

    return filters if filters else None

def print_save_results(results, query):
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        print(" No results found.")
        return

    output_chunks = []
    print(f"\n Top {len(docs)} Results:\n" + "-" * 50)
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        print(f"\nResult {i}:")
        print(f" File: {meta.get('filename', 'N/A')}")
        print(f" Path: {meta.get('path', 'N/A')}")
        print(f" Source: {meta.get('source', 'N/A')}, Category: {meta.get('category', 'N/A')}")
        print(f" Snippet: {doc[:400]}...")

        output_chunks.append({
            "chunk": doc,
            "metadata": meta,
            "rank": i,
            "score": meta.get("score", 0.0)
        })

    # Write to retrieved.json
    with open("retrieved.json", "w", encoding="utf-8") as f:
        json.dump({
            "query": query,
            "chunks": output_chunks
        }, f, indent=2, ensure_ascii=False)

    print("\n Saved retrieved results to retrieved.json")

# CLI interface
if __name__ == "__main__":
    print(" Semantic Search over ChromaDB")
    print("Type your query or 'exit' to quit.")

    while True:
        q = input("\n Enter your query: ").strip()
        if q.lower() == "exit":
            break
        user_filters = prompt_filters()
        results = search(q, top_k=30, filters=user_filters)
        print_save_results(results, q)
