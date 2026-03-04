import os
import json
import subprocess
import logging
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import torch

# === Logging Setup ===
logging.basicConfig(
    filename="rag_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# === Configuration ===
CHROMA_DIR = r"D:\project ir\chroma_db_semantic"
COLLECTION_NAME = "rag-files-semantic"
EMBED_MODEL = "all-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
TOP_K_RETRIEVE = 30
TOP_K_FINAL = 10
SCORE_THRESHOLD = 2.0
GENERATOR_PATH = r"D:\project ir\project code\rag_generator.py"

# === Load Models ===
logging.info("Loading Chroma collection and models.")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

embedder = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda"}
)

reranker = CrossEncoder(CROSS_ENCODER_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

# === Retrieval ===
def retrieve(query, top_k=TOP_K_RETRIEVE, filters=None):
    logging.info(f"Retrieving with query: {query} | filters: {filters}")
    try:
        embedding = embedder.embed_query(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters if filters else None
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return list(zip(docs, metas))
    except Exception as e:
        logging.error(f"Error in retrieval: {str(e)}")
        return []

# === Reranking ===
def rerank(query, docs, top_n=TOP_K_FINAL):
    if not docs:
        logging.warning("No documents retrieved — skipping reranking.")
        return []

    logging.info(f"Reranking {len(docs)} documents.")
    pairs = [(query, doc[0]) for doc in docs]
    scores = reranker.predict(pairs)

    filtered = [(doc, score) for doc, score in zip(docs, scores) if score >= SCORE_THRESHOLD]
    ranked = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]

    return ranked  # Return (doc, score) tuples


# === Saving ===
def save_chunks(query, ranked_results, file_path="rag_reranked.json"):
    data = [{
        "rank": i+1,
        "score": float(score),
        "chunk": doc[0],
        "metadata": doc[1]
    } for i, (doc, score) in enumerate(ranked_results)]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"query": query, "chunks": data}, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved {len(data)} reranked chunks with scores to {file_path}")
    print(f" Saved reranked chunks to {file_path}")


# === Generation ===
def generate_answer():
    print(" Generating answer...")
    try:
        result = subprocess.run(["python", GENERATOR_PATH], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Generation script failed: {result.stderr}")
            print(" Generation Error:", result.stderr)
        else:
            print("\n Answer:")
            print(result.stdout.strip())
    except Exception as e:
        logging.error(f"Error running generator: {str(e)}")
        print(" Failed to run generator.")

# === CLI Interface ===
if __name__ == "__main__":
    print("🔎 One-Stop RAG Pipeline Interface")
    while True:
        query = input("\n🔍 Enter your question (or 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        # Optional filters
        filters = {}
        use_filters = input(" Do you want to use metadata filters? (y/n): ").strip().lower()
        if use_filters == "y":
            for key in ["source", "category", "filename"]:
                value = input(f"Enter filter value for '{key}' (or press Enter to skip): ").strip()
                if value:
                    filters[key] = value

        docs = retrieve(query, filters=filters)
        reranked = rerank(query, docs)

        if not reranked:
            print(" No high-quality results available for generation.")
            continue

        save_chunks(query, reranked)
        generate_answer()
