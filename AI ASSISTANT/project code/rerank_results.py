import json
import argparse
from sentence_transformers import CrossEncoder
import torch

def load_retrieved_chunks(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            chunks = data.get("chunks", [])
            query = data.get("query", None)

            if not query:
                raise ValueError("Missing 'query' key in input file.")
            return query, chunks

    except FileNotFoundError:
        print(f" Error: File not found -> {file_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f" Error: Invalid JSON in {file_path}")
        exit(1)
    except ValueError as e:
        print(f" Error: {str(e)}")
        exit(1)

def rerank_chunks(query, chunks, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2", top_k=15):
    print(f"\n Reranking {len(chunks)} chunks for query: \"{query}\"")

    cross_encoder = CrossEncoder(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    inputs = [(query, doc["chunk"]) for doc in chunks]
    scores = cross_encoder.predict(inputs)

    for doc, score in zip(chunks, scores):
        doc["score"] = float(score)

    # Filter chunks with score >= 2.0 before top_k
    filtered = [doc for doc in chunks if doc["score"] >= 2.0]
    return sorted(filtered, key=lambda x: x["score"], reverse=True)[:top_k]

def save_reranked_results(reranked, query, output_file):
    result = {
        "query": query,
        "chunks": reranked
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n Saved reranked top {len(reranked)} chunks to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="retrieved.json", help="Input file with retrieved chunks and query")
    parser.add_argument("--output", type=str, default="rag_reranked.json", help="Output file to save reranked chunks")
    parser.add_argument("--top_k", type=int, default=15, help="Number of top reranked chunks to retain")
    args = parser.parse_args()

    query, chunks = load_retrieved_chunks(args.input)
    reranked = rerank_chunks(query, chunks, top_k=args.top_k)
    save_reranked_results(reranked, query, args.output)

if __name__ == "__main__":
    main()
