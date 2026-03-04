import os
import json
import nltk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

nltk.download('punkt')

# Load model with CUDA
model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

def split_into_sentences(text):
    return nltk.sent_tokenize(text)

def semantic_chunk(text, min_chunk_size=200, max_chunk_size=800):
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            if len(current_chunk.strip()) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if len(current_chunk.strip()) >= min_chunk_size:
        chunks.append(current_chunk.strip())

    return chunks

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_file(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    input_path = "enriched_with_text.json"
    output_path = "semantic_chunks.json"

    documents = load_json_file(input_path)
    all_chunks = []

    for doc in tqdm(documents, desc="Semantic Chunking"):
        text = doc.get("text", "")
        if not text.strip():
            continue

        chunks = semantic_chunk(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk": chunk,
                "chunk_index": i,
                "metadata": doc.get("metadata", {})
            })

    save_json_file(all_chunks, output_path)
    print(f" Done. Saved {len(all_chunks)} semantic chunks to {output_path}")

if __name__ == "__main__":
    main()
