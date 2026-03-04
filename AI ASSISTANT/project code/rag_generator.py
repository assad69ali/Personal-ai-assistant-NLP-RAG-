# rag_generator_openai.py

import os
import json
from openai import OpenAI

# === Set your API key here or load from env ===
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # Set this in your environment or a .env file
)

MODEL = "gpt-3.5-turbo"  # You can also use "gpt-4o-mini" or "gpt-3.5-turbo"

def load_reranked_chunks(file_path="rag_reranked.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["query"], data["chunks"]

def build_prompt(query, chunks):
    context = ""
    for entry in chunks:
        label = f"[{entry['rank']}] {entry['metadata']['filename']} ({entry['metadata']['path']})"
        doc = entry["chunk"].strip()
        context += f"{label}:\n{doc}\n\n"

    return f"""You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, reply with "I don't know."

Context:
{context}
Question: {query}
Answer:"""

def generate_with_openai(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based only on the context provided."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print(" RAG Generation using OpenAI API (v1.x)")

    try:
        query, chunks = load_reranked_chunks()
        prompt = build_prompt(query, chunks)

        print(" Generating answer from OpenAI...\n")
        answer = generate_with_openai(prompt)
        print(" Answer:\n" + answer)
    except Exception as e:
        print(f" Error: {e}")
