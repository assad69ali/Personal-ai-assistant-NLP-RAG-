import os
import json
from pathlib import Path

#  Correct path to your actual dataset
BASE_DIR = Path(r"D:\project ir\dataset for ir")

#  Output directory
OUTPUT_DIR = Path(r"D:\project ir")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "indexed_metadata.json"

# Mapping top-level folders to metadata (normalized to lowercase)
SOURCE_CATEGORY_MAP = {
    "ucp": ("academic", "semester_study"),
    "coding files": ("academic", "code_assignments"),
    "py online class": ("teaching", "python_teaching"),
    "projects": ("freelance", "project"),
    "course + books material": ("personal", "learning_material"),
}

# Target file types
TEXT_EXTENSIONS = {'.txt', '.py', '.ipynb', '.pdf', '.docx', '.cpp', '.h'}

indexed_files = []

print(f" Scanning files under: {BASE_DIR}\n")

for root, _, files in os.walk(BASE_DIR):
    for file in files:
        full_path = Path(root) / file
        ext = full_path.suffix.lower()

        if ext in TEXT_EXTENSIONS:
            rel_path = full_path.relative_to(BASE_DIR)
            parts = rel_path.parts
            top_folder = parts[0].lower().strip() if parts else "unknown"

            # Match folder to source/category
            source, category = SOURCE_CATEGORY_MAP.get(top_folder, ("unknown", "uncategorized"))

            indexed_files.append({
                "filename": file,
                "path": str(rel_path).replace("\\", "/"),
                "file_type": "code" if ext in {'.py', '.ipynb', '.cpp', '.h'} else "document",
                "extension": ext,
                "source": source,
                "category": category
            })

            print(f" Indexed: {rel_path}")

# Save result to JSON
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(indexed_files, f, indent=2)

print(f"\n Total indexed: {len(indexed_files)} files")
print(f" Saved metadata to: {OUTPUT_PATH}")
