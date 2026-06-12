"""
Seed Knowledge Base
────────────────────
Reads documents from data/knowledge_base/ and indexes them into Qdrant
using the Haystack indexing pipeline.

Run once before starting the server:
  python -m scripts.seed_knowledge_base

Supports: .txt, .md files
Add your own documents to data/knowledge_base/ before running.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from haystack import Document
from src.rag_pipeline import build_indexing_pipeline

DOCS_DIR = Path("./data/knowledge_base")
SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents() -> list[Document]:
    documents = []
    for file_path in DOCS_DIR.rglob("*"):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                documents.append(
                    Document(
                        content=content,
                        meta={
                            "source": file_path.name,
                            "file_path": str(file_path),
                        },
                    )
                )
                print(f"  Loaded: {file_path.name} ({len(content)} chars)")
    return documents


def main():
    print("=== Seeding Knowledge Base into Qdrant ===\n")

    documents = load_documents()
    if not documents:
        print("No documents found in data/knowledge_base/")
        print("Add .txt or .md files there and re-run this script.")
        return

    print(f"\nIndexing {len(documents)} document(s)...\n")
    pipeline = build_indexing_pipeline()

    result = pipeline.run({"cleaner": {"documents": documents}})
    written = result.get("writer", {}).get("documents_written", "?")
    print(f"\nDone. {written} chunks written to Qdrant (collection: knowledge_base)")


if __name__ == "__main__":
    main()
