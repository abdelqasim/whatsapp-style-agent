"""
Seed Style Memory
──────────────────
Embeds and stores style examples into Qdrant (style_memory collection).

Style examples are loaded from data/style_examples/examples.json.

Run once before starting the server:
  python -m scripts.seed_style_memory

To add YOUR own style: edit data/style_examples/examples.json with real
message/reply pairs exported from your WhatsApp chat history.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.style_memory import index_style_examples

EXAMPLES_FILE = Path("./data/style_examples/examples.json")


def main():
    print("=== Seeding Style Memory into Qdrant ===\n")

    if not EXAMPLES_FILE.exists():
        print(f"File not found: {EXAMPLES_FILE}")
        print("Create it following the format in data/style_examples/examples.json")
        return

    with open(EXAMPLES_FILE, encoding="utf-8") as f:
        examples = json.load(f)

    print(f"Found {len(examples)} style examples\n")
    for i, ex in enumerate(examples[:5]):
        print(f"  [{ex.get('tone', '?')}] \"{ex['incoming'][:50]}...\"")
    if len(examples) > 5:
        print(f"  ... and {len(examples) - 5} more\n")

    index_style_examples(examples)
    print(f"\nDone. {len(examples)} examples indexed into Qdrant (collection: style_memory)")


if __name__ == "__main__":
    main()
