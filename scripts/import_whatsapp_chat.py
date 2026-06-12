"""
WhatsApp Chat History Importer — Novel Contribution
─────────────────────────────────────────────────────
Automatically builds the style memory dataset from a real exported WhatsApp
chat history file (.txt format from WhatsApp's "Export Chat" feature).

This directly solves the biggest gap in the proposal: the style memory needs
real examples of HOW YOU write, not hand-crafted synthetic ones. By parsing
your actual chat history, the system learns your genuine communication style.

How to export your WhatsApp chat:
  1. Open a WhatsApp conversation
  2. Tap ⋮ → More → Export Chat → Without Media
  3. Save the .txt file and pass it to this script

WhatsApp export format:
  [DD/MM/YYYY, HH:MM:SS] Name: Message     (iOS)
  DD/MM/YYYY, HH:MM - Name: Message         (Android)

Usage:
  python -m scripts.import_whatsapp_chat \\
    --file path/to/exported_chat.txt \\
    --your-name "Your Name" \\
    --min-pairs 20 \\
    --detect-tones

The script:
  1. Parses message/reply pairs where YOU replied
  2. Filters out system messages, media placeholders, links
  3. Optionally uses LLM to detect tone for each pair
  4. Saves to data/style_examples/examples.json (merges with existing)
  5. Seeds Qdrant automatically
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ── WhatsApp Export Parser ────────────────────────────────────────────────────

# Matches both iOS and Android WhatsApp export formats
_IOS_PATTERN = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\]\s+(.+?):\s+(.+)$",
    re.IGNORECASE,
)
_ANDROID_PATTERN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\s+-\s+(.+?):\s+(.+)$",
    re.IGNORECASE,
)

# Messages to skip (system messages, media, etc.)
_SKIP_PATTERNS = [
    re.compile(r"<Media omitted>", re.IGNORECASE),
    re.compile(r"image omitted", re.IGNORECASE),
    re.compile(r"video omitted", re.IGNORECASE),
    re.compile(r"audio omitted", re.IGNORECASE),
    re.compile(r"sticker omitted", re.IGNORECASE),
    re.compile(r"document omitted", re.IGNORECASE),
    re.compile(r"GIF omitted", re.IGNORECASE),
    re.compile(r"Voice message omitted", re.IGNORECASE),
    re.compile(r"null", re.IGNORECASE),
    re.compile(r"Messages and calls are end-to-end encrypted"),
    re.compile(r"created group"),
    re.compile(r"added"),
    re.compile(r"left"),
    re.compile(r"changed the group"),
    re.compile(r"You were added"),
]


def _should_skip(text: str) -> bool:
    if len(text.strip()) < 3:
        return True
    for pattern in _SKIP_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _parse_line(line: str) -> tuple[str, str] | None:
    """Parse a single line and return (sender, message) or None."""
    for pattern in (_IOS_PATTERN, _ANDROID_PATTERN):
        m = pattern.match(line.strip())
        if m:
            sender = m.group(2).strip()
            message = m.group(3).strip()
            return sender, message
    return None


def parse_chat_file(file_path: str) -> list[dict]:
    """
    Parse a WhatsApp export file into a list of {sender, message} dicts.
    Multi-line messages are joined correctly.
    """
    messages = []
    current = None

    with open(file_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            parsed = _parse_line(line)
            if parsed:
                if current:
                    messages.append(current)
                sender, text = parsed
                current = {"sender": sender, "message": text}
            elif current and line.strip():
                # Continuation of previous message
                current["message"] += " " + line.strip()

    if current:
        messages.append(current)

    return messages


def extract_style_pairs(
    messages: list[dict],
    your_name: str,
    min_length: int = 5,
    max_length: int = 300,
) -> list[dict]:
    """
    Extract (incoming_message, your_reply) pairs from the parsed messages.

    A valid pair: someone sends a message → you reply.
    Skips very short, very long, or flagged messages.
    """
    pairs = []

    for i in range(1, len(messages)):
        prev = messages[i - 1]
        curr = messages[i]

        # The current message must be from you
        if curr["sender"].lower() != your_name.lower():
            continue

        # The previous message must be from someone else
        if prev["sender"].lower() == your_name.lower():
            continue

        incoming = prev["message"].strip()
        reply = curr["message"].strip()

        if _should_skip(incoming) or _should_skip(reply):
            continue

        if len(incoming) < min_length or len(reply) < min_length:
            continue

        if len(incoming) > max_length or len(reply) > max_length:
            continue

        pairs.append({"incoming": incoming, "reply": reply, "tone": "semi_formal"})

    return pairs


def detect_tones_batch(pairs: list[dict]) -> list[dict]:
    """Use LLM to detect tone for each pair. Processes in batches to save API calls."""
    from src.self_rag import detect_tone

    print(f"\n  Detecting tones for {len(pairs)} pairs...")
    for i, pair in enumerate(pairs):
        # Detect tone based on the reply (your message, not theirs)
        pair["tone"] = detect_tone(pair["reply"])
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(pairs)}")

    return pairs


def merge_and_save(new_pairs: list[dict], output_path: Path):
    """Merge new pairs with existing examples.json and save."""
    existing = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    # Deduplicate by incoming message text
    existing_incoming = {ex["incoming"] for ex in existing}
    added = [p for p in new_pairs if p["incoming"] not in existing_incoming]
    merged = existing + added

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    return len(added), len(merged)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import WhatsApp chat history to build style memory"
    )
    parser.add_argument("--file", required=True, help="Path to WhatsApp exported .txt file")
    parser.add_argument(
        "--your-name", required=True,
        help="Your name exactly as it appears in the chat (e.g. 'Abdelrahman')"
    )
    parser.add_argument(
        "--min-pairs", type=int, default=10,
        help="Minimum number of pairs to extract (default: 10)"
    )
    parser.add_argument(
        "--detect-tones", action="store_true",
        help="Use LLM to detect tone for each pair (costs API calls, but improves quality)"
    )
    parser.add_argument(
        "--output", default="./data/style_examples/examples.json",
        help="Output JSON file (default: data/style_examples/examples.json)"
    )
    parser.add_argument(
        "--seed-qdrant", action="store_true",
        help="Automatically seed Qdrant after saving"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  WhatsApp Chat History Importer")
    print("=" * 60)

    # Parse
    print(f"\nParsing chat file: {args.file}")
    all_messages = parse_chat_file(args.file)
    print(f"  Total messages parsed: {len(all_messages)}")

    # Show senders found
    senders = sorted({m["sender"] for m in all_messages})
    print(f"  Participants found: {', '.join(senders[:10])}")

    # Extract pairs
    pairs = extract_style_pairs(all_messages, args.your_name)
    print(f"  Valid reply pairs extracted: {len(pairs)}")

    if len(pairs) < args.min_pairs:
        print(
            f"\nWARNING: Only {len(pairs)} pairs found (minimum requested: {args.min_pairs})."
        )
        print(
            f"  Make sure --your-name matches exactly: try one of {senders[:5]}"
        )
        if len(pairs) == 0:
            sys.exit(1)

    # Sample preview
    print("\n  Sample pairs:")
    for p in pairs[:3]:
        print(f"    Received: \"{p['incoming'][:60]}\"")
        print(f"    Replied:  \"{p['reply'][:60]}\"")
        print()

    # Detect tones
    if args.detect_tones:
        pairs = detect_tones_batch(pairs)
        tone_dist = {}
        for p in pairs:
            tone_dist[p["tone"]] = tone_dist.get(p["tone"], 0) + 1
        print(f"\n  Tone distribution: {tone_dist}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    added, total = merge_and_save(pairs, output_path)
    print(f"\n  Added {added} new pairs. Total in dataset: {total}")
    print(f"  Saved to: {output_path}")

    # Seed Qdrant
    if args.seed_qdrant:
        print("\n  Seeding Qdrant with updated dataset...")
        from src.style_memory import index_style_examples
        with open(output_path, encoding="utf-8") as f:
            all_examples = json.load(f)
        index_style_examples(all_examples)
        print(f"  Done. {len(all_examples)} examples in Qdrant.")

    print("\n" + "=" * 60)
    print("  Import complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
