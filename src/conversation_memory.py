"""
Conversation Memory
────────────────────
Tracks per-sender conversation history so the agent can maintain context
across multiple message turns (stateful conversations).

Storage: JSON files in data/conversation_logs/memory/<sender>.json
Each file holds the last N turns for that sender.

Why this matters:
  Without memory, every message is treated as a fresh conversation.
  A user saying "Can you clarify that?" or "What about the second option?"
  would get a confused response. Memory fixes this.
"""

import json
import logging
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

MEMORY_DIR = Path("./data/conversation_logs/memory")
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _memory_file(sender: str) -> Path:
    # Sanitize sender ID to safe filename
    safe_sender = "".join(c if c.isalnum() else "_" for c in sender)
    return MEMORY_DIR / f"{safe_sender}.json"


def _load(sender: str) -> list[dict]:
    path = _memory_file(sender)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save(sender: str, history: list[dict]):
    try:
        _memory_file(sender).write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as e:
        logger.error(f"Failed to save memory for {sender}: {e}")


def add_turn(sender: str, user_message: str, assistant_reply: str):
    """Append a new exchange to this sender's history, trimming to max_turns."""
    history = _load(sender)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_reply})

    # Keep only the last max_turns * 2 messages (user + assistant per turn)
    max_messages = settings.memory_max_turns * 2
    if len(history) > max_messages:
        history = history[-max_messages:]

    _save(sender, history)


def get_history(sender: str) -> list[dict]:
    """Return the full conversation history for a sender as a list of {role, content}."""
    return _load(sender)


def format_history_for_prompt(sender: str) -> str:
    """
    Format conversation history as readable text for injecting into prompts.

    Returns empty string if no history exists.
    """
    history = _load(sender)
    if not history:
        return ""

    lines = ["--- Previous conversation ---"]
    for msg in history:
        prefix = "User:" if msg["role"] == "user" else "Assistant:"
        lines.append(f"{prefix} {msg['content']}")
    lines.append("--- End of previous conversation ---")
    return "\n".join(lines)


def clear_history(sender: str):
    """Clear all conversation history for a sender (e.g., after daily reset)."""
    path = _memory_file(sender)
    if path.exists():
        path.unlink()
