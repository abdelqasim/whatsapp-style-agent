"""
Daily Conversation Summarizer
───────────────────────────────
Generates a structured daily summary of all logged conversations.

The summary includes:
  - Total messages received
  - Intent distribution (casual / knowledge / scheduling)
  - Key topics discussed (knowledge queries)
  - Meetings scheduled
  - Any unresolved or low-confidence interactions

This module is triggered by the n8n daily summary workflow (scheduled cron job).
Conversation logs are stored in-memory in a simple append log during the day.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)

TIMEZONE = "Europe/Istanbul"
LOG_DIR = Path("./data/conversation_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─── In-memory log for current session ────────────────────────────────────────


def _load_existing_log() -> list[dict]:
    """Load today's log from disk on startup so we don't lose data across restarts."""
    today = date.today().isoformat()
    log_file = LOG_DIR / f"{today}.json"
    if log_file.exists():
        try:
            with open(log_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


_daily_log: list[dict] = _load_existing_log()


def log_interaction(
    sender: str,
    incoming: str,
    intent: str,
    reply: str,
    grounded: bool = True,
    latency_ms: float = 0.0,
):
    """Append an interaction to the in-memory daily log."""
    _daily_log.append({
        "timestamp": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "sender": sender,
        "incoming": incoming,
        "intent": intent,
        "reply": reply,
        "grounded": grounded,
        "latency_ms": latency_ms,
    })
    _persist_log()


def _persist_log():
    """Write the current log to a dated JSON file."""
    today = date.today().isoformat()
    log_file = LOG_DIR / f"{today}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(_daily_log, f, ensure_ascii=False, indent=2)


def _load_todays_log() -> list[dict]:
    """Load today's log from disk (used when summary is generated)."""
    today = date.today().isoformat()
    log_file = LOG_DIR / f"{today}.json"
    if log_file.exists():
        with open(log_file, encoding="utf-8") as f:
            return json.load(f)
    return _daily_log


SUMMARY_PROMPT = """You are summarizing a day's worth of WhatsApp conversations for a personal assistant.

Here is the conversation log for today:

{log_text}

Write a concise daily summary (suitable for WhatsApp) covering:
1. How many messages were received and from how many unique senders
2. Main topics/questions people asked about (knowledge queries)
3. Any meetings or appointments scheduled
4. Any unresolved or unclear requests
5. Any notable patterns

Keep it short and clear — bullet points are fine. Use a friendly but professional tone.
"""


def generate_daily_summary() -> dict:
    """
    Generate a daily summary from today's conversation log.

    Returns:
        {
            "summary": str,
            "stats": {
                "total_messages": int,
                "unique_senders": int,
                "intent_counts": dict,
                "avg_latency_ms": float,
                "ungrounded_count": int
            }
        }
    """
    log = _load_todays_log()

    if not log:
        return {
            "summary": "No conversations were logged today.",
            "stats": {
                "total_messages": 0,
                "unique_senders": 0,
                "intent_counts": {},
                "avg_latency_ms": 0.0,
                "ungrounded_count": 0,
            },
        }

    # Compute stats
    intent_counts = defaultdict(int)
    latencies = []
    ungrounded = 0
    senders = set()

    for entry in log:
        intent_counts[entry.get("intent", "unknown")] += 1
        if entry.get("latency_ms"):
            latencies.append(entry["latency_ms"])
        if not entry.get("grounded", True):
            ungrounded += 1
        senders.add(entry.get("sender", "unknown"))

    stats = {
        "total_messages": len(log),
        "unique_senders": len(senders),
        "intent_counts": dict(intent_counts),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "ungrounded_count": ungrounded,
    }

    # Format log for LLM
    log_text = "\n".join(
        [
            f"[{e['timestamp'][11:16]}] ({e['intent']}) From {e['sender']}: \"{e['incoming'][:100]}\""
            for e in log
        ]
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "user",
                    "content": SUMMARY_PROMPT.format(log_text=log_text),
                }
            ],
            temperature=0.4,
            max_tokens=400,
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        summary = (
            f"Daily stats: {stats['total_messages']} messages from "
            f"{stats['unique_senders']} senders. "
            f"Intents: {dict(intent_counts)}."
        )

    return {"summary": summary, "stats": stats}
