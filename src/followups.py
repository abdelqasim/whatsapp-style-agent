"""
Proactive Follow-ups
─────────────────────
Schedules and manages follow-up messages to send after events like meetings,
unanswered questions, or idle conversations.

Storage: JSON file at data/followups.json
Each entry: {sender, message, scheduled_at, sent, type, language}

Types:
  - post_meeting: "How did your meeting go?" sent 30 min after meeting end
  - check_in: daily check-in for active senders who haven't messaged today
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

TIMEZONE = "Europe/Istanbul"
FOLLOWUPS_FILE = Path("./data/followups.json")
FOLLOWUPS_FILE.parent.mkdir(parents=True, exist_ok=True)

_client = OpenAI(api_key=settings.openai_api_key)


def _load_followups() -> list[dict]:
    if FOLLOWUPS_FILE.exists():
        try:
            return json.loads(FOLLOWUPS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_followups(followups: list[dict]):
    FOLLOWUPS_FILE.write_text(
        json.dumps(followups, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def schedule_followup(
    sender: str,
    followup_type: str,
    context: str,
    delay_minutes: int = 30,
    language: str = "en",
):
    followups = _load_followups()
    scheduled_at = (
        datetime.now(ZoneInfo(TIMEZONE)) + timedelta(minutes=delay_minutes)
    ).isoformat()

    followups.append({
        "sender": sender,
        "type": followup_type,
        "context": context,
        "language": language,
        "scheduled_at": scheduled_at,
        "sent": False,
    })
    _save_followups(followups)
    logger.info(f"Follow-up scheduled for {sender} at {scheduled_at} ({followup_type})")


def _generate_followup_message(entry: dict) -> str:
    lang = entry.get("language", "en")
    ftype = entry["type"]
    context = entry["context"]

    lang_instruction = {
        "ar": "Reply in Arabic.",
        "tr": "Reply in Turkish.",
        "en": "Reply in English.",
    }.get(lang, "Reply in English.")

    if ftype == "post_meeting":
        prompt = f"""You are a friendly WhatsApp assistant. The user had a meeting: "{context}".
Write a short, warm follow-up message asking how it went. Keep it to 1-2 sentences, casual and natural for WhatsApp.
{lang_instruction}"""
    elif ftype == "check_in":
        prompt = f"""You are a friendly WhatsApp assistant. The user ({context}) hasn't messaged today.
Write a short, warm check-in message. Keep it to 1 sentence, casual and natural for WhatsApp. Don't be pushy.
{lang_instruction}"""
    else:
        prompt = f"""You are a friendly WhatsApp assistant. Write a short follow-up about: "{context}".
Keep it to 1-2 sentences, casual and natural for WhatsApp.
{lang_instruction}"""

    response = _client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


def get_due_followups() -> list[dict]:
    followups = _load_followups()
    now = datetime.now(ZoneInfo(TIMEZONE))
    due = []

    for entry in followups:
        if entry["sent"]:
            continue
        scheduled = datetime.fromisoformat(entry["scheduled_at"])
        if now >= scheduled:
            message = _generate_followup_message(entry)
            due.append({
                "sender": entry["sender"],
                "message": message,
                "type": entry["type"],
                "context": entry["context"],
            })
            entry["sent"] = True

    _save_followups(followups)
    return due


def get_pending_count() -> int:
    followups = _load_followups()
    return sum(1 for f in followups if not f["sent"])


def schedule_post_meeting_followup(
    sender: str, meeting_title: str, meeting_end_time: str, language: str = "en"
):
    try:
        end_dt = datetime.fromisoformat(meeting_end_time)
        now = datetime.now(ZoneInfo(TIMEZONE))
        minutes_until_end = max(0, (end_dt - now).total_seconds() / 60)
        delay = int(minutes_until_end) + 30
    except (ValueError, TypeError):
        delay = 90

    schedule_followup(
        sender=sender,
        followup_type="post_meeting",
        context=meeting_title,
        delay_minutes=delay,
        language=language,
    )
