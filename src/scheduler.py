"""
Meeting Scheduler
──────────────────
Handles meeting/appointment requests via Google Calendar API.

Flow:
  1. Extract meeting details from the message (LLM-based entity extraction)
  2. Check calendar availability for the requested time slot
  3. Create the calendar event if available
  4. Return a confirmation or suggest alternatives if slot is taken

Entity extraction output:
  {
    "title":       str,     e.g. "Project discussion"
    "date":        str,     e.g. "2026-04-10"
    "time":        str,     e.g. "14:00"
    "duration":    int,     minutes, default 60
    "attendee":    str,     email or name of the other person (optional)
    "description": str      optional notes
  }
"""

import json
import logging
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]
TIMEZONE = "Europe/Istanbul"

client = OpenAI(api_key=settings.openai_api_key)

EXTRACTION_SYSTEM_PROMPT = """You are an entity extractor for a meeting scheduling system.
Extract meeting details from the user's message and return a JSON object.
Also detect the language of the user's message.

Today's date is {today}. The current time is {now}. Timezone: Europe/Istanbul.
Use these to resolve relative expressions like "tomorrow", "next Monday", "in 10 minutes", "in an hour", etc.

Return ONLY this JSON structure (no extra text):
{{
  "title": "<meeting title or 'Meeting' if not specified>",
  "date": "<YYYY-MM-DD>",
  "time": "<HH:MM in 24h format>",
  "duration": <minutes as integer, default 60>,
  "attendee": "<name or email if mentioned, else null>",
  "description": "<any notes or agenda mentioned, else null>",
  "language": "<detected language: en, ar, or tr>"
}}

If no specific date/time is mentioned, return null for date and time.
"""


def _extract_meeting_details(message: str) -> dict:
    now_dt = datetime.now(ZoneInfo(TIMEZONE))
    today = now_dt.strftime("%Y-%m-%d (%A)")
    now = now_dt.strftime("%H:%M")
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": EXTRACTION_SYSTEM_PROMPT.format(today=today, now=now),
            },
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=150,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def _get_calendar_service():
    """Authenticate and return a Google Calendar API service."""
    creds = None
    token_path = settings.google_calendar_token
    credentials_path = settings.google_calendar_credentials

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def _check_availability(service, start_dt: datetime, end_dt: datetime) -> bool:
    """Returns True if the time slot is free on the calendar."""
    body = {
        "timeMin": start_dt.isoformat(),
        "timeMax": end_dt.isoformat(),
        "items": [{"id": settings.google_calendar_id}],
    }
    result = service.freebusy().query(body=body).execute()
    busy_slots = result["calendars"][settings.google_calendar_id]["busy"]
    return len(busy_slots) == 0


def _create_event(service, details: dict, start_dt: datetime, end_dt: datetime) -> str:
    """Create a calendar event and return the event link."""
    event = {
        "summary": details.get("title", "Meeting"),
        "description": details.get("description") or "",
        "start": {"dateTime": start_dt.isoformat(), "timeZone": TIMEZONE},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": TIMEZONE},
    }

    if details.get("attendee") and "@" in str(details["attendee"]):
        event["attendees"] = [{"email": details["attendee"]}]

    created = service.events().insert(
        calendarId=settings.google_calendar_id, body=event
    ).execute()

    return created.get("htmlLink", "")


def handle_scheduling_request(message: str) -> dict:
    """
    Parse the message, check calendar, and create an event if possible.

    Returns:
        {
            "reply": str,           confirmation or clarification message
            "event_created": bool,
            "event_link": str | None
        }
    """
    try:
        details = _extract_meeting_details(message)

        lang = details.get("language", "en")

        if not details.get("date") or not details.get("time"):
            clarify = {
                "ar": "يسعدني جدولة ذلك! هل يمكنك إخباري بالتاريخ والوقت المحددين؟",
                "tr": "Bunu planlamaktan memnuniyet duyarım! Tarih ve saati belirtir misiniz?",
            }
            return {
                "reply": clarify.get(lang, "I'd be happy to schedule that! Could you let me know the specific date and time you have in mind?"),
                "event_created": False,
                "event_link": None,
            }

        tz = ZoneInfo(TIMEZONE)
        start_dt = datetime.strptime(
            f"{details['date']} {details['time']}", "%Y-%m-%d %H:%M"
        ).replace(tzinfo=tz)
        duration_mins = details.get("duration") or 60
        end_dt = start_dt + timedelta(minutes=duration_mins)

        service = _get_calendar_service()
        is_free = _check_availability(service, start_dt, end_dt)

        if not is_free:
            alt_start = start_dt + timedelta(hours=1)
            alt_end = alt_start + timedelta(minutes=duration_mins)
            alt_time = alt_start.strftime("%I:%M %p")
            busy_msg = {
                "ar": f"هذا الموعد ({details['time']}) محجوز. هل يناسبك الساعة {alt_time} بدلاً من ذلك؟",
                "tr": f"Bu saat ({details['time']}) dolu. {alt_time} uygun olur mu?",
            }
            return {
                "reply": busy_msg.get(lang, f"That slot ({details['time']}) is already taken. Would {alt_time} work instead?"),
                "event_created": False,
                "event_link": None,
            }

        event_link = _create_event(service, details, start_dt, end_dt)
        formatted_time = start_dt.strftime("%A, %B %d at %I:%M %p")

        done_msg = {
            "ar": f"تم! جدولت \"{details['title']}\" يوم {formatted_time} ({duration_mins} دقيقة). {f'رابط الحدث: {event_link}' if event_link else ''}",
            "tr": f"Tamam! \"{details['title']}\" toplantısını {formatted_time} ({duration_mins} dk) olarak planladım. {f'Etkinlik linki: {event_link}' if event_link else ''}",
        }
        return {
            "reply": done_msg.get(lang, f"Done! I've scheduled \"{details['title']}\" for {formatted_time} ({duration_mins} min). {f'Event link: {event_link}' if event_link else ''}"),
            "event_created": True,
            "event_link": event_link,
            "event_end": end_dt.isoformat(),
            "language": lang,
        }

    except Exception as e:
        logger.error(f"Scheduling failed: {e}")
        return {
            "reply": (
                "I wasn't able to access the calendar right now. "
                "Please try again or book manually."
            ),
            "event_created": False,
            "event_link": None,
        }
