"""
Intent Classifier
─────────────────
Classifies an incoming WhatsApp message into one of three categories:
  - casual_chat       → general conversation, greetings, small talk
  - knowledge_query   → questions about business info, FAQs, policies
  - scheduling        → meeting/appointment requests

Uses GPT-4 in constrained classification mode (structured output).
"""

import json
import logging
from enum import Enum

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.openai_api_key)


class Intent(str, Enum):
    CASUAL_CHAT = "casual_chat"
    KNOWLEDGE_QUERY = "knowledge_query"
    SCHEDULING = "scheduling"


CLASSIFICATION_SYSTEM_PROMPT = """You are an intent classifier for a WhatsApp conversational agent.
The message may be in English, Arabic, or Turkish. Classify it regardless of language.

Classify the user's message into EXACTLY ONE of these three categories:

1. casual_chat     — Greetings, small talk, general questions, personal conversations,
                     anything that does not require external knowledge or calendar access.
                     Examples: "Hey, how are you?", "What do you think about X?", "Thanks!"

2. knowledge_query — Questions about business information, policies, FAQs, product details,
                     pricing, rules, or any topic that requires looking up stored documents.
                     Examples: "What are your working hours?", "What is your return policy?",
                     "How do I reset my password?"

3. scheduling      — Requests to book, schedule, check, or cancel a meeting or appointment.
                     Examples: "Can we meet tomorrow at 3pm?", "Schedule a call for Monday",
                     "Are you free next week?"

Respond with a JSON object in this exact format:
{"intent": "<casual_chat|knowledge_query|scheduling>", "confidence": <0.0-1.0>}

Do not include any other text.
"""


def classify_intent(message: str) -> dict:
    """
    Classify the intent of an incoming message.

    Returns:
        {
            "intent": "casual_chat" | "knowledge_query" | "scheduling",
            "confidence": float
        }
    """
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Message: {message}"},
            ],
            temperature=0,       # deterministic for classification
            max_tokens=60,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        # Validate intent value
        intent_value = result.get("intent", "casual_chat")
        if intent_value not in [e.value for e in Intent]:
            logger.warning(f"Unexpected intent value '{intent_value}', defaulting to casual_chat")
            intent_value = Intent.CASUAL_CHAT.value

        return {
            "intent": intent_value,
            "confidence": float(result.get("confidence", 0.9)),
        }

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        # Safe fallback
        return {"intent": Intent.CASUAL_CHAT.value, "confidence": 0.5}
