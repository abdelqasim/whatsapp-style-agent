"""
Self-RAG Inspired Adaptive Retrieval
══════════════════════════════════════
Novel contribution — directly inspired by Section 2.3 of the project proposal
(Asai et al., 2023 "Self-RAG: Learning to Retrieve, Generate and Critique").

Standard RAG always retrieves documents for every query. This is wasteful and
can introduce irrelevant context when retrieval is unnecessary — e.g., greetings
or general conversational messages that don't need external knowledge.

This module implements an LLM-based retrieval gate:
  BEFORE hitting Qdrant, the LLM decides whether the query actually requires
  external knowledge retrieval. If not, a direct LLM response is generated
  without touching the vector database.

Two components:
  1. retrieval_needed(question, history) → bool
       Decides whether to trigger the RAG pipeline.

  2. detect_tone(message) → "formal" | "semi_formal" | "casual"
       Classifies the tone/register of an incoming message so that
       style examples can be tone-matched during retrieval.
       Used by the style memory module.
"""

import json
import logging

from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.openai_api_key)


# ── Retrieval Gate ─────────────────────────────────────────────────────────────

_RETRIEVAL_GATE_PROMPT = """You are a retrieval decision system for a knowledge base assistant.

Your job is to decide whether the following user question requires looking up external
documents from a knowledge base to answer correctly.

Answer YES if the question:
- Asks about specific business policies, prices, hours, procedures, or rules
- Requires factual information that may be stored in documents
- References services, products, or processes that need to be verified

Answer NO if the question:
- Is general knowledge that any LLM already knows
- Is conversational, casual, or a follow-up to previous chat
- Can be answered from the conversation history provided
- Is a greeting, thank you, or social message

Conversation history:
{history}

Question: {question}

Respond with ONLY a JSON object: {{"needs_retrieval": true}} or {{"needs_retrieval": false}}
"""


def retrieval_needed(question: str, conversation_history: str = "") -> bool:
    """
    Decides whether the RAG pipeline should be triggered for this question.

    Returns True  → proceed with Qdrant retrieval
    Returns False → answer directly with LLM (no vector DB call)
    """
    try:
        response = _client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "user",
                    "content": _RETRIEVAL_GATE_PROMPT.format(
                        history=conversation_history or "None",
                        question=question,
                    ),
                }
            ],
            temperature=0,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        decision = result.get("needs_retrieval", True)
        logger.info(f"Retrieval gate: {'RETRIEVE' if decision else 'SKIP'} for: {question[:60]}")
        return bool(decision)

    except Exception as e:
        logger.error(f"Retrieval gate failed: {e}. Defaulting to retrieve=True.")
        return True  # safe default: always retrieve on failure


# ── Tone Detector ──────────────────────────────────────────────────────────────

_TONE_DETECTION_PROMPT = """Classify the communication tone of the following WhatsApp message.
Return your answer as JSON.

Choose exactly one:
- "formal"      → professional, structured, uses formal vocabulary, polite distance
- "semi_formal" → friendly but professional, business-casual, measured warmth
- "casual"      → relaxed, colloquial, informal, may use contractions or slang

Message: "{message}"

Respond with ONLY: {{"tone": "formal"}} or {{"tone": "semi_formal"}} or {{"tone": "casual"}}
"""


def detect_tone(message: str) -> str:
    """
    Detect the tone/register of an incoming message.
    Used to select style memory examples that match the sender's communication style.

    Returns: "formal" | "semi_formal" | "casual"
    """
    try:
        response = _client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {
                    "role": "user",
                    "content": _TONE_DETECTION_PROMPT.format(message=message),
                }
            ],
            temperature=0,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        tone = result.get("tone", "semi_formal")
        if tone not in ("formal", "semi_formal", "casual"):
            tone = "semi_formal"
        return tone

    except Exception as e:
        logger.error(f"Tone detection failed: {e}")
        return "semi_formal"
