"""
Style Memory Module
────────────────────
Generates style-adaptive responses for casual chat using few-shot retrieval
from a Qdrant collection of curated message/reply examples.

Improvements over the original version:
  1. Tone detection: the incoming message's tone is detected first (formal /
     semi_formal / casual), then Qdrant retrieves examples filtered to that tone.
     This ensures the few-shot examples actually match the communication register.

  2. Conversation memory: previous turns for the sender are injected into the
     prompt so the response is contextually aware of the ongoing conversation.

  3. Graceful fallback: if Qdrant has no style examples yet (collection empty),
     generates a friendly generic response instead of crashing.
"""

import logging

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from src.config import settings

logger = logging.getLogger(__name__)

_openai = OpenAI(api_key=settings.openai_api_key)
_qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

# ── Prompts ───────────────────────────────────────────────────────────────────

_STYLE_PROMPT = """You are a style-adaptive messaging assistant writing a WhatsApp reply.
Your PRIMARY goal is to MATCH the communication style of the conversation.

STYLE RULES:
1. LANGUAGE: Reply in the SAME LANGUAGE as the incoming message (Arabic→Arabic, Turkish→Turkish, English→English)
2. TONE: Match the detected tone — {tone_label}
   - casual: use contractions, slang, emojis, short sentences, informal vocabulary
   - semi_formal: friendly but professional, measured warmth, clear sentences
   - formal: structured, polished vocabulary, no contractions, respectful distance
3. LENGTH: Mirror the message length — short messages get short replies, detailed messages get detailed replies
4. VOCABULARY: Use similar word choices and expression patterns as shown in the examples
5. ENERGY: Match the emotional energy — excited→excited, calm→calm, concerned→supportive

Study these examples carefully and mirror the style EXACTLY:

{examples}

{history}
Now write a reply that perfectly matches the style above. Output ONLY the reply text — nothing else.

Incoming message: {incoming}

Reply:"""

_GENERIC_PROMPT = """You are a helpful, friendly assistant replying on WhatsApp.
Keep the reply concise and natural.
IMPORTANT: Always reply in the SAME LANGUAGE as the user's message. If they write in Arabic, reply in Arabic. If Turkish, reply in Turkish. If English, reply in English.

{history}
Message: {incoming}

Reply:"""


# ── Qdrant Helpers ────────────────────────────────────────────────────────────

def _get_embedding(text: str) -> list[float]:
    response = _openai.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )
    return response.data[0].embedding


def ensure_collection_exists():
    existing = [c.name for c in _qdrant.get_collections().collections]
    if settings.qdrant_style_collection not in existing:
        _qdrant.create_collection(
            collection_name=settings.qdrant_style_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {settings.qdrant_style_collection}")


def index_style_examples(examples: list[dict]):
    """
    Embed and upsert style examples into Qdrant.

    Each example: {"incoming": str, "reply": str, "tone": str}
    """
    ensure_collection_exists()
    points = []
    for i, ex in enumerate(examples):
        vector = _get_embedding(ex["incoming"])
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "incoming": ex["incoming"],
                    "reply": ex["reply"],
                    "tone": ex.get("tone", "semi_formal"),
                },
            )
        )
    _qdrant.upsert(collection_name=settings.qdrant_style_collection, points=points)
    logger.info(f"Indexed {len(points)} style examples into Qdrant")


def _retrieve_examples(incoming: str, tone: str) -> list[dict]:
    """
    Retrieve top-k style examples filtered by detected tone.
    Falls back to unfiltered search if tone-filtered results are insufficient.
    """
    query_vector = _get_embedding(incoming)

    # First: tone-filtered search
    tone_filter = Filter(
        must=[FieldCondition(key="tone", match=MatchValue(value=tone))]
    )
    results = _qdrant.search(
        collection_name=settings.qdrant_style_collection,
        query_vector=query_vector,
        query_filter=tone_filter,
        limit=settings.style_top_k,
        with_payload=True,
    )

    # If fewer than 2 tone-matched results, fall back to unfiltered
    if len(results) < 2:
        results = _qdrant.search(
            collection_name=settings.qdrant_style_collection,
            query_vector=query_vector,
            limit=settings.style_top_k,
            with_payload=True,
        )

    return [hit.payload for hit in results]


# ── Main Interface ────────────────────────────────────────────────────────────

def generate_style_response(incoming: str, sender: str = "unknown") -> dict:
    """
    Generate a style-adaptive reply for casual chat.

    Steps:
      1. Detect tone of incoming message
      2. Retrieve tone-matched style examples from Qdrant
      3. Load conversation history for sender
      4. Build few-shot prompt and generate reply

    Returns:
      {"reply": str, "tone": str, "examples_used": int}
    """
    from src.conversation_memory import format_history_for_prompt
    from src.self_rag import detect_tone

    try:
        # Step 1: detect tone
        tone = detect_tone(incoming)

        # Step 2: retrieve tone-matched examples
        examples = _retrieve_examples(incoming, tone)

        # Step 3: load conversation history
        history = format_history_for_prompt(sender)

        if not examples:
            return _generic_reply(incoming, history)

        # Step 4: build few-shot prompt
        formatted = "\n\n".join(
            f'Received: "{ex["incoming"]}"\nReplied:  "{ex["reply"]}"'
            for ex in examples
        )

        prompt = _STYLE_PROMPT.format(
            tone_label=tone,
            examples=formatted,
            history=history + "\n" if history else "",
            incoming=incoming,
        )

        response = _openai.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=250,
        )

        reply = response.choices[0].message.content.strip()
        return {"reply": reply, "tone": tone, "examples_used": len(examples)}

    except Exception as e:
        logger.error(f"Style response generation failed: {e}")
        from src.conversation_memory import format_history_for_prompt
        return _generic_reply(incoming, format_history_for_prompt(sender))


def _generic_reply(incoming: str, history: str) -> dict:
    """Fallback when no style examples are available."""
    prompt = _GENERIC_PROMPT.format(
        history=history + "\n" if history else "",
        incoming=incoming,
    )
    response = _openai.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    return {
        "reply": response.choices[0].message.content.strip(),
        "tone": "semi_formal",
        "examples_used": 0,
    }
