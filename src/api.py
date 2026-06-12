"""
FastAPI — AI Inference Service
────────────────────────────────
This is the Python AI backend. It is NOT the webhook receiver.
n8n receives the WhatsApp webhook, orchestrates the flow, and calls
these endpoints for AI inference.

Architecture:
  Meta → n8n (webhook) → Python backend (AI) → n8n → WhatsApp reply

Endpoints:
  GET  /webhook/verify       — Meta webhook verification (called by n8n webhook setup)
  POST /classify-intent      — Intent classification
  POST /rag-query            — RAG knowledge response
  POST /style-response       — Style-adaptive casual reply
  POST /schedule             — Google Calendar meeting booking
  POST /summarize            — Daily conversation summary
  GET  /health               — Health check
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from src.config import settings
from src.dashboard import router as dashboard_router
from src.conversation_memory import add_turn
from src.intent_classifier import classify_intent
from src.rag_pipeline import rag_service, build_indexing_pipeline
from src.scheduler import handle_scheduling_request
from src.style_memory import generate_style_response, index_style_examples, ensure_collection_exists
from src.followups import get_due_followups, get_pending_count, schedule_post_meeting_followup
from src.summarizer import generate_daily_summary, log_interaction
from src.audio_transcriber import handle_audio_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Startup: auto-index knowledge base & style examples ──────────────────────

def _seed_knowledge_base():
    """Index knowledge base documents into Qdrant on startup (if not already populated)."""
    try:
        from haystack import Document as HaystackDoc
        kb_dir = Path("./data/knowledge_base")
        if not kb_dir.exists():
            logger.warning("Knowledge base directory not found, skipping indexing")
            return

        docs = []
        for f in kb_dir.glob("*.txt"):
            text = f.read_text(encoding="utf-8")
            docs.append(HaystackDoc(content=text, meta={"source": f.name}))

        if not docs:
            logger.warning("No knowledge base documents found")
            return

        pipeline = build_indexing_pipeline()
        result = pipeline.run({"cleaner": {"documents": docs}})
        written = result.get("writer", {}).get("documents_written", 0)
        logger.info(f"[startup] Indexed {written} knowledge base chunks from {len(docs)} files")
    except Exception as e:
        logger.error(f"[startup] Knowledge base indexing failed: {e}")


def _seed_style_memory():
    """Index style examples into Qdrant on startup."""
    try:
        examples_file = Path("./data/style_examples/examples.json")
        if not examples_file.exists():
            logger.warning("Style examples file not found, skipping")
            return

        with open(examples_file, encoding="utf-8") as f:
            examples = json.load(f)

        if not examples:
            logger.warning("No style examples found")
            return

        index_style_examples(examples)
        logger.info(f"[startup] Indexed {len(examples)} style examples into Qdrant")
    except Exception as e:
        logger.error(f"[startup] Style memory indexing failed: {e}")


@asynccontextmanager
async def lifespan(app):
    """Startup: seed Qdrant with knowledge base and style examples."""
    logger.info("=" * 60)
    logger.info("WhatsApp Agent — Starting up")
    logger.info("=" * 60)
    _seed_knowledge_base()
    _seed_style_memory()
    logger.info("[startup] All indexes ready")
    yield
    logger.info("WhatsApp Agent — Shutting down")


app = FastAPI(
    title="WhatsApp Agent — AI Backend",
    description="Inference service for the style-adaptive WhatsApp conversational agent.",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(dashboard_router)


# ── Request / Response Models ─────────────────────────────────────────────────

class IntentRequest(BaseModel):
    message: str
    sender: str = "unknown"


class IntentResponse(BaseModel):
    intent: str
    confidence: float


class RAGRequest(BaseModel):
    question: str
    sender: str = "unknown"


class RAGResponse(BaseModel):
    answer: str
    grounded: bool
    sources: int
    retrieval_skipped: bool = False


class StyleRequest(BaseModel):
    message: str
    sender: str = "unknown"


class StyleResponse(BaseModel):
    reply: str
    tone: str
    examples_used: int


class ScheduleRequest(BaseModel):
    message: str
    sender: str = "unknown"


class ScheduleResponse(BaseModel):
    reply: str
    event_created: bool
    event_link: str | None = None


class LogRequest(BaseModel):
    sender: str
    incoming: str
    intent: str
    reply: str
    grounded: bool = True
    latency_ms: float = 0.0


class TranscribeRequest(BaseModel):
    media_id: str
    sender: str = "unknown"


class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration: float
    sender: str
    error: str | None = None


# ── Unified WhatsApp Webhook ─────────────────────────────────────────────────
# GET  → Meta verification challenge (handled here)
# POST → Forward to n8n webhook for orchestration

@app.get("/webhook/whatsapp", response_class=PlainTextResponse)
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        logger.info("Webhook verification successful")
        return hub_challenge
    logger.warning("Webhook verification failed — token mismatch")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


@app.post("/webhook/whatsapp")
async def forward_to_n8n(request: Request):
    body = await request.json()
    n8n_url = "http://n8n:5678/webhook/whatsapp"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(n8n_url, json=body)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        logger.error(f"Failed to forward to n8n: {e}")
        return JSONResponse(content={"status": "ok"}, status_code=200)


# ── AI Inference Endpoints (called by n8n) ────────────────────────────────────

@app.post("/classify-intent", response_model=IntentResponse)
async def classify_intent_endpoint(req: IntentRequest):
    """Classify the intent of an incoming message."""
    start = time.time()
    result = classify_intent(req.message)
    logger.info(
        f"[intent] {result['intent']} ({result['confidence']:.2f}) "
        f"in {(time.time()-start)*1000:.0f}ms"
    )
    return IntentResponse(**result)


@app.post("/rag-query", response_model=RAGResponse)
async def rag_query_endpoint(req: RAGRequest):
    """
    Knowledge-grounded response for business/factual queries.
    Includes Self-RAG gate and explicit grounding check.
    """
    start = time.time()
    result = rag_service.query(req.question, sender=req.sender)
    logger.info(
        f"[rag] grounded={result['grounded']} sources={result['sources']} "
        f"skipped={result.get('retrieval_skipped', False)} "
        f"in {(time.time()-start)*1000:.0f}ms"
    )
    return RAGResponse(**result)


@app.post("/style-response", response_model=StyleResponse)
async def style_response_endpoint(req: StyleRequest):
    """
    Style-adaptive casual chat response.
    Uses tone detection + tone-filtered few-shot retrieval from Qdrant.
    """
    start = time.time()
    result = generate_style_response(req.message, sender=req.sender)
    logger.info(
        f"[style] tone={result['tone']} examples={result['examples_used']} "
        f"in {(time.time()-start)*1000:.0f}ms"
    )
    return StyleResponse(**result)


@app.post("/schedule", response_model=ScheduleResponse)
async def schedule_endpoint(req: ScheduleRequest):
    """Meeting scheduling via Google Calendar."""
    start = time.time()
    result = handle_scheduling_request(req.message)
    if result["event_created"]:
        schedule_post_meeting_followup(
            sender=req.sender,
            meeting_title=result["reply"].split('"')[1] if '"' in result["reply"] else "Meeting",
            meeting_end_time=result.get("event_end", ""),
            language=result.get("language", "en"),
        )
    logger.info(
        f"[schedule] created={result['event_created']} "
        f"in {(time.time()-start)*1000:.0f}ms"
    )
    return ScheduleResponse(**result)


@app.post("/log-interaction")
async def log_interaction_endpoint(req: LogRequest):
    """
    Called by n8n after every interaction to persist the log and update
    conversation memory for the sender.
    """
    log_interaction(
        sender=req.sender,
        incoming=req.incoming,
        intent=req.intent,
        reply=req.reply,
        grounded=req.grounded,
        latency_ms=req.latency_ms,
    )
    add_turn(
        sender=req.sender,
        user_message=req.incoming,
        assistant_reply=req.reply,
    )
    return {"status": "logged"}


@app.post("/summarize")
async def summarize_endpoint():
    """Generate daily conversation summary (called by n8n daily cron workflow)."""
    return generate_daily_summary()


@app.post("/summarize-and-send")
async def summarize_and_send_endpoint():
    """Generate daily summary AND send it to admin via WhatsApp."""
    from datetime import date as _date

    result = generate_daily_summary()
    stats = result["stats"]
    summary = result["summary"]

    stats_text = (
        f"Total messages: {stats['total_messages']}\n"
        f"Unique senders: {stats['unique_senders']}\n"
        f"Avg response time: {stats['avg_latency_ms']}ms\n"
        f"Ungrounded responses: {stats['ungrounded_count']}"
    )
    message = f"*Daily Summary — {_date.today().strftime('%d/%m/%Y')}*\n\n{summary}\n\n*System Stats:*\n{stats_text}"

    # Send via WhatsApp API
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"https://graph.facebook.com/v19.0/{settings.whatsapp_phone_number_id}/messages",
            headers={
                "Authorization": f"Bearer {settings.whatsapp_token}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "to": settings.admin_phone_number,
                "type": "text",
                "text": {"body": message},
            },
        )

    sent = resp.status_code == 200
    logger.info(f"[daily-summary] sent={sent} status={resp.status_code}")
    return {"status": "sent" if sent else "failed", "message": message}


@app.get("/followups/due")
async def check_due_followups():
    """
    Returns follow-up messages that are due to be sent.
    Called by n8n on a periodic schedule (every 10 minutes).
    Each returned item has {sender, message, type, context}.
    """
    due = get_due_followups()
    logger.info(f"[followups] {len(due)} due, {get_pending_count()} pending")
    return {"followups": due, "count": len(due)}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_endpoint(req: TranscribeRequest):
    """
    Transcribe a WhatsApp voice note using OpenAI Whisper.
    Downloads the audio from Meta's CDN, transcribes it, and returns the text.
    The transcribed text can then be fed into the normal intent→response pipeline.
    Supports English, Arabic, Turkish, and 50+ other languages.
    """
    start = time.time()
    result = await handle_audio_message(req.media_id, req.sender)
    logger.info(
        f"[transcribe] lang={result.get('language')} "
        f"duration={result.get('duration', 0):.1f}s "
        f"text_len={len(result.get('text', ''))} "
        f"in {(time.time()-start)*1000:.0f}ms"
    )
    return TranscribeResponse(**result)


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "2.0.0"}
