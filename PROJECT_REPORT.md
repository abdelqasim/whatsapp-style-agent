# Style-Adaptive WhatsApp Conversational Agent — Project Report

## 1. Project Overview

This project implements a **multilingual, style-adaptive WhatsApp conversational agent** that combines intent classification, retrieval-augmented generation (RAG), tone-aware response styling, meeting scheduling, proactive follow-ups, and conversation memory into a unified system accessible through WhatsApp.

The agent serves as a personal assistant that can:
- Have natural casual conversations matching the user's communication style
- Answer knowledge-based questions from a curated document store
- Schedule meetings on Google Calendar via natural language
- Remember conversation context across multiple messages
- Proactively follow up after meetings
- Operate in English, Arabic, and Turkish
- Generate daily conversation summaries for the admin

---

## 2. System Architecture

```
User (WhatsApp) → Meta Cloud API → Cloudflare Tunnel → FastAPI Backend → n8n Workflow Engine
                                                                              ↓
                                                        ┌─────────────────────┼─────────────────────┐
                                                        ↓                     ↓                     ↓
                                                  Intent Classifier     RAG Pipeline          Style Memory
                                                        ↓                     ↓                     ↓
                                                  Route by Intent       Qdrant Vector DB      Qdrant Vector DB
                                                        ↓                     ↓                     ↓
                                                  ┌─────┼─────┐         Haystack 2.x          Tone Detection
                                                  ↓     ↓     ↓         + OpenAI LLM          + Few-shot
                                                RAG  Schedule Style                            Retrieval
                                                  ↓     ↓     ↓
                                                  └─────┼─────┘
                                                        ↓
                                                  WhatsApp Reply → User
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Messaging | WhatsApp Business Cloud API (Meta) | User-facing chat interface |
| Webhook Tunnel | Cloudflare Tunnel (cloudflared) | Exposes local server to the internet |
| AI Backend | FastAPI (Python) | All AI inference endpoints |
| Workflow Engine | n8n (self-hosted) | Orchestrates message flow and scheduling |
| Vector Database | Qdrant | Stores document embeddings and style examples |
| LLM | OpenAI GPT-4o-mini | Intent classification, response generation, entity extraction |
| Embeddings | text-embedding-3-small (OpenAI) | Document and query embedding (1536 dimensions) |
| RAG Framework | Haystack 2.x | Document indexing, retrieval, and generation pipeline |
| Calendar | Google Calendar API (OAuth 2.0) | Meeting creation and availability checking |
| Containerization | Docker Compose | Runs Qdrant, n8n, and the backend as services |
| Monitoring | Custom Dashboard (FastAPI + HTML) | Real-time system monitoring at /dashboard |

### Docker Services

The system runs three containerized services via `docker-compose.yml`:

1. **capstone_qdrant** — Qdrant vector database on ports 6333 (REST) and 6334 (gRPC)
2. **capstone_n8n** — n8n workflow automation on port 5678
3. **capstone_backend** — FastAPI Python backend on port 8000

---

## 3. Agent Capabilities

### 3.1 Intent Classification (`/classify-intent`)

**What it does:** Every incoming message is classified into one of three intents before routing.

**How it works:**
- Uses GPT-4o-mini with a structured classification prompt
- Returns a JSON object: `{"intent": "...", "confidence": 0.0-1.0}`
- Supports messages in English, Arabic, and Turkish
- Temperature set to 0 for deterministic classification
- Falls back to `casual_chat` with 0.5 confidence on any error

**Intent Categories:**
| Intent | Description | Example Messages |
|--------|-------------|-----------------|
| `casual_chat` | Greetings, small talk, general conversation | "Hey, how are you?", "مرحبا", "Merhaba" |
| `knowledge_query` | Questions requiring factual/business information | "What are your working hours?", "ما هي ساعات العمل؟" |
| `scheduling` | Meeting or appointment requests | "Schedule a meeting tomorrow at 3pm", "Yarın toplantı ayarla" |

**File:** `src/intent_classifier.py`

---

### 3.2 RAG Knowledge Pipeline (`/rag-query`)

**What it does:** Answers factual questions using a knowledge base stored in Qdrant, with a Self-RAG gate to skip retrieval when unnecessary.

**How it works (step by step):**

1. **Conversation Memory Load** — Retrieves the sender's recent conversation history (last 5 turns) for multi-turn context
2. **Self-RAG Gate** (`src/self_rag.py`) — An LLM-based decision function determines if the question actually needs document retrieval:
   - "What are your working hours?" → YES, needs retrieval
   - "Thanks for your help!" → NO, skip retrieval
3. **If retrieval NOT needed** → Generates a direct LLM answer without touching Qdrant (saves latency and cost)
4. **If retrieval IS needed:**
   - **Embed** the question using text-embedding-3-small
   - **Retrieve** top-5 similar documents from Qdrant's `knowledge_base` collection
   - **Generate** a grounded answer using the retrieved documents as context
   - The prompt explicitly instructs: "answer ONLY from the context documents"
5. **Grounding Check** — If no meaningful answer is generated, returns a safe fallback message instead of hallucinating
6. **Multilingual** — Responds in the same language as the question

**Self-RAG Innovation:** This is directly inspired by Asai et al. (2023) "Self-RAG: Learning to Retrieve, Generate and Critique." Standard RAG always retrieves, which is wasteful for casual messages. The Self-RAG gate prevents unnecessary vector DB queries and avoids injecting irrelevant context.

**Pipelines (Haystack 2.x):**
- **Indexing Pipeline:** DocumentCleaner → DocumentSplitter (200 words, 30 overlap) → OpenAIDocumentEmbedder → DocumentWriter
- **Query Pipeline:** OpenAITextEmbedder → QdrantEmbeddingRetriever → PromptBuilder → OpenAIGenerator

**Files:** `src/rag_pipeline.py`, `src/self_rag.py`

---

### 3.3 Style-Adaptive Responses (`/style-response`)

**What it does:** Generates casual chat replies that match the user's communication tone using few-shot learning from curated examples.

**How it works (step by step):**

1. **Tone Detection** (`src/self_rag.py:detect_tone`) — Classifies the incoming message as `formal`, `semi_formal`, or `casual`
2. **Few-Shot Example Retrieval** — Embeds the incoming message and searches Qdrant's `style_memory` collection filtered by the detected tone
   - If fewer than 2 tone-matched results, falls back to unfiltered search
3. **Conversation Memory** — Loads the sender's recent history for contextual replies
4. **Prompt Construction** — Builds a few-shot prompt with the retrieved style examples:
   ```
   Study these examples of how the user writes. Mirror their tone, vocabulary, and sentence length exactly.
   
   Received: "Hey, how's it going?"
   Replied: "All good! Just keeping busy. You?"
   
   Now write a reply to: [incoming message]
   ```
5. **Response Generation** — GPT-4o-mini generates a reply mimicking the style examples
6. **Multilingual** — Responds in the same language as the incoming message

**Fallback:** If no style examples exist in Qdrant, uses a generic friendly prompt.

**File:** `src/style_memory.py`

---

### 3.4 Meeting Scheduling (`/schedule`)

**What it does:** Parses natural language meeting requests, checks Google Calendar availability, and creates events.

**How it works (step by step):**

1. **Entity Extraction** — GPT-4o-mini extracts structured meeting details from natural language:
   ```json
   {
     "title": "Interview",
     "date": "2026-05-12",
     "time": "14:00",
     "duration": 60,
     "attendee": null,
     "language": "en"
   }
   ```
   Resolves relative dates ("tomorrow", "next Monday") using the current date.

2. **Validation** — If date or time couldn't be extracted, asks for clarification (in the user's language)

3. **Availability Check** — Queries Google Calendar's FreeBusy API to check if the time slot is free

4. **Event Creation** — If available, creates the event via Google Calendar API and returns an event link

5. **Conflict Handling** — If the slot is taken, suggests the next available slot (+1 hour)

6. **Proactive Follow-up** — Automatically schedules a follow-up message for 30 minutes after the meeting ends

7. **Multilingual** — Confirmation messages are generated in English, Arabic, or Turkish based on the detected language

**Authentication:** Uses OAuth 2.0 Desktop app flow with refresh token for persistent access.

**File:** `src/scheduler.py`

---

### 3.5 Conversation Memory

**What it does:** Tracks per-sender conversation history so the agent maintains context across multiple message turns.

**How it works:**
- Stores the last 5 message pairs (user + assistant) per sender
- Saved as JSON files at `data/conversation_logs/memory/<sender>.json`
- Injected into all response prompts as "Previous conversation" context
- Enables coherent multi-turn conversations (e.g., "Can you clarify that?" refers to the previous answer)

**File:** `src/conversation_memory.py`

---

### 3.6 Proactive Follow-ups (`/followups/due`)

**What it does:** Automatically sends follow-up messages after events like meetings.

**How it works:**
1. When a meeting is scheduled, a follow-up entry is registered in `data/followups.json`
2. The follow-up is scheduled for 30 minutes after the meeting end time
3. Every 10 minutes, the n8n follow-up workflow calls `/followups/due`
4. The backend checks for any follow-ups whose scheduled time has passed
5. For each due follow-up, the LLM generates a personalized message (e.g., "Hey! How did your Interview meeting go?")
6. n8n sends the message via WhatsApp

**Follow-up Types:**
- `post_meeting` — Sent after a scheduled meeting ends
- `check_in` — Daily check-in for active users (extensible)

**File:** `src/followups.py`

---

### 3.7 Daily Summary (`/summarize`)

**What it does:** Generates a structured daily summary of all conversations for the admin.

**How it works:**
1. All interactions are logged throughout the day in `data/conversation_logs/<date>.json`
2. At 8 PM (Riyadh time), the n8n daily summary workflow triggers
3. The backend loads today's log and computes statistics:
   - Total messages, unique senders
   - Intent distribution (casual/knowledge/scheduling)
   - Average response latency
   - Ungrounded response count
4. The full log is sent to GPT-4o-mini to generate a narrative summary
5. The summary + stats are formatted and sent to the admin's WhatsApp number

**File:** `src/summarizer.py`

---

### 3.8 Multilingual Support (English, Arabic, Turkish)

**What it does:** Detects the language of incoming messages and responds in the same language.

**How it works:**
- All response prompts include: "Always reply in the SAME LANGUAGE as the user's message"
- The intent classifier is instructed to handle messages regardless of language
- The scheduler extracts the message language and returns localized confirmations
- GPT-4o-mini natively supports Arabic, Turkish, and English
- No external translation API needed — the LLM handles language matching

**Supported across:** Casual chat, RAG queries, scheduling, follow-ups

---

### 3.9 Real-Time Monitoring Dashboard (`/dashboard`)

**What it does:** A web-based dashboard showing live system status, conversation logs, and pipeline metrics.

**Sections:**
| Section | Content |
|---------|---------|
| Services | Health status of backend, n8n, and Qdrant (green/red indicators) |
| Today's Stats | Message count, unique senders, avg latency, pending follow-ups, intent distribution |
| Recent Interactions | Every message with timestamp, sender, intent badge, incoming text, and bot reply |
| Conversation Memory | Per-sender chat history showing the last few turns |
| Backend Logs | Live Python logs color-coded by type (blue=intent, purple=style, green=RAG, red=errors) |

**Auto-refreshes** every 5 seconds. Accessible at `http://localhost:8000/dashboard` or via the tunnel URL.

**File:** `src/dashboard.py`

---

## 4. n8n Workflows (Step-by-Step)

### 4.1 Main Workflow — "WhatsApp Agent — Main Router"

This is the core workflow that processes every incoming WhatsApp message.

```
WhatsApp Incoming Message → Extract Message → Skip? → Classify Intent → Attach Intent
     ↓ (true=skip)                                         ↓
  Respond 200 OK                                    Route by Intent
                                                    ↓        ↓        ↓
                                                  RAG    Schedule   Style
                                                    ↓        ↓        ↓
                                                  Extract Reply Text
                                                         ↓
                                                  Send WhatsApp Reply
                                                         ↓
                                                  Log Interaction
                                                         ↓
                                                  Respond 200 to Meta
```

**Step-by-step:**

1. **WhatsApp Incoming Message** (Webhook node)
   - Receives POST requests from the backend (forwarded from Meta)
   - Trigger: any incoming WhatsApp message

2. **Extract Message** (Code node)
   - Parses the Meta webhook payload to extract: sender phone number, message text, message ID, timestamp
   - Filters out non-text messages (images, status updates, delivery receipts)
   - Sets `skip: true` if there's no text message to process

3. **Skip?** (If node)
   - True path → Respond 200 OK (acknowledges the webhook without processing)
   - False path → Continue to intent classification

4. **Classify Intent** (HTTP Request)
   - POST to `http://backend:8000/classify-intent`
   - Sends: `{ message, sender }`
   - Returns: `{ intent, confidence }`

5. **Attach Intent to Message** (Code node)
   - Merges the intent classification result with the original message data
   - Output: `{ sender, messageText, intent, confidence }`

6. **Route by Intent** (Switch node)
   - Routes to different AI endpoints based on the classified intent:
     - `knowledge_query` → RAG Query
     - `scheduling` → Schedule Meeting
     - `casual_chat` → Style-Adaptive Response
     - Fallback → Style-Adaptive Response

7. **RAG Query / Schedule Meeting / Style-Adaptive Response** (HTTP Request nodes)
   - Each calls the respective backend endpoint
   - RAG: POST `/rag-query` with `{ question, sender }`
   - Schedule: POST `/schedule` with `{ message, sender }`
   - Style: POST `/style-response` with `{ message, sender }`

8. **Extract Reply Text** (Code node)
   - Normalizes the response from whichever endpoint ran
   - RAG returns `answer`, Schedule returns `reply`, Style returns `reply`
   - Output: `{ replyText, grounded, sender, intent, incoming }`

9. **Send WhatsApp Reply** (HTTP Request)
   - POST to `https://graph.facebook.com/v19.0/{phone_number_id}/messages`
   - Sends the AI-generated reply back to the user's WhatsApp
   - Uses the WhatsApp Business API with Bearer token authentication

10. **Log Interaction** (HTTP Request)
    - POST to `http://backend:8000/log-interaction`
    - Persists: sender, incoming message, intent, reply, grounded status
    - Updates conversation memory for the sender

11. **Respond 200 to Meta** (Respond to Webhook node)
    - Returns `{"status": "ok"}` to Meta to acknowledge the webhook
    - Must respond within 20 seconds or Meta will retry

---

### 4.2 Daily Summary Workflow — "WhatsApp Agent — Daily Summary"

Generates and sends a daily conversation summary to the admin.

```
Daily at 8 PM → Generate Daily Summary → Format Summary Message → Send Summary to Admin
```

**Step-by-step:**

1. **Daily at 8 PM** (Schedule Trigger)
   - Cron expression: `0 20 * * *` (8:00 PM daily, Riyadh time)

2. **Generate Daily Summary** (HTTP Request)
   - POST to `http://backend:8000/summarize`
   - Returns: `{ summary: "...", stats: { total_messages, unique_senders, intent_counts, avg_latency_ms, ungrounded_count } }`

3. **Format Summary Message** (Code node)
   - Combines the narrative summary with system stats
   - Formats as a WhatsApp-friendly message with bold headers:
   ```
   *Daily Summary — 11/05/2026*
   
   [AI-generated narrative summary]
   
   *System Stats:*
   Total messages: 15
   Unique senders: 3
   Avg response time: 2450ms
   Ungrounded responses: 1
   ```

4. **Send Summary to Admin** (HTTP Request)
   - Sends the formatted summary to the admin's WhatsApp number (905316339030)
   - Uses the same WhatsApp Business API as the main workflow

---

### 4.3 Follow-up Workflow — "WhatsApp Agent — Proactive Follow-ups"

Checks for and sends due follow-up messages every 10 minutes.

```
Every 10 Minutes → Check Due Follow-ups → Any Due? → Split Follow-ups → Send Follow-up
```

**Step-by-step:**

1. **Every 10 Minutes** (Schedule Trigger)
   - Runs at a 10-minute interval

2. **Check Due Follow-ups** (HTTP Request)
   - GET `http://backend:8000/followups/due`
   - Returns: `{ followups: [...], count: N }`
   - Each follow-up: `{ sender, message, type, context }`

3. **Any Due?** (If node)
   - If count > 0 → proceed to send
   - If count = 0 → stop (nothing to do)

4. **Split Follow-ups** (Code node)
   - Converts the followups array into individual items so each gets its own WhatsApp message

5. **Send Follow-up** (HTTP Request)
   - Sends each follow-up message to the respective sender via WhatsApp API
   - The message was already generated by the backend's LLM (personalized and language-matched)

---

## 5. How the Project Runs End-to-End

### Startup Sequence

1. **Start Docker services:** `docker compose up -d`
   - Qdrant starts and loads vector collections from persistent storage
   - n8n starts, loads saved workflows, and activates webhook listeners
   - Backend starts with uvicorn, connects to Qdrant and loads configurations

2. **Start Cloudflare tunnel:** `cloudflared tunnel --url http://localhost:8000`
   - Creates a public HTTPS URL pointing to the local backend
   - This URL is registered as the webhook callback in Meta's WhatsApp settings

3. **Seed data (first time only):**
   - `python -m scripts.seed_knowledge_base` — indexes FAQ documents into Qdrant
   - `python -m scripts.seed_style_memory` — indexes style examples into Qdrant

### Message Flow (Complete Journey)

```
1. User sends "What are your working hours?" on WhatsApp
2. Meta receives the message and sends a webhook POST to the Cloudflare tunnel URL
3. Cloudflare forwards to localhost:8000/webhook/whatsapp
4. FastAPI backend receives it and forwards to n8n at n8n:5678/webhook/whatsapp
5. n8n "Extract Message" node parses: sender=905316339030, text="What are your working hours?"
6. n8n calls POST /classify-intent → returns {intent: "knowledge_query", confidence: 0.95}
7. n8n routes to the "RAG Query" branch
8. n8n calls POST /rag-query with {question: "What are your working hours?", sender: "905316339030"}
9. Backend loads conversation history for this sender
10. Self-RAG gate decides: YES, retrieval needed (this is a factual question)
11. Haystack pipeline embeds the question → retrieves top-5 documents from Qdrant
12. LLM generates answer from retrieved context: "Our team is available Monday through Friday, 9:00 AM to 6:00 PM (Riyadh time, GMT+3)."
13. Backend returns {answer: "...", grounded: true, sources: 1}
14. n8n "Extract Reply Text" normalizes the response
15. n8n "Send WhatsApp Reply" calls Meta's API to send the reply to the user
16. n8n "Log Interaction" calls POST /log-interaction to persist the exchange
17. Backend saves to daily log and updates conversation memory
18. n8n "Respond 200 to Meta" acknowledges the original webhook
19. User sees the reply on WhatsApp
```

### Key Configuration Files

| File | Purpose |
|------|---------|
| `.env` | All API keys and credentials (OpenAI, WhatsApp, Google Calendar) |
| `docker-compose.yml` | Service definitions for Qdrant, n8n, and the backend |
| `src/config.py` | Pydantic settings model that loads from .env |
| `data/knowledge_base/faq.txt` | Knowledge base documents for RAG |
| `data/style_examples/examples.json` | Curated style examples for few-shot learning |
| `data/google_credentials.json` | Google OAuth 2.0 client credentials |
| `data/google_token.json` | Google OAuth refresh token (auto-refreshes) |

---

## 6. API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/webhook/whatsapp` | Meta webhook verification (hub.challenge) |
| POST | `/webhook/whatsapp` | Receives and forwards WhatsApp webhooks to n8n |
| POST | `/classify-intent` | Classifies message intent (casual/knowledge/scheduling) |
| POST | `/rag-query` | Knowledge-grounded response with Self-RAG gate |
| POST | `/style-response` | Style-adaptive casual chat reply |
| POST | `/schedule` | Natural language meeting scheduling |
| POST | `/log-interaction` | Persists interaction log and updates memory |
| POST | `/summarize` | Generates daily conversation summary |
| GET | `/followups/due` | Returns follow-up messages ready to send |
| GET | `/dashboard` | Real-time monitoring dashboard (HTML) |
| GET | `/api/dashboard-data` | Dashboard data endpoint (JSON) |
| GET | `/health` | Health check |

---

## 7. File Structure

```
capstone_CL/
├── src/
│   ├── api.py                  # FastAPI app with all endpoints
│   ├── config.py               # Settings loaded from .env
│   ├── intent_classifier.py    # GPT-4o-mini intent classification
│   ├── rag_pipeline.py         # Haystack 2.x RAG with Self-RAG gate
│   ├── self_rag.py             # Retrieval gate + tone detection
│   ├── style_memory.py         # Tone-aware few-shot style responses
│   ├── scheduler.py            # Google Calendar meeting scheduling
│   ├── conversation_memory.py  # Per-sender conversation history
│   ├── summarizer.py           # Daily conversation summary generator
│   ├── followups.py            # Proactive follow-up scheduling
│   └── dashboard.py            # Real-time monitoring dashboard
├── scripts/
│   ├── seed_knowledge_base.py  # Index FAQ documents into Qdrant
│   ├── seed_style_memory.py    # Index style examples into Qdrant
│   └── import_whatsapp_chat.py # Import WhatsApp chat exports
├── evaluation/
│   └── evaluate.py             # RAGAS evaluation framework
├── n8n_workflows/
│   ├── main_workflow.json      # Main message processing workflow
│   ├── daily_summary_workflow.json  # Daily summary cron workflow
│   └── followup_workflow.json  # Proactive follow-up cron workflow
├── data/
│   ├── knowledge_base/faq.txt  # Knowledge base source documents
│   ├── style_examples/examples.json  # Style training examples
│   ├── google_credentials.json # Google OAuth credentials
│   ├── google_token.json       # Google OAuth token
│   ├── conversation_logs/      # Daily interaction logs
│   │   └── memory/             # Per-sender conversation memory
│   └── followups.json          # Pending follow-up entries
├── docker-compose.yml          # Docker service definitions
├── Dockerfile                  # Python backend container
├── requirements.txt            # Python dependencies
└── .env                        # Environment variables and secrets
```

---

## 8. Novel Contributions

1. **Self-RAG Gate** — Adaptive retrieval that skips unnecessary vector DB queries for casual messages, reducing latency and cost while maintaining accuracy for knowledge queries.

2. **Tone-Aware Style Memory** — Few-shot examples are filtered by detected tone (formal/semi-formal/casual) so the agent mirrors the user's communication register.

3. **Proactive Follow-ups** — The agent doesn't just respond reactively; it initiates contextual follow-up messages after meetings, demonstrating proactive conversational AI behavior.

4. **Multilingual Tone Adaptation** — Combines language detection with tone matching across English, Arabic, and Turkish without any external translation service.

5. **Unified Webhook Architecture** — A single endpoint handles both Meta's GET verification and POST message delivery, forwarding to n8n for orchestration while keeping AI inference in the Python backend.
