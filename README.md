# whatsapp-style-agent

A production-grade, multilingual WhatsApp conversational AI agent that adapts its tone and writing style to match each user. Built as a senior capstone project.

The agent handles three types of interactions through a single WhatsApp interface: casual conversation (style-mirrored), knowledge queries (RAG-grounded), and meeting scheduling (Google Calendar). All three run on a unified pipeline orchestrated by n8n, with a FastAPI backend handling all AI inference.

---

## Architecture

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

Three Docker services run the full stack: `Qdrant` (vector DB), `n8n` (workflow engine), and the `FastAPI` backend — all wired together via `docker-compose.yml`.

---

## Key Features

- **Style-Adaptive Responses** — detects each user's tone (formal / semi-formal / casual) and retrieves few-shot examples from Qdrant's style memory to mirror their exact vocabulary and sentence structure
- **Self-RAG Gate** — an LLM-based retrieval gate that decides whether a question actually needs vector search before hitting the database, reducing latency and cost on casual messages
- **RAG Knowledge Pipeline** — Haystack 2.x indexing and query pipeline with Qdrant embeddings (text-embedding-3-small, 1536 dimensions), grounding checks, and hallucination fallbacks
- **Natural Language Scheduling** — extracts meeting details from unstructured text, checks Google Calendar availability via FreeBusy API, creates events, and sends proactive follow-ups 30 minutes after meetings end
- **Conversation Memory** — per-sender history (last 5 turns) injected into every prompt for coherent multi-turn dialogue
- **Multilingual** — English, Arabic, and Turkish, with no external translation API
- **Proactive Follow-ups** — n8n cron workflow checks every 10 minutes for due follow-ups and sends personalized messages
- **Daily Admin Summary** — GPT-generated narrative + system stats (message count, intent distribution, avg latency, ungrounded responses) delivered to admin WhatsApp at 8 PM
- **Real-Time Dashboard** — live monitoring at `/dashboard`: service health, conversation logs, intent breakdown, response latencies

---

## Tech Stack

| Layer | Technology |
|---|---|
| Messaging | WhatsApp Business Cloud API (Meta) |
| Tunnel | Cloudflare Tunnel |
| AI Backend | FastAPI + Python |
| Workflow Engine | n8n (self-hosted) |
| Vector Database | Qdrant |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-small (1536d) |
| RAG Framework | Haystack 2.x |
| Calendar | Google Calendar API (OAuth 2.0) |
| Containerization | Docker Compose |
| Evaluation | RAGAS |

---

## Project Structure

```
whatsapp-style-agent/
├── src/
│   ├── api.py                    # FastAPI app — all endpoints
│   ├── intent_classifier.py      # GPT-4o-mini intent classification
│   ├── rag_pipeline.py           # Haystack 2.x RAG + Self-RAG gate
│   ├── self_rag.py               # Retrieval gate + tone detection
│   ├── style_memory.py           # Tone-aware few-shot style responses
│   ├── scheduler.py              # Google Calendar scheduling
│   ├── conversation_memory.py    # Per-sender conversation history
│   ├── summarizer.py             # Daily summary generator
│   ├── followups.py              # Proactive follow-up engine
│   └── dashboard.py              # Real-time monitoring dashboard
├── scripts/
│   ├── seed_knowledge_base.py    # Index FAQ documents into Qdrant
│   ├── seed_style_memory.py      # Index style examples into Qdrant
│   └── import_whatsapp_chat.py   # Import WhatsApp chat exports
├── n8n_workflows/
│   ├── main_workflow.json        # Main message routing workflow
│   ├── daily_summary_workflow.json
│   └── followup_workflow.json
├── evaluation/
│   └── evaluate.py               # RAGAS evaluation
├── data/
│   ├── knowledge_base/faq.txt    # RAG source documents
│   └── style_examples/examples.json
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key
- WhatsApp Business API credentials (Meta Developer account)
- Google Calendar API credentials (OAuth 2.0)
- Cloudflare account (for tunnel)

### 1. Clone and configure

```bash
git clone https://github.com/abdelqasim/whatsapp-style-agent.git
cd whatsapp-style-agent
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Start services

```bash
docker compose up -d
```

This starts three containers: Qdrant on port 6333, n8n on port 5678, and the FastAPI backend on port 8000.

### 3. Seed the knowledge base (first time only)

```bash
python -m scripts.seed_knowledge_base
python -m scripts.seed_style_memory
```

### 4. Start the Cloudflare tunnel

```bash
cloudflared tunnel --url http://localhost:8000
```

Copy the generated HTTPS URL and set it as your webhook callback in Meta's WhatsApp settings.

### 5. Import n8n workflows

Open n8n at `http://localhost:5678`, go to **Settings → Import**, and import all three JSON files from `n8n_workflows/`.

The agent is now live. Messages sent to your WhatsApp number will be processed in real time.

---

## How a Message Gets Processed

```
1. User sends "What are your working hours?" on WhatsApp
2. Meta sends a webhook POST to the Cloudflare tunnel
3. FastAPI receives and forwards to n8n
4. n8n classifies intent → "knowledge_query" (confidence: 0.95)
5. n8n routes to the RAG endpoint
6. Self-RAG gate: YES, retrieval needed
7. Haystack embeds the query → retrieves top-5 docs from Qdrant
8. GPT-4o-mini generates a grounded answer
9. n8n sends the reply via WhatsApp API
10. Interaction is logged, conversation memory updated
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/webhook/whatsapp` | Receives WhatsApp webhooks from Meta |
| POST | `/classify-intent` | Intent classification (casual / knowledge / scheduling) |
| POST | `/rag-query` | RAG-grounded knowledge response |
| POST | `/style-response` | Style-adaptive casual chat reply |
| POST | `/schedule` | Natural language meeting scheduling |
| GET | `/followups/due` | Returns follow-ups ready to send |
| POST | `/summarize` | Generates daily conversation summary |
| GET | `/dashboard` | Real-time monitoring dashboard |
| GET | `/health` | Health check |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
WHATSAPP_TOKEN=
WHATSAPP_PHONE_NUMBER_ID=
WHATSAPP_VERIFY_TOKEN=
ADMIN_PHONE_NUMBER=
QDRANT_HOST=localhost
QDRANT_PORT=6333
N8N_HOST=localhost
GOOGLE_CALENDAR_CREDENTIALS=./data/google_credentials.json
```

---

## Novel Design Decisions

**Self-RAG Gate** — Standard RAG always retrieves regardless of whether the question needs it. This project implements a lightweight LLM decision function (inspired by Asai et al., 2023) that skips vector search for casual messages, cutting unnecessary latency and database load.

**Tone-Filtered Few-Shot Retrieval** — Style examples in Qdrant are tagged by tone. When generating a reply, the system first detects the user's tone, then filters retrieved examples to only tone-matched ones before building the prompt. This produces significantly more natural style mirroring than unfiltered retrieval.

**Proactive Architecture** — Most conversational agents are purely reactive. This agent registers follow-up tasks at scheduling time and delivers them proactively via a separate n8n cron loop, making the interaction feel more like a real assistant.

---

## Full Documentation

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for a complete breakdown of every component, n8n workflow step-by-step, the full message flow, and design rationale.
