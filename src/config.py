from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")

    # ── WhatsApp Business API ──────────────────────────────────────────────────
    whatsapp_token: str = Field(..., env="WHATSAPP_TOKEN")
    whatsapp_phone_number_id: str = Field(..., env="WHATSAPP_PHONE_NUMBER_ID")
    whatsapp_verify_token: str = Field("my_random_secret", env="WHATSAPP_VERIFY_TOKEN")
    whatsapp_api_url: str = "https://graph.facebook.com/v19.0"
    # Admin phone number that receives daily summaries (international format, no +)
    admin_phone_number: str = Field(..., env="ADMIN_PHONE_NUMBER")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_kb_collection: str = "knowledge_base"
    qdrant_style_collection: str = "style_memory"

    # ── Google Calendar ───────────────────────────────────────────────────────
    google_calendar_credentials: str = Field(
        "./data/google_credentials.json", env="GOOGLE_CALENDAR_CREDENTIALS"
    )
    google_calendar_token: str = "./data/google_token.json"
    google_calendar_id: str = Field("primary", env="GOOGLE_CALENDAR_ID")

    # ── RAG settings ──────────────────────────────────────────────────────────
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.3    # below this score → no retrieval

    # ── Style memory settings ─────────────────────────────────────────────────
    style_top_k: int = 3                     # few-shot examples to retrieve

    # ── Conversation memory settings ─────────────────────────────────────────
    memory_max_turns: int = 5                # last N message pairs to remember per sender

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
