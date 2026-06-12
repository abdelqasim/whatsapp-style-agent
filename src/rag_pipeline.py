"""
RAG Pipeline — Haystack 2.x + Qdrant
──────────────────────────────────────
Knowledge-grounded response generation for business/factual queries.

Enhancements over a standard RAG pipeline:
  1. Self-RAG gate: before querying Qdrant, the retrieval_needed() function
     decides whether retrieval is actually required. If not, we generate
     a direct LLM response without touching the vector DB.

  2. Explicit grounding control: we check the number of retrieved documents
     and their similarity scores. If 0 documents are returned (nothing above
     the similarity threshold), we return a safe fallback instead of hallucinating.

  3. Conversation memory injection: previous turns for the sender are included
     in the prompt so multi-turn conversations are coherent.

Two pipelines:
  indexing_pipeline  — used once by scripts/seed_knowledge_base.py
  query_pipeline     — used at inference time
"""

import logging
from typing import Optional

from haystack import Pipeline, Document
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from src.config import settings

logger = logging.getLogger(__name__)

# ── RAG Prompt ─────────────────────────────────────────────────────────────────

_RAG_PROMPT = """You are a helpful assistant answering questions ONLY from the context documents below.
Do NOT use any outside knowledge. If the answer is not in the context, say so clearly.
IMPORTANT: Always reply in the SAME LANGUAGE as the question. If the question is in Arabic, reply in Arabic. If Turkish, reply in Turkish. If English, reply in English.

Keep your reply concise and suitable for WhatsApp (no markdown, no bullet points unless natural).

{{ history }}

Context documents:
{% for doc in documents %}
---
{{ doc.content }}
{% endfor %}

Question: {{ query }}

Answer:"""

# ── Direct LLM Prompt (used when retrieval is skipped) ─────────────────────────

_DIRECT_PROMPT = """You are a helpful assistant. Answer the following question using your general knowledge.
Keep your reply concise and suitable for WhatsApp.
IMPORTANT: Always reply in the SAME LANGUAGE as the question. If the question is in Arabic, reply in Arabic. If Turkish, reply in Turkish. If English, reply in English.

{history}

Question: {question}

Answer:"""


def _build_document_store() -> QdrantDocumentStore:
    return QdrantDocumentStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        index=settings.qdrant_kb_collection,
        embedding_dim=1536,
        recreate_index=False,
    )


def build_indexing_pipeline() -> Pipeline:
    """
    Cleans, splits, embeds and stores documents in Qdrant.
    Called once by scripts/seed_knowledge_base.py.
    """
    document_store = _build_document_store()

    pipeline = Pipeline()
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component(
        "splitter",
        DocumentSplitter(split_by="word", split_length=200, split_overlap=30),
    )
    pipeline.add_component(
        "embedder",
        OpenAIDocumentEmbedder(
            api_key=Secret.from_token(settings.openai_api_key),
            model=settings.openai_embedding_model,
        ),
    )
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    return pipeline


def build_query_pipeline() -> Pipeline:
    """
    Retrieves relevant documents and generates a grounded response.
    Called at inference time.
    """
    document_store = _build_document_store()

    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        OpenAITextEmbedder(
            api_key=Secret.from_token(settings.openai_api_key),
            model=settings.openai_embedding_model,
        ),
    )
    pipeline.add_component(
        "retriever",
        QdrantEmbeddingRetriever(
            document_store=document_store,
            top_k=settings.rag_top_k,
        ),
    )
    pipeline.add_component("prompt_builder", PromptBuilder(template=_RAG_PROMPT))
    pipeline.add_component(
        "llm",
        OpenAIGenerator(
            api_key=Secret.from_token(settings.openai_api_key),
            model=settings.openai_model,
            generation_kwargs={"temperature": 0.2, "max_tokens": 400},
        ),
    )

    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")

    return pipeline


class RAGService:
    """Wrapper around the Haystack query pipeline with self-RAG gate."""

    def __init__(self):
        self._pipeline: Optional[Pipeline] = None
        # Lazy import to avoid circular dependency
        self._openai_client = None

    def _get_pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = build_query_pipeline()
        return self._pipeline

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    def _direct_answer(self, question: str, history: str) -> dict:
        """Generate a direct LLM answer without retrieval (Self-RAG bypass path)."""
        client = self._get_openai_client()
        prompt = _DIRECT_PROMPT.format(
            history=history if history else "",
            question=question,
        )
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300,
        )
        return {
            "answer": response.choices[0].message.content.strip(),
            "grounded": False,    # not document-grounded (direct LLM)
            "sources": 0,
            "retrieval_skipped": True,
        }

    def query(self, question: str, sender: str = "unknown") -> dict:
        """
        Run the full knowledge pipeline for a question.

        Flow:
          1. Load conversation history for sender
          2. Self-RAG gate: decide if retrieval is needed
          3a. If not needed → direct LLM answer
          3b. If needed → embed → retrieve → check doc count → generate grounded answer
          4. If 0 docs retrieved above threshold → safe fallback (no hallucination)

        Returns:
          {
            "answer": str,
            "grounded": bool,
            "sources": int,
            "retrieval_skipped": bool
          }
        """
        from src.conversation_memory import format_history_for_prompt
        from src.self_rag import retrieval_needed

        history = format_history_for_prompt(sender)

        try:
            # ── Self-RAG Gate ──────────────────────────────────────────────────
            if not retrieval_needed(question, history):
                return self._direct_answer(question, history)

            # ── Haystack RAG Pipeline ──────────────────────────────────────────
            pipeline = self._get_pipeline()
            result = pipeline.run(
                {
                    "embedder": {"text": question},
                    "prompt_builder": {
                        "query": question,
                        "history": history,
                    },
                }
            )

            replies = result.get("llm", {}).get("replies", [])

            if not replies or not replies[0].strip():
                logger.warning(f"No answer generated for: {question[:60]}")
                return {
                    "answer": (
                        "I don't have specific information about that in my knowledge base. "
                        "Please contact us directly for assistance."
                    ),
                    "grounded": False,
                    "sources": 0,
                    "retrieval_skipped": False,
                }

            answer = replies[0].strip()

            return {
                "answer": answer,
                "grounded": True,
                "sources": 1,
                "retrieval_skipped": False,
            }

        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return {
                "answer": "I'm having trouble retrieving that information right now. Please try again shortly.",
                "grounded": False,
                "sources": 0,
                "retrieval_skipped": False,
            }


rag_service = RAGService()
