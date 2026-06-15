"""OpenAI embeddings wrapper with retry and logging.

Wraps the OpenAI text-embedding-3-small model with tenacity retry
(max 3 attempts, exponential backoff) and structlog response time logging.

Usage:
    embedder = get_embedder()
    vectors = embedder.embed_texts(["chunk 1", "chunk 2"])
"""

from __future__ import annotations

import os
import time
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from logging_config import get_logger

logger = get_logger(__name__)


class ResumeEmbedder:
    """Embeds text chunks using OpenAI text-embedding-3-small.

    Attributes:
        model: The OpenAI embedding model name.
        client: OpenAI API client instance.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the embedder.

        Args:
            api_key: OpenAI API key. Falls back to env var if not provided.
            model: Embedding model name. Defaults to text-embedding-3-small.
        """
        settings = get_settings()
        self.model = model or settings.openai_embedding_model
        resolved_key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")

        self.client = OpenAI(api_key=resolved_key)

        logger.info("embedder.init", model=self.model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            openai.APIError: On API failure after 3 retries.
        """
        if not texts:
            return []

        start_time = time.perf_counter()

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        embeddings = [item.embedding for item in response.data]

        logger.info(
            "embedder.embed_texts",
            count=len(texts),
            model=self.model,
            dimensions=len(embeddings[0]) if embeddings else 0,
            response_time_ms=elapsed_ms,
        )

        return embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text into a vector.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            openai.APIError: On API failure after 3 retries.
        """
        start_time = time.perf_counter()

        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        embedding = response.data[0].embedding

        logger.info(
            "embedder.embed_query",
            model=self.model,
            dimensions=len(embedding),
            response_time_ms=elapsed_ms,
        )

        return embedding


def get_embedder(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> ResumeEmbedder:
    """Factory function to create a ResumeEmbedder instance.

    Args:
        api_key: Optional OpenAI API key override.
        model: Optional embedding model name override.

    Returns:
        Configured ResumeEmbedder instance.
    """
    return ResumeEmbedder(api_key=api_key, model=model)
