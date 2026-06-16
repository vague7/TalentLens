"""ChromaDB vector store — collection management and retrieval.

Provides a wrapper around ChromaDB for storing resume chunk embeddings
and performing top-K semantic similarity search against job descriptions.

Supports both:
  - In-process (ephemeral/persistent) mode for testing and local dev
  - HTTP client mode for production (connects to docker-compose ChromaDB)

Usage:
    store = get_vector_store()
    store.add_chunks(chunks, embeddings)
    results = store.query(query_embedding, top_k=5)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import get_settings
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result from the vector store.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        text: The chunk text content.
        candidate_id: UUID of the candidate this chunk belongs to.
        section: Resume section this chunk came from.
        distance: Similarity distance (lower = more similar).
        metadata: Additional metadata from the chunk.
    """

    chunk_id: str
    text: str
    candidate_id: str
    section: str
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Aggregated result of a vector store query.

    Attributes:
        results: List of retrieval results, ordered by similarity.
        query_time_ms: Time taken for the query in milliseconds.
        total_results: Number of results returned.
    """

    results: list[RetrievalResult] = field(default_factory=list)
    query_time_ms: int = 0
    total_results: int = 0


class VectorStore:
    """ChromaDB-backed vector store for resume chunks.

    Manages a single ChromaDB collection for storing and retrieving
    resume chunk embeddings with metadata.

    Attributes:
        collection_name: Name of the ChromaDB collection.
    """

    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_name: Optional[str] = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            client: ChromaDB client instance.
            collection_name: Name of the collection. Defaults to config value.
        """
        settings = get_settings()
        self.collection_name = collection_name or settings.chroma_collection
        self._client = client
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(
            "vectorstore.init",
            collection=self.collection_name,
            count=self._collection.count(),
        )

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """Add resume chunks with their embeddings to the vector store.

        Args:
            chunk_ids: Unique IDs for each chunk.
            texts: Text content of each chunk.
            embeddings: Pre-computed embedding vectors.
            metadatas: Metadata dicts for each chunk (must include candidate_id, section).

        Returns:
            Number of chunks successfully added.

        Raises:
            ValueError: If input lists have mismatched lengths.
        """
        if not (len(chunk_ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError(
                f"Input lists must have equal length. Got: "
                f"ids={len(chunk_ids)}, texts={len(texts)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )

        if not chunk_ids:
            return 0

        start_time = time.perf_counter()

        # Ensure all metadata values are ChromaDB-compatible types
        clean_metadatas = []
        for meta in metadatas:
            clean = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_metadatas.append(clean)

        self._collection.upsert(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_metadatas,
        )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            "vectorstore.add_chunks",
            count=len(chunk_ids),
            collection=self.collection_name,
            response_time_ms=elapsed_ms,
        )

        return len(chunk_ids)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        candidate_filter: Optional[list[str]] = None,
    ) -> QueryResult:
        """Query the vector store for similar chunks.

        Args:
            query_embedding: The query embedding vector.
            top_k: Maximum number of results to return.
            candidate_filter: Optional list of candidate_ids to filter by.

        Returns:
            QueryResult with ranked retrieval results.
        """
        start_time = time.perf_counter()

        where_filter = None
        if candidate_filter:
            where_filter = {"candidate_id": {"$in": candidate_filter}}

        raw_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        results: list[RetrievalResult] = []
        if raw_results and raw_results["ids"] and raw_results["ids"][0]:
            ids = raw_results["ids"][0]
            docs = raw_results["documents"][0] if raw_results["documents"] else [""] * len(ids)
            metas = raw_results["metadatas"][0] if raw_results["metadatas"] else [{}] * len(ids)
            dists = raw_results["distances"][0] if raw_results["distances"] else [0.0] * len(ids)

            for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
                results.append(
                    RetrievalResult(
                        chunk_id=chunk_id,
                        text=doc or "",
                        candidate_id=meta.get("candidate_id", ""),
                        section=meta.get("section", ""),
                        distance=dist,
                        metadata=meta,
                    )
                )

        logger.info(
            "vectorstore.query",
            top_k=top_k,
            results_returned=len(results),
            collection=self.collection_name,
            response_time_ms=elapsed_ms,
        )

        return QueryResult(
            results=results,
            query_time_ms=elapsed_ms,
            total_results=len(results),
        )

    def get_candidate_ids(self) -> list[str]:
        """Get all unique candidate IDs stored in the collection.

        Returns:
            Sorted list of unique candidate IDs.
        """
        all_data = self._collection.get(include=["metadatas"])
        if not all_data or not all_data["metadatas"]:
            return []

        candidate_ids = set()
        for meta in all_data["metadatas"]:
            cid = meta.get("candidate_id", "")
            if cid:
                candidate_ids.add(cid)

        return sorted(candidate_ids)

    def get_chunks_by_candidate(self, candidate_id: str) -> list[dict[str, Any]]:
        """Retrieve all chunks for a specific candidate.

        Args:
            candidate_id: The candidate UUID to retrieve chunks for.

        Returns:
            List of dicts with id, text, and metadata for each chunk.
        """
        results = self._collection.get(
            where={"candidate_id": candidate_id},
            include=["documents", "metadatas"],
        )

        chunks = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                chunks.append({
                    "id": chunk_id,
                    "text": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })

        return chunks

    def delete_candidate(self, candidate_id: str) -> None:
        """Delete all chunks for a specific candidate.

        Args:
            candidate_id: The candidate UUID whose chunks should be deleted.
        """
        chunks = self.get_chunks_by_candidate(candidate_id)
        if chunks:
            ids_to_delete = [c["id"] for c in chunks]
            self._collection.delete(ids=ids_to_delete)
            logger.info(
                "vectorstore.delete_candidate",
                candidate_id=candidate_id,
                chunks_deleted=len(ids_to_delete),
            )

    def count(self) -> int:
        """Return the total number of chunks in the collection.

        Returns:
            Total chunk count.
        """
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection. Use with caution.

        Warning:
            This permanently deletes all data in the collection.
        """
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("vectorstore.reset", collection=self.collection_name)


def get_vector_store(
    mode: str = "http",
    collection_name: Optional[str] = None,
) -> VectorStore:
    """Factory function to create a VectorStore instance.

    Args:
        mode: Connection mode. Options:
            - "http": Connect to a remote ChromaDB server (production)
            - "ephemeral": In-memory store (testing)
            - "persistent": On-disk persistent store (local dev)
        collection_name: Optional collection name override.

    Returns:
        Configured VectorStore instance.

    Raises:
        ValueError: If mode is not recognized.
    """
    settings = get_settings()

    if mode == "http":
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    elif mode == "ephemeral":
        client = chromadb.EphemeralClient(
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    elif mode == "persistent":
        client = chromadb.PersistentClient(
            path="./chroma_data",
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    else:
        raise ValueError(f"Unknown vector store mode: '{mode}'. Use 'http', 'ephemeral', or 'persistent'.")

    logger.info("vectorstore.factory", mode=mode)
    return VectorStore(client=client, collection_name=collection_name)
