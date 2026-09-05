import logging
from collections.abc import Mapping
from typing import Any

import httpx

from agent_runner.config import Settings
from agent_runner.context.models import RagChunk

logger = logging.getLogger(__name__)


class RAGAdapter:
    """
    Adapter for RAG (Retrieval-Augmented Generation) service interactions.

    Provides methods to retrieve relevant knowledge chunks from the
    external knowledge service based on query context.

    Attributes:
        base_url: Base URL for the knowledge service.
        client: Async HTTP client for service communication.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the RAG adapter with service URL and HTTP client.

        Args:
            settings: Effective application settings for the operation.
        """
        current_settings: Settings = settings or Settings()
        self.base_url = current_settings.knowledge_service_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def retrieve(
        self,
        query: str,
        agent_id: int,
        user_id: int | None = None,
        top_k: int = 5,
    ) -> list[RagChunk]:
        """
        Retrieve relevant RAG chunks from the knowledge service.

        Args:
            query: The query text to find relevant chunks for.
            agent_id: The agent ID to scope the retrieval.
            user_id: Optional user ID for personalized retrieval.
            top_k: Number of top chunks to retrieve.

        Returns:
            list[RagChunk]: Normalized relevant chunks, or an empty list if retrieval fails.
        """

        try:
            response: httpx.Response = await self.client.post(
                f"{self.base_url}/api/v1/rag/retrieve",
                json={
                    "query": query,
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "top_k": top_k,
                },
            )

            if response.status_code == 200:
                data: Any = response.json()

                if not isinstance(data, Mapping):
                    return []
                chunks: Any = data.get("chunks", [])

                if not isinstance(chunks, list):
                    return []

                return [
                    RagChunk(
                        content=str(chunk.get("content", "")),
                        source=str(chunk.get("source", "Unknown")),
                    )
                    for chunk in chunks
                    if isinstance(chunk, Mapping) and chunk.get("content")
                ]
            logger.warning(f"Failed to retrieve RAG chunks: {response.status_code}")

            return []
        except Exception:
            logger.exception("Error retrieving RAG chunks")
            return []

    async def close(self) -> None:
        """
        Close the HTTP client connection.
        """
        await self.client.aclose()
