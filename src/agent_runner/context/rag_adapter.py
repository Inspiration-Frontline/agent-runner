import logging
from typing import Any

import httpx

from agent_runner.config import get_settings

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

    def __init__(self):
        """
        Initialize the RAG adapter with service URL and HTTP client.
        """
        current_settings = get_settings()
        self.base_url = current_settings.knowledge_service_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def retrieve(
        self,
        query: str,
        agent_id: str,
        user_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant RAG chunks from the knowledge service.

        Args:
            query: The query text to find relevant chunks for.
            agent_id: The agent ID to scope the retrieval.
            user_id: Optional user ID for personalized retrieval.
            top_k: Number of top chunks to retrieve.

        Returns:
            list[dict[str, Any]]: List of relevant RAG chunks, or empty list if retrieval fails.
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/rag/retrieve",
                json={
                    "query": query,
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "top_k": top_k,
                },
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("chunks", [])
            logger.warning(f"Failed to retrieve RAG chunks: {response.status_code}")
            return []
        except Exception:
            logger.exception("Error retrieving RAG chunks")
            return []

    async def close(self):
        """
        Close the HTTP client connection.
        """
        await self.client.aclose()
