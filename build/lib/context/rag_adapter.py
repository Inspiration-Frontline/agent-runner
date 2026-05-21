import logging
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)


class RAGAdapter:
    def __init__(self):
        self.base_url = settings.knowledge_service_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def retrieve(
        self,
        query: str,
        agent_id: str,
        user_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
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
        except Exception as e:
            logger.exception("Error retrieving RAG chunks")
            return []

    async def close(self):
        await self.client.aclose()
