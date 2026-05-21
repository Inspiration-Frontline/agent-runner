import logging
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)


class ProfileAdapter:
    def __init__(self):
        self.base_url = settings.user_profiler_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def retrieve(self, user_id: str) -> dict[str, Any]:
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/profile/{user_id}")
            if response.status_code == 200:
                return response.json()
            logger.warning(f"Failed to retrieve profile for user {user_id}: {response.status_code}")
            return {}
        except Exception as e:
            logger.exception(f"Error retrieving profile for user {user_id}")
            return {}

    async def update(self, user_id: str, profile_data: dict[str, Any]) -> bool:
        try:
            response = await self.client.patch(
                f"{self.base_url}/api/v1/profile/{user_id}",
                json=profile_data,
            )
            return response.status_code == 200
        except Exception as e:
            logger.exception(f"Error updating profile for user {user_id}")
            return False

    async def close(self):
        await self.client.aclose()
