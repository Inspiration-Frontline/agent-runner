import logging
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)


class ProfileAdapter:
    """
    Adapter for user profile service interactions.

    Provides methods to retrieve and update user profile data
    from the external user profiler service.

    Attributes:
        base_url: Base URL for the user profiler service.
        client: Async HTTP client for service communication.
    """

    def __init__(self):
        """
        Initialize the profile adapter with service URL and HTTP client.
        """
        self.base_url = settings.user_profiler_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def retrieve(self, user_id: str) -> dict[str, Any]:
        """
        Retrieve user profile data from the service.

        Args:
            user_id: The unique identifier of the user.

        Returns:
            dict[str, Any]: User profile data, or empty dict if retrieval fails.
        """
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
        """
        Update user profile data in the service.

        Args:
            user_id: The unique identifier of the user.
            profile_data: The profile data to update.

        Returns:
            bool: True if update succeeded, False otherwise.
        """
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
        """
        Close the HTTP client connection.
        """
        await self.client.aclose()
