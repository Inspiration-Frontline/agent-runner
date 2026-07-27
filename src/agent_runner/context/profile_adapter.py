import logging
from collections.abc import Mapping

import httpx

from agent_runner.config import get_settings
from agent_runner.context.models import UserProfile

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

    def __init__(self) -> None:
        """
        Initialize the profile adapter with service URL and HTTP client.
        """
        current_settings = get_settings()
        self.base_url = current_settings.user_profiler_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def retrieve(self, user_id: int) -> UserProfile:
        """
        Retrieve user profile data from the service.

        Args:
            user_id: The unique identifier of the user.

        Returns:
            UserProfile: Normalized profile data, or an empty profile if retrieval fails.
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/v1/profile/{user_id}")
            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, Mapping):
                    return UserProfile.from_mapping(payload)
                logger.warning("Profile service returned a non-object payload for user %s", user_id)
                return UserProfile()
            logger.warning(f"Failed to retrieve profile for user {user_id}: {response.status_code}")
            return UserProfile()
        except Exception:
            logger.exception(f"Error retrieving profile for user {user_id}")
            return UserProfile()

    async def update(self, user_id: int, profile_data: UserProfile) -> bool:
        """
        Update user profile data in the service.

        Args:
            user_id: The unique identifier of the user.
            profile_data: Typed profile projection to update.

        Returns:
            bool: True if update succeeded, False otherwise.
        """
        try:
            response = await self.client.patch(
                f"{self.base_url}/api/v1/profile/{user_id}",
                json={attribute.name: attribute.value for attribute in profile_data.attributes},
            )
            return response.status_code == 200
        except Exception:
            logger.exception(f"Error updating profile for user {user_id}")
            return False

    async def close(self) -> None:
        """
        Close the HTTP client connection.
        """
        await self.client.aclose()
