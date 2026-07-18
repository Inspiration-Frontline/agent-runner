from typing import Any

from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc import (
    GetConversationReplayRequest,
    GetConversationReplayResponse,
    GetConversationRoundHistoryRequest,
    GetConversationRoundHistoryResponse,
    PrepareConversationFilesRequest,
    PrepareConversationFilesResponse,
    ReplayDetailLevel,
    SaveConversationRoundRequest,
    SaveConversationRoundResponse,
)
from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc.grpc import (
    ConversationRpcServiceClient,
)
from fancy_grpc import GrpcRuntimeConfig, start_client

from agent_runner.config import get_settings


class ConversationManagerClient:
    def __init__(self) -> None:
        settings = get_settings()
        config = GrpcRuntimeConfig(
            urls={"conversation-manager": [settings.conversation_rpc_url]},
            timeout=5.0,
            retries=1,
            retry_backoff=0.15,
        )
        self._runtime = start_client(config=config)
        self._service: Any = self._runtime.service(ConversationRpcServiceClient)

    async def get_round_history(
        self, user_id: int, conversation_id: str
    ) -> GetConversationRoundHistoryResponse:
        return await self._service.get_conversation_round_history(
            GetConversationRoundHistoryRequest(user_id=user_id, conversation_id=conversation_id)
        )

    async def save_round(self, request: SaveConversationRoundRequest) -> SaveConversationRoundResponse:
        return await self._service.save_conversation_round(request, timeout=10.0)

    async def prepare_files(
        self,
        user_id: int,
        conversation_id: str,
        request_id: str,
        file_ids: list[str],
    ) -> PrepareConversationFilesResponse:
        return await self._service.prepare_conversation_files(
            PrepareConversationFilesRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                request_id=request_id,
                file_ids=file_ids,
            ),
            timeout=10.0,
        )

    async def get_model_context(
        self, user_id: int, conversation_id: str, end_round_number: int
    ) -> GetConversationReplayResponse:
        return await self._service.get_conversation_replay(
            GetConversationReplayRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                end_round_number=end_round_number,
                detail_level=ReplayDetailLevel.MODEL_CONTEXT,
            )
        )

    async def close(self) -> None:
        await self._runtime.close()
