from typing import Any, cast

from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc import (
    ConversationReference,
    GetConversationReplayRequest,
    GetConversationReplayResponse,
    GetConversationRoundHistoryRequest,
    GetConversationRoundHistoryResponse,
    PrepareConversationFilesRequest,
    PrepareConversationFilesResponse,
    PrepareConversationReferencesRequest,
    PrepareConversationReferencesResponse,
    ReplayDetailLevel,
    SaveConversationRoundRequest,
    SaveConversationRoundResponse,
)
from agent_breaker_conversation_manager_protos.ifl.agentbreaker.conversationmanager.rpc.grpc import (
    ConversationRpcServiceClient,
)
from fancy_grpc import GrpcRuntimeConfig, start_client

from agent_runner.config import get_settings
from agent_runner.observability.tracing import inject_trace_context


class ConversationManagerClient:
    def __init__(self) -> None:
        """Create the request-scoped RPC client used for durable conversation state.

        Runner keeps this client thin because Conversation Manager owns authorization, Round
        numbering, file reservation, and PostgreSQL persistence; the client only translates the
        request-scoped calls and applies bounded retry/timeout settings.
        """
        settings = get_settings()
        config = GrpcRuntimeConfig(
            urls={"conversation-manager": [settings.conversation_rpc_url]},
            timeout=5.0,
            retries=1,
            retry_backoff=0.15,
        )
        self._runtime = start_client(config=config)
        self._service: Any = self._runtime.service(ConversationRpcServiceClient)

    @staticmethod
    def _trace_metadata() -> list[tuple[str, str]]:
        headers: dict[str, str] = {}
        inject_trace_context(headers)
        return list(headers.items())

    async def get_round_history(self, user_id: int, conversation_id: str) -> GetConversationRoundHistoryResponse:
        """Fetch caller-owned compact history before allocating the next Round number.

        Args:
            user_id: Trusted authenticated owner.
            conversation_id: Conversation whose high-water mark and summaries are needed.

        Returns:
            Typed history response containing the latest persisted Round number.
        """
        response = await self._service.get_conversation_round_history(
            GetConversationRoundHistoryRequest(user_id=user_id, conversation_id=conversation_id),
            metadata=self._trace_metadata(),
        )
        return cast(GetConversationRoundHistoryResponse, response)

    async def save_round(self, request: SaveConversationRoundRequest) -> SaveConversationRoundResponse:
        """Persist one terminal Round through the Manager RPC.

        Args:
            request: Complete Runner capture, including stable user content and nested Turn data.

        Returns:
            Domain response envelope; validation conflicts are data, while transport failures raise.
        """
        response = await self._service.save_conversation_round(
            request, timeout=10.0, metadata=self._trace_metadata()
        )
        return cast(SaveConversationRoundResponse, response)

    async def prepare_files(
        self,
        user_id: int,
        conversation_id: str,
        request_id: str,
        file_ids: list[str],
    ) -> PrepareConversationFilesResponse:
        """Authorize and reserve one frozen attachment selection in Conversation Manager.

        The request carries IDs because the manager owns file permissions, processing state, and
        signed image URLs. Runner polls with the same request ID until every resource is READY.

        Args:
            user_id: Trusted authenticated owner.
            conversation_id: Conversation that will own the eventual Round references.
            request_id: Stable poll/reservation correlation ID.
            file_ids: Stable file references selected by the browser; never file bytes.

        Returns:
            Per-file state and aggregate readiness/failure flags.
        """
        response = await self._service.prepare_conversation_files(
            PrepareConversationFilesRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                request_id=request_id,
                file_ids=file_ids,
            ),
            timeout=10.0,
            metadata=self._trace_metadata(),
        )
        return cast(PrepareConversationFilesResponse, response)

    async def get_model_context(
        self, user_id: int, conversation_id: str, end_round_number: int
    ) -> GetConversationReplayResponse:
        """Fetch normalized replay context for a completed Round boundary.

        Args:
            user_id: Trusted authenticated owner.
            conversation_id: Conversation to replay.
            end_round_number: Inclusive completed Round boundary.

        Returns:
            Typed model-context response used to reconstruct provider-neutral history.
        """
        response = await self._service.get_conversation_replay(
            GetConversationReplayRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                end_round_number=end_round_number,
                detail_level=ReplayDetailLevel.MODEL_CONTEXT,
            ),
            metadata=self._trace_metadata(),
        )
        return cast(GetConversationReplayResponse, response)

    async def prepare_references(
        self,
        user_id: int,
        destination_conversation_id: str,
        references: list[ConversationReference],
    ) -> PrepareConversationReferencesResponse:
        """Authorize and resolve frozen same-Group Conversation evidence in one RPC."""
        response = await self._service.prepare_conversation_references(
            PrepareConversationReferencesRequest(
                user_id=user_id,
                destination_conversation_id=destination_conversation_id,
                references=references,
            ),
            timeout=10.0,
            metadata=self._trace_metadata(),
        )
        return cast(PrepareConversationReferencesResponse, response)

    async def close(self) -> None:
        """Close the underlying RPC channel so request cleanup does not leak sockets/tasks."""
        await self._runtime.close()
