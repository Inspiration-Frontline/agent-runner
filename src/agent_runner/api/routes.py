"""HTTP routes for real-time Agent Conversation requests."""

import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_runner.application_services import ApplicationServices
from agent_runner.config import CancelConversationRequest, ConversationRequest
from agent_runner.conversation import ConversationBusyError
from agent_runner.observability.conversation_tracing import ConversationTracing
from agent_runner.runtime.orchestrator import RuntimeOrchestrator


class CancelConversationResponse(BaseModel):
    """Acknowledgement returned after requesting Conversation cancellation."""

    cancelled: bool


def get_trusted_user_id(x_user_id: str | None) -> int:
    """Parse the gateway-provided authenticated user header and reject spoofed values."""
    if x_user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is required.")
    try:
        user_id = int(x_user_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Trusted user identity is invalid.",
        ) from error
    if user_id <= 0:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is invalid.")
    return user_id


async def cancel_conversation(
    request: Request,
    cancel_request: CancelConversationRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> CancelConversationResponse:
    """Request cancellation of the active generation for one authenticated Conversation."""
    user_id = get_trusted_user_id(x_user_id)
    services: ApplicationServices = request.app.state.services
    cancelled = False
    for _ in range(10):
        cancelled = services.cancellations.cancel(user_id, cancel_request.conversation_id)
        if cancelled:
            break
        await asyncio.sleep(0.05)
    return CancelConversationResponse(cancelled=cancelled)


async def stream_conversation(
    request: Request,
    conversation_request: ConversationRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> StreamingResponse:
    """Stream one Agent Conversation through the established SSE HTTP contract."""
    user_id = get_trusted_user_id(x_user_id)
    services: ApplicationServices = request.app.state.services
    settings = services.get_settings()
    orchestrator = RuntimeOrchestrator(
        settings=settings,
        tracer=services.tracing.tracer,
        cancellation_registry=services.cancellations,
        mcp_connection_pool=getattr(services, "mcp_connection_pool", None),
        mcp_schema_cache=getattr(services, "mcp_schema_cache", None),
    )
    try:
        await orchestrator.acquire_conversation(conversation_request.conversation_id)
    except ConversationBusyError as error:
        await orchestrator.close()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONVERSATION_BUSY", "message": str(error)},
        ) from error

    async def generate_events() -> AsyncGenerator[str]:
        conversation_tracing = ConversationTracing(services.tracing.tracer, settings)
        with conversation_tracing.trace_request(dict(request.headers), conversation_request) as trace:
            try:
                async for event in orchestrator.run(conversation_request, user_id, request):
                    trace.record_event(event)
                    yield f"data: {event.model_dump_json()}\n\n"
            finally:
                await orchestrator.close()

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def create_agent_router() -> APIRouter:
    """Create the stateless HTTP routing table for one application instance."""
    router = APIRouter()
    router.add_api_route(
        "/chat/cancel",
        cancel_conversation,
        methods=["POST"],
        status_code=status.HTTP_202_ACCEPTED,
        response_model=CancelConversationResponse,
    )
    router.add_api_route("/chat/stream", stream_conversation, methods=["POST"])
    return router
