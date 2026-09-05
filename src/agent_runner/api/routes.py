"""HTTP routes for real-time Agent Conversation requests."""

import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent_runner.application_services import ApplicationServices
from agent_runner.config import CancelConversationRequest, ConversationRequest, Settings
from agent_runner.conversation import ConversationBusyError
from agent_runner.observability.conversation_tracing import ConversationTrace, ConversationTracing
from agent_runner.runtime.orchestrator import RuntimeOrchestrator


class CancelConversationResponse(BaseModel):
    """Acknowledgement returned after requesting Conversation cancellation."""

    cancelled: bool
    """Whether an active request accepted the cancellation signal."""


def get_trusted_user_id(x_user_id: str | None) -> int:
    """Parse the gateway-provided authenticated user header and reject spoofed values.

    Args:
        x_user_id: Trusted numeric user identifier forwarded by the Gateway.

    Returns:
        Authenticated numeric user identifier.
    """

    if x_user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is required.")

    try:
        user_id: int = int(x_user_id)
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
    """Request cancellation of the active generation for one authenticated Conversation.

    Args:
        request: FastAPI request carrying application services.
        cancel_request: Cancellation payload identifying the active Conversation.
        x_user_id: Trusted numeric user identifier forwarded by the Gateway.

    Returns:
        Cancellation acknowledgement for the authenticated Conversation.
    """
    user_id: int = get_trusted_user_id(x_user_id)
    services: ApplicationServices = request.app.state.services
    cancelled: bool = False

    for _ in range(10):
        cancelled = services.cancellations.cancel(user_id, cancel_request.conversation_id)

        if cancelled:
            break

        await asyncio.sleep(0.05)

    return CancelConversationResponse(cancelled=cancelled)


async def generate_conversation_events(
    trace: ConversationTrace,
    orchestrator: RuntimeOrchestrator,
    conversation_request: ConversationRequest,
    user_id: int,
    http_request: Request,
) -> AsyncGenerator[str]:
    """Yield serialized SSE events for an already acquired Conversation request.

    Args:
        trace: Request-scoped trace recorder that owns aggregate stream evidence.
        orchestrator: Runtime coordinator whose resources must be closed after streaming.
        conversation_request: Validated public Conversation request.
        user_id: Authenticated user ID resolved by the route boundary.
        http_request: FastAPI request used to detect client disconnects in the orchestrator.

    Yields:
        Server-Sent Events containing one typed runtime event per message.

    Side Effects:
        Records every emitted event, finalizes the request span, and closes the orchestrator.
    """

    try:
        with trace.activate():
            async for event in orchestrator.run(conversation_request, user_id, http_request):
                trace.record_event(event)
                yield f"data: {event.model_dump_json()}\n\n"
    finally:
        trace.finish()
        await orchestrator.close()


async def stream_conversation_events(
    request: Request,
    conversation_request: ConversationRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> StreamingResponse:
    """Stream one Agent Conversation through the established SSE HTTP contract.

    Args:
        request: FastAPI request carrying application services and disconnect state.
        conversation_request: Validated text, attachment, and reference input.
        x_user_id: Trusted user identity forwarded by Gateway.

    Returns:
        A streaming response whose body is the durable Round event sequence.

    Raises:
        HTTPException: When the authenticated identity is missing or the Conversation is busy.
    """
    user_id: int = get_trusted_user_id(x_user_id)
    services: ApplicationServices = request.app.state.services
    settings: Settings = services.get_settings()
    orchestrator: RuntimeOrchestrator = RuntimeOrchestrator(
        settings=settings,
        tracer=services.tracing.tracer,
        cancellation_registry=services.cancellations,
        mcp_connection_pool=getattr(services, "mcp_connection_pool", None),
        mcp_schema_cache=getattr(services, "mcp_schema_cache", None),
        mcp_secret_provider=getattr(services, "mcp_secret_provider", None),
    )
    conversation_tracing: ConversationTracing = ConversationTracing(services.tracing.tracer, settings)
    trace: ConversationTrace = conversation_tracing.start_request(dict(request.headers), conversation_request)

    try:
        with trace.activate():
            await orchestrator.acquire_conversation(conversation_request.conversation_id)
    except ConversationBusyError as error:
        trace.finish()
        await orchestrator.close()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONVERSATION_BUSY", "message": str(error)},
            headers={"X-Round-Trace-Id": trace.trace_id},
        ) from error
    except BaseException:
        trace.finish()
        await orchestrator.close()

        raise

    return StreamingResponse(
        generate_conversation_events(trace, orchestrator, conversation_request, user_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Round-Trace-Id": trace.trace_id,
        },
    )


def create_agent_router() -> APIRouter:
    """Create the stateless HTTP routing table for one application instance.

    Returns:
        Stateless HTTP routing table for one application instance.
    """
    router: APIRouter = APIRouter()
    router.add_api_route(
        "/chat/cancel",
        cancel_conversation,
        methods=["POST"],
        status_code=status.HTTP_202_ACCEPTED,
        response_model=CancelConversationResponse,
    )
    router.add_api_route("/chat/stream", stream_conversation_events, methods=["POST"])

    return router
