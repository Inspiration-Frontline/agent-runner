"""
API routes module for agent runner service.

This module defines the FastAPI routes for agent chat interactions,
providing streaming response endpoints for real-time agent communication.
"""

import asyncio

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from agent_runner.config import CancelChatRequest, ChatRequest
from agent_runner.conversation import ConversationBusyError
from agent_runner.runtime.cancellation import conversation_cancellation_registry
from agent_runner.runtime.orchestrator import RuntimeOrchestrator

router = APIRouter()


def trusted_user_id(x_user_id: str | None) -> int:
    """Parse the gateway-provided authenticated user header and reject spoofed values."""
    if x_user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is required.")
    try:
        user_id = int(x_user_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is invalid.") from error
    if user_id <= 0:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is invalid.")
    return user_id


@router.post("/chat/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_chat(
    cancel_request: CancelChatRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
):
    """Request cancellation of the active generation for one authenticated Conversation."""
    user_id = trusted_user_id(x_user_id)
    cancelled = False
    for _ in range(10):
        cancelled = conversation_cancellation_registry.cancel(user_id, cancel_request.conversation_id)
        if cancelled:
            break
        await asyncio.sleep(0.05)
    return {"cancelled": cancelled}


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
):
    """
    Stream agent chat responses through SSE (Server-Sent Events).

    This endpoint accepts a chat request and returns a streaming response
    containing real-time agent output, including text tokens, tool calls,
    and completion markers.

    Args:
        request: The FastAPI request object for connection management.
        chat_request: The chat request containing:
            - agent_id: Agent identifier to invoke
            - version: Optional agent version
            - conversation_id: Optional conversation ID
            - user_id: User identifier
            - message: User's message content

    Returns:
        StreamingResponse: SSE stream with events:
            - Token delta events (text content)
            - Tool start/result events
            - Error events
            - Done event (completion marker)

    Example:
        POST /v1/agent/chat/stream
        {
            "agent_id": "assistant-v1",
            "user_id": "user123",
            "message": "Hello, how can I help?"
        }
    """
    user_id = trusted_user_id(x_user_id)

    orchestrator = RuntimeOrchestrator()
    try:
        await orchestrator.acquire_conversation(chat_request.conversation_id)
    except ConversationBusyError as error:
        await orchestrator.close()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={
            "code": "CONVERSATION_BUSY",
            "message": str(error),
        }) from error

    async def event_generator():
        """Yield orchestrator events as SSE frames while preserving disconnect cleanup."""
        try:
            async for event in orchestrator.run(chat_request, user_id, request):
                yield f"data: {event.model_dump_json()}\n\n"
        finally:
            await orchestrator.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
