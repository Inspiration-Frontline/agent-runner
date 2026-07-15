"""
API routes module for agent runner service.

This module defines the FastAPI routes for agent chat interactions,
providing streaming response endpoints for real-time agent communication.
"""

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from agent_runner.config import ChatRequest
from agent_runner.runtime.orchestrator import RuntimeOrchestrator

router = APIRouter()


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
    if x_user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is required.")
    try:
        user_id = int(x_user_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is invalid.") from error
    if user_id <= 0:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is invalid.")

    orchestrator = RuntimeOrchestrator()

    async def event_generator():
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
