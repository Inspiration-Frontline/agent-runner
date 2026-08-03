"""
API routes module for agent runner service.

This module defines the FastAPI routes for agent chat interactions,
providing streaming response endpoints for real-time agent communication.
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

from agent_runner.api.streaming import DoneEvent, ErrorEvent, PersistedEvent, SavingEvent, TokenDeltaEvent, UsageEvent
from agent_runner.config import CancelChatRequest, ChatRequest, get_settings
from agent_runner.conversation import ConversationBusyError
from agent_runner.observability.tracing import Span, extract_trace_context, get_tracer, trace_json
from agent_runner.runtime.cancellation import conversation_cancellation_registry
from agent_runner.runtime.orchestrator import RuntimeOrchestrator

router = APIRouter()


class CancelChatResponse(BaseModel):
    """Acknowledgement returned after requesting Conversation cancellation."""

    cancelled: bool


@dataclass
class ChatTraceStats:
    """Aggregated SSE evidence attached once to the request span."""

    event_count: int = 0
    token_event_count: int = 0
    response_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    saving: bool = False
    persisted: bool = False
    done: bool = False
    error_count: int = 0

    def record(self, span: Span, event: object) -> None:
        """Record one semantic event without creating a span per token delta."""
        self.event_count += 1
        if isinstance(event, TokenDeltaEvent):
            self.token_event_count += 1
            self.response_chars += len(event.content or "")
        elif isinstance(event, UsageEvent):
            prompt_tokens = event.prompt_tokens or 0
            completion_tokens = event.completion_tokens or 0
            total_tokens = event.total_tokens or 0
            self.input_tokens += prompt_tokens
            self.output_tokens += completion_tokens
            self.total_tokens += total_tokens
            span.add_event(
                "chat.usage",
                {
                    "gen_ai.usage.input_tokens": prompt_tokens,
                    "gen_ai.usage.output_tokens": completion_tokens,
                    "gen_ai.usage.total_tokens": total_tokens,
                },
            )
        elif isinstance(event, SavingEvent):
            self.saving = True
            span.add_event("chat.saving")
        elif isinstance(event, PersistedEvent):
            self.persisted = True
            span.add_event("chat.persisted")
        elif isinstance(event, DoneEvent):
            self.done = True
            span.add_event("chat.done")
        elif isinstance(event, ErrorEvent):
            self.error_count += 1
            attributes = {"error.type": event.error_code or "UNKNOWN", "error.phase": event.phase or "unknown"}
            span.add_event("chat.error", attributes)

    def finish(self, span: Span) -> None:
        """Attach bounded aggregate counters and the terminal stream outcome."""
        span.set_attribute("chat.event_count", self.event_count)
        span.set_attribute("chat.token_event_count", self.token_event_count)
        span.set_attribute("chat.response_chars", self.response_chars)
        span.set_attribute("gen_ai.usage.input_tokens", self.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", self.output_tokens)
        span.set_attribute("gen_ai.usage.total_tokens", self.total_tokens)
        span.set_attribute("chat.saving", self.saving)
        span.set_attribute("chat.persisted", self.persisted)
        span.set_attribute("chat.done", self.done)
        span.set_attribute("chat.error_count", self.error_count)
        span.set_attribute("chat.outcome", "completed" if self.done else "failed" if self.error_count else "interrupted")


def trusted_user_id(x_user_id: str | None) -> int:
    """Parse the gateway-provided authenticated user header and reject spoofed values."""
    if x_user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is required.")
    try:
        user_id = int(x_user_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is invalid."
        ) from error
    if user_id <= 0:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Trusted user identity is invalid.")
    return user_id


@router.post("/chat/cancel", status_code=status.HTTP_202_ACCEPTED)
async def cancel_chat(
    cancel_request: CancelChatRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> CancelChatResponse:
    """Request cancellation of the active generation for one authenticated Conversation."""
    user_id = trusted_user_id(x_user_id)
    cancelled = False
    for _ in range(10):
        cancelled = conversation_cancellation_registry.cancel(user_id, cancel_request.conversation_id)
        if cancelled:
            break
        await asyncio.sleep(0.05)
    return CancelChatResponse(cancelled=cancelled)


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> StreamingResponse:
    """
    Stream agent chat responses through SSE (Server-Sent Events).

    This endpoint accepts a chat request and returns a streaming response
    containing real-time agent output, including text tokens, tool calls,
    and completion markers.

    Args:
        request: The FastAPI request object for connection management.
        chat_request: The chat request containing:
            - conversation_id: Optional conversation ID
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
            "conversation_id": "conv_example123",
            "message": "Hello, how can I help?"
        }
    """
    user_id = trusted_user_id(x_user_id)

    orchestrator = RuntimeOrchestrator()
    try:
        await orchestrator.acquire_conversation(chat_request.conversation_id)
    except ConversationBusyError as error:
        await orchestrator.close()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "CONVERSATION_BUSY",
                "message": str(error),
            },
        ) from error

    async def event_generator() -> AsyncGenerator[str]:
        """Yield orchestrator events as SSE frames while preserving disconnect cleanup."""
        parent_context = extract_trace_context(dict(request.headers))
        attributes = {
            "conversation.id": chat_request.conversation_id,
            "conversation.file_count": len(chat_request.file_ids),
            "conversation.reference_count": len(chat_request.references),
            "chat.locale": chat_request.ui_locale,
            "chat.message_chars": len(chat_request.message),
            "chat.attachment_only": not bool(chat_request.message),
        }
        settings = get_settings()
        if settings.otel_capture_content:
            attributes["agentbreaker.chat.request"] = trace_json(
                chat_request.model_dump(mode="json"), settings.otel_content_max_chars
            )
        with get_tracer().span(
            "chat.request", attributes, parent_context=parent_context, kind=SpanKind.SERVER
        ) as span:
            stats = ChatTraceStats()
            span.add_event("chat.accepted")
            try:
                async for event in orchestrator.run(chat_request, user_id, request):
                    stats.record(span, event)
                    yield f"data: {event.model_dump_json()}\n\n"
            finally:
                stats.finish(span)
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
