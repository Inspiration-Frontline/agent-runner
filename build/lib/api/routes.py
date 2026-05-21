from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from config import ChatRequest
from runtime.orchestrator import RuntimeOrchestrator

router = APIRouter()


@router.post("/chat/stream")
async def chat_stream(request: Request, chat_request: ChatRequest):
    orchestrator = RuntimeOrchestrator()

    async def event_generator():
        async for event in orchestrator.run(chat_request, request):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
