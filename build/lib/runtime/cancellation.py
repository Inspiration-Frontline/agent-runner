import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CancellationToken:
    _cancelled: bool = field(default=False, init=False)
    _callbacks: list[Any] = field(default_factory=list, init=False)

    def cancel(self):
        self._cancelled = True
        for callback in self._callbacks:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback())
            else:
                callback()

    def is_cancelled(self) -> bool:
        return self._cancelled

    def add_callback(self, callback: Any):
        self._callbacks.append(callback)

    def remove_callback(self, callback: Any):
        if callback in self._callbacks:
            self._callbacks.remove(callback)


class CancellationManager:
    def __init__(self):
        self._tokens: dict[str, CancellationToken] = {}

    def create_token(self, token_id: str | None = None) -> CancellationToken:
        import uuid

        token_id = token_id or str(uuid.uuid4())
        token = CancellationToken()
        self._tokens[token_id] = token
        return token

    def get_token(self, token_id: str) -> CancellationToken | None:
        return self._tokens.get(token_id)

    def cancel_token(self, token_id: str):
        token = self._tokens.get(token_id)
        if token:
            token.cancel()

    async def cleanup(self, token: CancellationToken):
        token.cancel()
        await asyncio.sleep(0)

    def remove_token(self, token_id: str):
        if token_id in self._tokens:
            del self._tokens[token_id]
