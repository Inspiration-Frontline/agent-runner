import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def load_fixture_module() -> ModuleType:
    path = Path(__file__).parent / "integration" / "mcp_streamable_http_fixture.py"
    spec = importlib.util.spec_from_file_location("mcp_streamable_http_fixture", path)

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load MCP fixture module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


async def invoke(app: ASGIApp, headers: dict[str, str], query_string: str = "") -> tuple[int, bytes]:
    sent: list[Message] = []
    received = False

    async def receive() -> Message:
        nonlocal received

        if not received:
            received = True

            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": query_string.encode("utf-8"),
        "root_path": "",
        "headers": [(name.lower().encode("ascii"), value.encode("utf-8")) for name, value in headers.items()],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8765),
        "state": {},
    }
    await app(scope, receive, send)
    status = next(cast(int, item["status"]) for item in sent if item["type"] == "http.response.start")
    body = b"".join(cast(bytes, item.get("body", b"")) for item in sent if item["type"] == "http.response.body")

    return status, body


def build_middleware(**requirement_values: Any) -> ASGIApp:
    module = load_fixture_module()
    requirement = module.StaticCredentialRequirement(**requirement_values)

    async def accepted(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware_factory = cast(Callable[[ASGIApp, Any], ASGIApp], module.StaticCredentialMiddleware)

    return middleware_factory(accepted, requirement)


@pytest.mark.parametrize("status", [401, 403])
async def test_bearer_credential_accepts_correct_and_rejects_missing_or_wrong(status: int) -> None:
    secret = "fixture-bearer-secret"
    app = build_middleware(
        header_name="Authorization",
        header_value=f"Bearer {secret}",
        rejection_status=status,
    )

    assert await invoke(app, {"Authorization": f"Bearer {secret}"}) == (204, b"")
    missing = await invoke(app, {})
    wrong = await invoke(app, {"Authorization": "Bearer wrong"})

    assert missing[0] == status
    assert wrong[0] == status
    assert secret not in missing[1].decode("utf-8")
    assert secret not in wrong[1].decode("utf-8")


async def test_custom_header_credential_matrix() -> None:
    app = build_middleware(header_name="X-API-Key", header_value="custom-secret", rejection_status=401)

    assert await invoke(app, {"X-API-Key": "custom-secret"}) == (204, b"")
    assert (await invoke(app, {}))[0] == 401
    assert (await invoke(app, {"X-API-Key": "wrong"}))[0] == 401


async def test_url_query_credential_matrix() -> None:
    app = build_middleware(query_name="api_key", query_value="url-secret", rejection_status=403)

    assert await invoke(app, {}, "api_key=url-secret") == (204, b"")
    assert (await invoke(app, {}))[0] == 403
    assert (await invoke(app, {}, "api_key=wrong"))[0] == 403


async def test_credential_file_rotation_is_visible_to_new_requests(tmp_path: Path) -> None:
    credential_file = tmp_path / "credential.txt"
    credential_file.write_text("Bearer first", encoding="utf-8")
    app = build_middleware(
        header_name="Authorization",
        credential_file=str(credential_file),
        rejection_status=401,
    )

    assert await invoke(app, {"Authorization": "Bearer first"}) == (204, b"")

    credential_file.write_text("Bearer second", encoding="utf-8")

    assert (await invoke(app, {"Authorization": "Bearer first"}))[0] == 401
    assert await invoke(app, {"Authorization": "Bearer second"}) == (204, b"")
