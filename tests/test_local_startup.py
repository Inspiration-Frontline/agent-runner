import pytest

from agent_runner import local_startup


class HealthyResponse:
    status: int = 200

    def __enter__(self) -> "HealthyResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_open_docs_when_server_is_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    opened_urls: list[str] = []
    monkeypatch.setattr(
        "agent_runner.local_startup.urllib.request.urlopen",
        lambda *args, **kwargs: HealthyResponse(),
    )
    monkeypatch.setattr("agent_runner.local_startup.webbrowser.open", opened_urls.append)

    local_startup.open_docs_when_ready(8000, timeout_seconds=0.1)

    assert opened_urls == ["http://127.0.0.1:8000/docs"]
