from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_runner.mcps.catalog import McpServerCatalog, McpServerCatalogDocument


class FakeSecrets:
    def resolve(self, reference: str) -> str:
        return {"${secret:MCP_TOKEN}": "resolved-token"}[reference]


def test_catalog_accepts_standard_envelope_and_resolves_secrets(tmp_path: Path) -> None:
    catalog_path = tmp_path / "mcp.json"
    catalog_path.write_text(
        '{"mcpServers":{"secured":{"url":"https://example.test/${secret:MCP_TOKEN}/mcp",'
        '"headers":{"Authorization":"${secret:MCP_TOKEN}"}}}}',
        encoding="utf-8",
    )

    server = McpServerCatalog.from_file(catalog_path, FakeSecrets()).resolve("secured")

    assert server.url == "https://example.test/resolved-token/mcp"
    assert server.headers == {"Authorization": "resolved-token"}


@pytest.mark.parametrize(
    "profile",
    [
        {"command": "python", "args": ["server.py"]},
        {"url": "file:///tmp/mcp"},
        {"url": "https://example.test/mcp", "headers": {"Authorization": "Bearer literal"}},
        {"url": "https://example.test/mcp", "policy": "REQUIRE_APPROVAL"},
    ],
)
def test_catalog_rejects_out_of_scope_or_unsafe_profiles(profile: object) -> None:
    with pytest.raises(ValidationError):
        McpServerCatalogDocument.model_validate({"mcpServers": {"bad": profile}})


def test_catalog_rejects_unknown_server() -> None:
    catalog = McpServerCatalog.empty()

    with pytest.raises(ValueError, match="Unknown MCP server"):
        catalog.resolve("missing")
