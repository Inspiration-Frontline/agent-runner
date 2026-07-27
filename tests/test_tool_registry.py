from typing import Any

from agent_runner.tools.registry import ToolDefinition, ToolRegistry, ToolSourceType


def create_definition(
    *,
    parameters: dict[str, Any] | None = None,
    strict: bool = False,
    source_type: ToolSourceType = ToolSourceType.INTERNAL,
) -> ToolDefinition:
    return ToolDefinition(
        tool_key="builtin.web_search",
        tool_name="web_search",
        description="Search the web",
        parameters=parameters or {"type": "object", "properties": {}},
        strict=strict,
        source_type=source_type,
    )


def test_definition_hash_is_stable_for_equivalent_json_schema() -> None:
    first = create_definition(parameters={"type": "object", "properties": {"query": {"type": "string"}}})
    second = create_definition(parameters={"properties": {"query": {"type": "string"}}, "type": "object"})

    assert first.definition_hash == second.definition_hash
    assert len(first.definition_hash) == 64


def test_definition_hash_changes_with_normalized_metadata() -> None:
    non_strict = create_definition(strict=False)
    strict = create_definition(strict=True)

    assert non_strict.definition_hash != strict.definition_hash


def test_registration_is_idempotent_and_provider_type_is_derived() -> None:
    registry = ToolRegistry()
    registry.register(create_definition())
    registry.register(create_definition())

    definitions = registry.get_by_source(ToolSourceType.INTERNAL)
    specs = registry.get_tool_specs(["builtin.web_search"])

    assert len(definitions) == 1
    assert specs == [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            },
        }
    ]
