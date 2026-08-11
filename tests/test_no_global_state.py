import ast
from pathlib import Path


def test_source_has_no_global_statements() -> None:
    source_root = Path(__file__).parents[1] / "src" / "agent_runner"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                violations.append(f"{path}:{node.lineno}: global statement")
    assert violations == []


def test_openai_adapter_methods_do_not_exceed_fifty_lines() -> None:
    path = Path(__file__).parents[1] / "src" / "agent_runner" / "runtime" / "openai_agents_sdk_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OpenAIAgentsSdkAdapter"
    )
    violations = {
        node.name: node.end_lineno - node.lineno + 1
        for node in adapter.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.end_lineno is not None
        and node.end_lineno - node.lineno + 1 > 50
    }

    assert violations == {}
