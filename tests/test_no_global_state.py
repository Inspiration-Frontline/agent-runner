import ast
from pathlib import Path


def test_source_has_no_global_statements_or_mutable_container_definitions() -> None:
    source_root = Path(__file__).parents[1] / "src" / "agent_runner"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                violations.append(f"{path}:{node.lineno}: global statement")
        scopes = [tree, *(node for node in tree.body if isinstance(node, ast.ClassDef))]
        for scope in scopes:
            for node in scope.body:
                if isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(node.value, ast.List | ast.Dict | ast.Set):
                    violations.append(f"{path}:{node.lineno}: mutable module/class container")

    assert violations == []
