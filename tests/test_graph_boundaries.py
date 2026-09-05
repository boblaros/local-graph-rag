from __future__ import annotations

import ast
from pathlib import Path


def test_graph_implementation_has_no_evaluation_or_script_dependencies():
    graph_root = Path(__file__).parents[1] / "src" / "graph"
    forbidden: list[tuple[str, str]] = []
    for path in sorted(graph_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                parts = set(name.split("."))
                if parts.intersection({"evaluation", "scripts"}):
                    forbidden.append((path.name, name))

    assert forbidden == []
