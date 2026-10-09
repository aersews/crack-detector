import ast
from pathlib import Path


def test_all_python_files_parse():
    root = Path(__file__).parents[1]
    for path in root.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
