"""Keep exception details in structured log fields, where PII redaction can reach them."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log"}


class ExceptionLogVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.exceptions: dict[str, str] = {}
        self.calls: list[tuple[ast.Call, str]] = []

    def exception_in(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute) and node.attr in {"error", "exception", "message"}:
            return ast.unparse(node)
        # Exception class names are safe metadata, unlike the exception's free-form text.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "type"
            and len(node.args) == 1
        ):
            return None
        if isinstance(node, ast.Name):
            return self.exceptions.get(node.id)
        return next(
            (exc for child in ast.iter_child_nodes(node) if (exc := self.exception_in(child))),
            None,
        )

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        previous = self.exceptions.copy()
        if node.name:
            self.exceptions[node.name] = node.name
        self.generic_visit(node)
        self.exceptions = previous

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        previous = self.exceptions.copy()
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self.exceptions.pop(arg.arg, None)
            if arg.annotation and any(
                isinstance(n, ast.Name) and n.id.endswith(("Error", "Exception"))
                for n in ast.walk(arg.annotation)
            ):
                self.exceptions[arg.arg] = arg.arg
        self.generic_visit(node)
        self.exceptions = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        exception = self.exception_in(node.value)
        if isinstance(node.value, ast.Call):
            func = node.value.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else ""
            )
            if name.endswith(("Error", "Exception")) or name in {"exception", "_error_from_frame"}:
                exception = next((t.id for t in node.targets if isinstance(t, ast.Name)), None)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if exception:
                    self.exceptions[target.id] = exception
                else:
                    self.exceptions.pop(target.id, None)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in LOG_METHODS
            and "log" in ast.unparse(node.func.value).lower()
        ):
            args = node.args[1:] if node.func.attr == "log" else node.args
            # Structured fields are allowed. Python logging only formats positional arguments.
            exception = next((exc for arg in args if (exc := self.exception_in(arg))), None)
            if exception:
                self.calls.append((node, exception))
            for keyword in node.keywords:
                if keyword.arg != "extra" or not isinstance(keyword.value, ast.Dict):
                    continue
                for key in keyword.value.keys:
                    if isinstance(key, ast.Constant) and key.value in {
                        "error",
                        "err",
                        "errors",
                        "exception",
                    }:
                        self.calls.append((node, f"extra[{key.value!r}]"))
        self.generic_visit(node)


def check_source(source: str) -> list[tuple[int, str]]:
    visitor = ExceptionLogVisitor()
    visitor.visit(ast.parse(source))
    return [(call.lineno, exception) for call, exception in visitor.calls]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(arg) for arg in sys.argv[1:]] or [
        root / "livekit-agents",
        root / "livekit-plugins",
    ]
    failed = False
    for path in paths:
        for file in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
            for line, exception in check_source(file.read_text(encoding="utf-8")):
                print(
                    f"{file}:{line}: unredactable exception {exception!r}; use exc_info or lk.pii.*"
                )
                failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
