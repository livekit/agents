"""Guard test: a participant identity or room name must not reach a log message body.

``telemetry.pii._PIIFilteringLogProcessor`` strips log attributes whose key carries a
dot-delimited ``pii`` segment (``lk.pii.<name>``), and ``_TraceLevelLoggingHandler``
does the same for the records the framework's own handler creates. Neither one touches
the record body, which is the reason ``REVIEW.md`` asks for a static message with the
value moved to a structured attribute: a value interpolated into the body is exported
verbatim even when the project has redaction enabled.

Identities and room names are the two categories checked here. ``REVIEW.md`` names both,
and both already have a marked constant in ``telemetry/trace_types.py``
(``ATTR_PARTICIPANT_IDENTITY``, ``ATTR_ROOM_NAME``), so the fix is always the same:

    logger.info("participant disconnected", extra={"lk.pii.participant_identity": ident})

The check reads the interpolated expression, never a value, so it is deliberately
narrow. Only a plain name or attribute chain counts: ``f"{p.identity}"`` is flagged,
while ``f"{identity is not None}"`` carries no identity and ``f"{redact(ident)}"`` has
already passed through a formatter this test cannot see into. Widening it means teaching
it about a value source, not adding another name.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# livekit-agents/ and livekit-plugins/ are the shipped library. examples/ is sample code
# that a user reads and edits, and it does not run inside a redacting exporter.
SOURCE_ROOTS = ("livekit-agents", "livekit-plugins")

LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})

# `p.identity`, `self._human_agent_identity`, `room.name`, `self._room_name`.
SENSITIVE_EXPR_RE = re.compile(r"\w*identity$|room\.name$|\w*room_name$", re.IGNORECASE)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _log_message_expressions(tree: ast.AST) -> list[tuple[int, str]]:
    """Every expression interpolated into a logger call's message, with its line."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in LOG_LEVELS:
            continue
        target = node.func.value
        name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
        if "log" not in name.lower() or not node.args:
            continue

        message = node.args[0]
        if isinstance(message, ast.JoinedStr):
            values = [part.value for part in message.values if isinstance(part, ast.FormattedValue)]
        elif (
            isinstance(message, ast.Constant)
            and isinstance(message.value, str)
            and "%s" in message.value
        ):
            values = list(node.args[1:])
        else:
            continue

        for value in values:
            if isinstance(value, (ast.Name, ast.Attribute)):
                found.append((value.lineno, ast.unparse(value)))
    return found


def test_no_identity_or_room_name_in_a_log_message_body() -> None:
    root = _repo_root()
    offenders: list[str] = []

    for source_root in SOURCE_ROOTS:
        for path in sorted((root / source_root).rglob("*.py")):
            if "tests" in path.parts or "examples" in path.parts:
                continue
            try:
                # bytes rather than text: ast.parse honours a `# -*- coding: -*-` declaration
                # the way the interpreter does, instead of assuming every source file is UTF-8.
                tree = ast.parse(path.read_bytes(), filename=str(path))
            except SyntaxError:  # a plugin pinned to a newer syntax than this interpreter
                continue
            for lineno, expression in _log_message_expressions(tree):
                if SENSITIVE_EXPR_RE.search(expression):
                    offenders.append(f"{path.relative_to(root)}:{lineno}  {expression}")

    assert not offenders, (
        "a participant identity or room name is interpolated into a log message body, "
        "where no exporter can redact it:\n  " + "\n  ".join(offenders) + "\n"
        "Keep the message static and move the value to a structured attribute, e.g. "
        'extra={"lk.pii.participant_identity": ident} or extra={"lk.pii.room_name": name}.'
    )
