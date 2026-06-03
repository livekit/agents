"""Run async tests concurrently, with correct per-test output capture and tunable grouping.

This is a thin layer on top of the ``pytest-asyncio-concurrent`` plugin. That plugin
gives us the hard part -- collecting the async tests of a *group*, running them together
on a single event loop via ``asyncio.gather`` -- which is what makes a slow, I/O-bound
module like ``test_agent_session.py`` finish in a fraction of the wall-clock time.

It does **not** give us the things we need, which this module adds:

1. Correct output capture (the painful part).
   pytest's stdout/stderr/log capture is process-global: it is resumed/suspended around
   *one* test at a time and the buffer is read at the boundary. When the plugin runs N
   tests concurrently, the first one to finish suspends the shared capture and reads the
   shared buffer -- so output from the others leaks straight to the console and is
   attributed to the wrong test (or dropped). See the plugin's ``hook_wrapper_entered``.

   We fix this by replacing the plugin's per-call capture with a per-task capture keyed on
   a ``contextvars.ContextVar``. Each gathered test runs in its own asyncio task (with its
   own context), so a context var set at the start of the test -- and inherited by every
   background task it spawns -- tells our ``sys.stdout``/``sys.stderr`` proxies and our root
   log handler which test's buffer to write to. The captured text is attached to the test's
   report sections, so pytest shows it **only when that test fails**, exactly like normal.

2. Control over *which* tests run concurrently and *how they are grouped*, via two markers
   and a switch:

   - ``@pytest.mark.concurrent``            -- always run concurrently (unless off). Follows
     the global grouping level.
   - ``@pytest.mark.concurrent("<level>")`` -- same, but **force** this test's grouping level
     (``"class"`` / ``"module"`` / ``"category"`` / ``"session"``), so a test that touches
     shared state in a way only *some* peers tolerate can be isolated onto a finer group.
   - ``@pytest.mark.no_concurrent``         -- never run concurrently (process-global state,
     signal handlers, capsys, ...); honoured even under ``--concurrent``.
   - ``--concurrent [module|category|session]`` (or ``LK_TEST_CONCURRENCY=<level>``) runs
     *every* async test concurrently, grouped at that level (``all`` is an alias for
     ``session``); a bare ``--concurrent`` uses :data:`DEFAULT_LEVEL`. The default run
     (no flag) runs only ``concurrent``-marked tests; ``--no-concurrent`` (or
     ``LK_TEST_CONCURRENCY=0``) forces everything sequential and overrides every marker.

   Grouping levels, finest to coarsest -- a level decides which tests share one event loop:
     * ``class``    -- tests under the same class (or the same module, if not in a class).
     * ``module``   -- tests in the same file. **The default and the proven, safe level.**
     * ``category`` -- tests in the same category (``unit``, ``plugin:openai``, ...).
     * ``session``  -- every concurrent test in the run, on one loop.

   Markers apply at test, class, or module level (``pytestmark = pytest.mark.no_concurrent``).
   Tests using ``capsys``/``capfd`` are auto-excluded (those fixtures are process-global).

To switch the shipped default grouping level, change :data:`DEFAULT_LEVEL` (one line).

This module is loaded as a global plugin via ``-p tests.concurrency`` (see pyproject.toml);
the ``pytest_*`` hooks at the bottom drive everything. Grouping is automatic.

Notes / limitations:
- We keep ``asyncio_mode = "auto"`` (see pyproject). In auto mode pytest-asyncio claims every
  async test and would run our group sequentially, so :func:`repromote_collected` restores the
  concurrent items after pytest-asyncio has converted them. Every other module is untouched.
- ``category`` / ``session`` put tests from different modules on one loop. pytest sets up
  module/class/package-scoped fixtures on a per-node stack that those cross-module groups do
  not share, so :func:`effective_level` automatically caps a test that uses such a fixture to a
  level its fixtures can tolerate (``module`` for a module-scoped fixture, ``class`` for a
  class-scoped one). Session-scoped fixtures are always safe.
- Concurrent execution shares one event loop, so a test that hogs the loop can perturb another
  test's timing. If a concurrent run gets flaky, narrow the level (or use ``--no-concurrent``).
- pytest-xdist distributes by item and would split a group across workers; don't combine ``-n``
  with concurrent modules.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import io
import logging
import os
import shutil
import sys
import warnings
import weakref
from typing import TYPE_CHECKING, Any

import pytest
from pytest_asyncio_concurrent.grouping import (
    AsyncioConcurrentGroup,
    AsyncioConcurrentGroupMember,
)
from pytest_asyncio_concurrent.plugin import (
    _call_and_report,
    _call_runtest_async,
    _check_interactive_exception,
    _setup_child,
    _teardown_child,
)

if TYPE_CHECKING:
    from _pytest.config.argparsing import Parser
    from pluggy import Result

# ``caplog`` reads its handler/records out of these item-stash keys. We populate them per
# test ourselves (pytest's logging plugin never runs for concurrent members). Guarded so a
# pytest internals change degrades gracefully to "no caplog support" instead of a hard error.
try:
    from _pytest.logging import (
        LogCaptureHandler,
        caplog_handler_key,
        caplog_records_key,
    )

    _CAPLOG_SUPPORTED = True
except Exception:  # pragma: no cover - defensive
    LogCaptureHandler = None  # type: ignore[assignment,misc]
    _CAPLOG_SUPPORTED = False

_ENV_TOGGLE = "LK_TEST_CONCURRENCY"
_DEFAULT_LOG_FORMAT = "%(levelname)-8s %(name)s:%(filename)s:%(lineno)s %(message)s"


# --------------------------------------------------------------------------- #
# Grouping levels, markers, options, and the per-test concurrency decision
# --------------------------------------------------------------------------- #
# A *level* says which tests share one event loop. From finest (most isolated) to coarsest:
#   class    -> per class (or per module for a top-level test)
#   module   -> per file                         (the default; what `module` isolation means)
#   category -> per test category (unit, plugin:openai, ...)
#   session  -> the whole run, one loop
#
# Markers (per test, class, or module via ``pytestmark``):
#   @pytest.mark.concurrent            -- run concurrently, following the global level
#   @pytest.mark.concurrent("module")  -- run concurrently, but force this test's level
#   @pytest.mark.no_concurrent         -- never run concurrently (even under --concurrent)
# Switch:
#   (default)         only @concurrent tests run concurrently; everything else sequential
#   --concurrent[=L]  every async test runs concurrently at level L (bare => DEFAULT_LEVEL)
#   --no-concurrent   everything sequential, overriding --concurrent and every marker
# (LK_TEST_CONCURRENCY=<level>|all|0 mirrors the flags.)

#: The grouping level a bare ``--concurrent`` (and a plain ``@concurrent`` marker) uses.
#: ``module`` keeps every concurrent group inside a single file -- the proven, safe default.
#: Change this one line to ship a coarser default.
DEFAULT_LEVEL = "module"

_ALL_LEVELS = ("class", "module", "category", "session")
#: Levels accepted as the argument to ``--concurrent`` (``class`` is per-test only).
_GLOBAL_LEVELS = ("module", "category", "session")
_LEVEL_ALIASES = {"all": "session"}  # ``--concurrent all`` reads as ``session``

# Rank a level by coarseness; a higher number shares the loop more widely. Used to clamp a
# requested level down to what a test's fixtures can tolerate (see _fixture_level_cap).
_LEVEL_RANK = {"class": 1, "module": 2, "category": 4, "session": 5}
_RANK_LEVEL = {rank: level for level, rank in _LEVEL_RANK.items()}

# capsys/capfd are process-global, so a test that uses them can't share the loop with another
# that does ("cannot use capsys and capsys at the same time"); such tests are auto-excluded.
_CAPTURE_FIXTURES = frozenset({"capsys", "capfd", "capsysbinary", "capfdbinary", "capteesys"})

_OFF_TOKENS = frozenset({"0", "off", "false", "no"})
_ON_TOKENS = frozenset({"1", "on", "yes", "true"})


def register_markers(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "concurrent(level): run this async test concurrently with its peers; the optional "
        "level (class|module|category|session) forces its grouping, else it follows --concurrent.",
    )
    config.addinivalue_line(
        "markers",
        "no_concurrent: never run this async test concurrently (it relies on process-global "
        "state, signal handlers, capsys, etc.).",
    )


def add_toggle_option(parser: Parser) -> None:
    group = parser.getgroup("concurrency", "concurrent async test execution")
    group.addoption(
        "--concurrent",
        nargs="?",
        const=True,
        default=None,
        metavar="LEVEL",
        help="Run every async test concurrently, grouped at LEVEL "
        f"({'|'.join(_GLOBAL_LEVELS)}; 'all' == 'session'; default {DEFAULT_LEVEL!r}). "
        "Excludes @pytest.mark.no_concurrent tests.",
    )
    group.addoption(
        "--no-concurrent",
        action="store_true",
        default=False,
        help="Run everything sequentially, overriding --concurrent and every "
        "@pytest.mark.concurrent marker.",
    )


def _coerce_level(value: object) -> str:
    level = str(value).strip().lower()
    return _LEVEL_ALIASES.get(level, level)


def resolve_run_mode(config: pytest.Config) -> tuple[str, str]:
    """Resolve ``(selection, level)`` for this run.

    ``selection`` is one of:
      * ``"off"``       -- nothing runs concurrently (``--no-concurrent`` wins over everything).
      * ``"broad"``     -- every async test runs concurrently (``--concurrent`` given).
      * ``"selective"`` -- only ``@concurrent``-marked tests run concurrently (the default).
    ``level`` is the global grouping level to use for tests that don't force their own.

    Precedence: the CLI flags are explicit and beat the ambient env var; ``--no-concurrent`` is
    the absolute kill switch (it also beats ``--concurrent`` if both are somehow passed, and it
    overrides every ``@concurrent`` / forced-level marker, since "off" skips promotion entirely
    in :func:`repromote_collected`). Only when neither CLI flag is given do we consult the env.
    """
    # --no-concurrent wins over everything. This is what makes it robust.
    if config.getoption("--no-concurrent", default=False):
        return "off", DEFAULT_LEVEL

    flag = config.getoption("--concurrent", default=None)
    if flag is not None:  # --concurrent present (bare True, or an explicit level string)
        level = DEFAULT_LEVEL if flag is True else _validate_global_level(config, flag)
        return "broad", level

    # No CLI flag -> the LK_TEST_CONCURRENCY env var decides.
    env = os.environ.get(_ENV_TOGGLE, "").strip().lower()
    if env in _OFF_TOKENS:
        return "off", DEFAULT_LEVEL
    if env in _ON_TOKENS:
        return "broad", DEFAULT_LEVEL
    if env:  # a level name (or alias) in the env var
        return "broad", _validate_global_level(config, env)

    return "selective", DEFAULT_LEVEL


def _validate_global_level(config: pytest.Config, value: object) -> str:
    level = _coerce_level(value)
    if level not in _GLOBAL_LEVELS:
        raise pytest.UsageError(
            f"--concurrent: invalid level {value!r}; choose one of "
            f"{', '.join(_GLOBAL_LEVELS)} (or 'all')."
        )
    return level


def concurrency_enabled(config: pytest.Config) -> bool:
    """Whether any concurrent execution happens at all."""
    return resolve_run_mode(config)[0] != "off"


def should_run_concurrently(item: pytest.Function, selection: str) -> bool:
    """Whether this async test should run as a concurrent group member (see marker table above)."""
    if selection == "off":
        return False
    if item.get_closest_marker("no_concurrent") is not None:
        return False
    if _CAPTURE_FIXTURES.intersection(getattr(item._fixtureinfo, "names_closure", ())):
        return False
    if selection == "broad":
        return True
    return item.get_closest_marker("concurrent") is not None


def effective_level(item: pytest.Function, global_level: str) -> str:
    """The grouping level for ``item``: its forced marker level if any, else the global level,
    clamped to what the item's fixtures can tolerate (see :func:`_fixture_level_cap`)."""
    marker = item.get_closest_marker("concurrent")
    if marker is not None and marker.args:
        level = _coerce_level(marker.args[0])
        if level not in _ALL_LEVELS:
            warnings.warn(
                f"@pytest.mark.concurrent: invalid level {marker.args[0]!r} on {item.nodeid}; "
                f"choose one of {', '.join(_ALL_LEVELS)}. Falling back to {global_level!r}.",
                stacklevel=2,
            )
            level = global_level
    else:
        level = global_level

    cap = _fixture_level_cap(item)
    if _LEVEL_RANK[level] > _LEVEL_RANK[cap]:
        return cap  # too coarse for this test's fixtures; isolate it onto a finer group
    return level


def _fixture_level_cap(item: pytest.Function) -> str:
    """Coarsest level whose group node keeps all of ``item``'s scoped fixtures on pytest's
    setup stack. A cross-module (category/session) group is set up under a single module, so a
    member that needs a module-scoped fixture must stay at ``module`` (and a class-scoped one at
    ``class``); function- and session-scoped fixtures impose no limit."""
    cap = _LEVEL_RANK["session"]
    info = getattr(item, "_fixtureinfo", None)
    if info is None:
        return _RANK_LEVEL[cap]
    for defs in info.name2fixturedefs.values():
        if not defs:
            continue
        scope = defs[-1].scope
        if scope == "class":
            cap = min(cap, _LEVEL_RANK["class"])
        elif scope in ("module", "package"):
            cap = min(cap, _LEVEL_RANK["module"])
    return _RANK_LEVEL[cap]


def group_key(item: pytest.Function, level: str) -> str:
    """The ``asyncio_concurrent(group=...)`` key for ``item`` at ``level``. Members sharing a key
    run together on one loop. Keys are level-prefixed so different levels never collide."""
    if level == "session":
        return "session"
    if level == "category":
        return "category:" + _category_of(item)
    if level == "class":
        return "class:" + item.parent.nodeid  # the class, or the module for a top-level test
    return "module:" + item.nodeid.split("::", 1)[0]


def _category_of(item: pytest.Function) -> str:
    """The test's category (``unit``, ``plugin:openai``, ...) for category-level grouping."""
    try:
        from tests.conftest import CATEGORIES
    except Exception:  # pragma: no cover - defensive
        CATEGORIES = ()  # type: ignore[assignment]
    for marker in item.iter_markers():
        if marker.name in CATEGORIES:
            return f"{marker.name}:{marker.args[0]}" if marker.args else marker.name
    return "uncategorized"


def is_concurrent_member(node: object) -> bool:
    """True if ``node`` is a test running as part of a concurrent group.

    Used by fixtures that can't behave correctly when tests share one event loop
    (e.g. per-test leaked-task detection). When concurrency is toggled off the tests
    are ordinary items and this returns False, so such fixtures run normally.
    """
    return isinstance(node, AsyncioConcurrentGroupMember)


def _allow_mixed_parents() -> None:
    """Let a group hold members from different parents (needed for category/session levels).

    The plugin's ``AsyncioConcurrentGroup.add_child`` marks every member ``skip`` the moment it
    sees mixed parents -- a same-file assumption baked in at collection time, before our protocol
    override runs. We intend cross-module groups, so replace it with a version that keeps the
    bookkeeping but never skips. (Setup safety for those groups is handled by
    :func:`_fixture_level_cap`, not by refusing to run them.)
    """
    if getattr(AsyncioConcurrentGroup.add_child, "_lk_patched", False):
        return

    def add_child(self: AsyncioConcurrentGroup, item: AsyncioConcurrentGroupMember) -> None:
        item.group = self
        self.children.append(item)
        self.children_finalizer[item] = []

    add_child._lk_patched = True  # type: ignore[attr-defined]
    AsyncioConcurrentGroup.add_child = add_child  # type: ignore[method-assign]


# --------------------------------------------------------------------------- #
# Per-task capture machinery
# --------------------------------------------------------------------------- #
_active: contextvars.ContextVar[_TestCapture | None] = contextvars.ContextVar(
    "lk_concurrent_capture", default=None
)


class _TestCapture:
    """Output sink for a single concurrently-running test."""

    def __init__(self) -> None:
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        # pytest's own handler type so ``caplog.text`` / ``caplog.records`` keep working.
        self.log_handler = LogCaptureHandler() if _CAPLOG_SUPPORTED else None
        if self.log_handler is not None:
            self.log_handler.setLevel(0)


class _DispatchStream:
    """A ``sys.stdout``/``sys.stderr`` stand-in that routes writes to the active test.

    With no active test (e.g. pytest's own machinery between tests) it falls through to the
    real stream, so nothing is silently swallowed.
    """

    def __init__(self, wrapped: Any, attr: str) -> None:
        self._wrapped = wrapped
        self._attr = attr

    def write(self, data: str) -> int:
        cap = _active.get()
        if cap is None:
            return self._wrapped.write(data)
        return getattr(cap, self._attr).write(data)

    def flush(self) -> None:
        if _active.get() is None:
            self._wrapped.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class _RoutingLogHandler(logging.Handler):
    """Root-logger handler that routes each record to the active test's capture."""

    def emit(self, record: logging.LogRecord) -> None:
        cap = _active.get()
        if cap is None or cap.log_handler is None:
            return
        # Replicate the level gating Logger.callHandlers would normally do, so
        # ``caplog.set_level(...)`` keeps working.
        if record.levelno >= cap.log_handler.level:
            cap.log_handler.handle(record)


class _CaptureController:
    """Installs the dispatch streams + routing handler once, around a concurrent gather.

    The group's members all run on one event loop in one thread. Each member installs on
    entry and uninstalls on exit; a refcount keeps the proxies in place for the whole gather
    window (every member starts -- and so increments -- before any of them awaits to
    completion, so the count only returns to zero once the last member is done).
    """

    def __init__(self) -> None:
        self._depth = 0
        self._saved: tuple[Any, Any] | None = None
        self._handler: logging.Handler | None = None

    def __enter__(self) -> _CaptureController:
        if self._depth == 0:
            self._saved = (sys.stdout, sys.stderr)
            sys.stdout = _DispatchStream(sys.stdout, "stdout")  # type: ignore[assignment]
            sys.stderr = _DispatchStream(sys.stderr, "stderr")  # type: ignore[assignment]
            self._handler = _RoutingLogHandler()
            self._handler.setLevel(0)
            logging.getLogger().addHandler(self._handler)
        self._depth += 1
        return self

    def __exit__(self, *exc: object) -> None:
        self._depth -= 1
        if self._depth == 0:
            assert self._saved is not None
            sys.stdout, sys.stderr = self._saved
            if self._handler is not None:
                logging.getLogger().removeHandler(self._handler)
            self._saved = None
            self._handler = None


_controller = _CaptureController()


def _format_log(records: list[logging.LogRecord], config: pytest.Config) -> str:
    fmt = config.getini("log_format") or _DEFAULT_LOG_FORMAT
    formatter = logging.Formatter(fmt)
    return "\n".join(formatter.format(record) for record in records)


# --------------------------------------------------------------------------- #
# Hook bodies (called from the pytest_* hooks at the bottom)
# --------------------------------------------------------------------------- #
def repromote_collected(outcome: Result, config: pytest.Config) -> None:
    """Turn the async tests that should run concurrently into concurrent group members.

    In auto mode pytest-asyncio claims every async test (and would run it sequentially); after it
    has had its say we promote the ones :func:`should_run_concurrently` selects, keying each by
    its :func:`group_key` (level resolved per test, so forced-level markers win). The user-facing
    markers are ``concurrent`` / ``no_concurrent``; ``asyncio_concurrent`` (which the underlying
    plugin keys on) is an internal detail we attach here.

    Meant to be called from a ``pytest_pycollect_makeitem`` *hookwrapper* registered
    ``tryfirst`` (outermost), so it runs after pytest-asyncio has had its say.
    """
    selection, global_level = resolve_run_mode(config)
    if selection == "off":
        return
    result = outcome.get_result()
    if not result:
        return
    items = result if isinstance(result, list) else [result]
    changed = False
    for index, item in enumerate(items):
        if (
            isinstance(item, pytest.Function)
            and not isinstance(item, AsyncioConcurrentGroupMember)
            and inspect.iscoroutinefunction(item.obj)
            and should_run_concurrently(item, selection)
        ):
            member = AsyncioConcurrentGroupMember.promote_from_function(item)
            level = effective_level(member, global_level)
            member.add_marker(pytest.mark.asyncio_concurrent(group=group_key(member, level)))
            items[index] = member
            changed = True
    if changed:
        outcome.force_result(items)


# --------------------------------------------------------------------------- #
# Task ownership (per-test leaked-task attribution)
# --------------------------------------------------------------------------- #
# Concurrent group members share one event loop, so the usual "diff asyncio.all_tasks() around
# each test" can't tell whose task is whose -- a still-running task from test B looks like a
# leak from test A. Instead we tag every task with the test that created it: a contextvar
# carries the owning test's nodeid, and a task factory on the group's loop records it. The
# factory runs synchronously in the creating task's context, and asyncio tasks inherit their
# parent's context, so background tasks a test spawns (and their descendants) are attributed to
# that test too. fail_on_leaked_tasks then asks only about *its own* still-pending tasks.
_task_owner: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "lk_task_owner", default=None
)
_owner_by_task: weakref.WeakKeyDictionary[asyncio.Task[Any], str] = weakref.WeakKeyDictionary()
_group_loop: asyncio.AbstractEventLoop | None = None


def _tagging_task_factory(
    loop: asyncio.AbstractEventLoop, coro: Any, **kwargs: Any
) -> asyncio.Task[Any]:
    task: asyncio.Task[Any] = asyncio.Task(coro, loop=loop, **kwargs)
    owner = _task_owner.get()
    if owner is not None:
        _owner_by_task[task] = owner
    return task


@contextlib.contextmanager
def _task_ownership(loop: asyncio.AbstractEventLoop) -> Any:
    """Install the tagging task factory + expose the loop for the duration of a group."""
    global _group_loop
    previous_factory = loop.get_task_factory()
    previous_loop = _group_loop
    loop.set_task_factory(_tagging_task_factory)
    _group_loop = loop
    try:
        yield
    finally:
        loop.set_task_factory(previous_factory)
        _group_loop = previous_loop


def owned_pending_tasks(owner: str) -> list[asyncio.Task[Any]]:
    """Still-pending tasks created by test ``owner`` (see module note on task ownership).

    Lets fail_on_leaked_tasks attribute leaks per test even though the whole group shares one
    event loop. Returns ``[]`` when no group is running (e.g. sequential mode), so callers fall
    back to their normal whole-loop check.
    """
    loop = _group_loop
    if loop is None:
        return []
    # asyncio.all_tasks() already excludes finished tasks.
    return [task for task in asyncio.all_tasks(loop) if _owner_by_task.get(task) == owner]


# --------------------------------------------------------------------------- #
# Live progress line(s)
# --------------------------------------------------------------------------- #
# A running (not-yet-finished) test: pytest has no symbol for this, so use a dot in blue. As
# each test finishes the cell becomes pytest's own letter ("." / "F" / "x" / "X" / "s" / "E")
# in pytest's own colour -- i.e. it matches the normal progress output, just shown live.
_RUNNING_CELL = (".", {"blue": True})


class _GroupProgress:
    """A transient, in-place progress display for a concurrent group.

    One cell per test, drawn with pytest's standard status letter and colour, but live: blue
    ``.`` while running, then the final letter/colour as each test finishes. A single-file group
    is one line (rewritten with ``\\r``); a cross-file (category/session) group is one line per
    file, rewritten as a block with cursor moves -- the multi-line case the bigger levels need.
    Purely visual; the real reports are emitted separately. Only used for a non-verbose colour
    TTY (capture on); otherwise the caller streams each report as it completes instead, so
    :attr:`live` is False.
    """

    def __init__(self, terminalreporter: Any, config: pytest.Config) -> None:
        self._tw = terminalreporter._tw
        self._config = config
        verbose = config.get_verbosity() > 0
        capture_off = config.option.capture == "no"
        self.live: bool = self._tw.hasmarkup and not verbose and not capture_off
        self._cell: dict[Any, tuple[str, dict[str, bool]]] = {}
        self._done: set[Any] = set()
        self._order: list[Any] = []
        self._files: list[tuple[str, list[Any]]] = []  # (label, children) preserving order
        self._lines_drawn = 0
        # single-line bookkeeping (one-file groups), mirrors the original renderer.
        self._label = ""
        self._width = 0
        self._last_plain = ""

    def begin(self, children: list[Any]) -> None:
        self._order = list(children)
        self._cell = dict.fromkeys(children, _RUNNING_CELL)
        grouped: dict[str, list[Any]] = {}
        for child in self._order:
            grouped.setdefault(child.nodeid.split("::", 1)[0], []).append(child)
        self._files = list(grouped.items())
        if self._order:
            self._label = self._files[0][0] + " "
        self._render()

    def set_running(self, child: Any) -> None:
        self._cell[child] = _RUNNING_CELL
        self._render()

    def set_done(self, child: Any, report: pytest.TestReport) -> None:
        self._cell[child] = self._letter_and_markup(report)
        self._done.add(child)
        self._render()

    def _letter_and_markup(self, report: pytest.TestReport) -> tuple[str, dict[str, bool]]:
        """The exact status letter + colour pytest would use for this report."""
        _, letter, word = self._config.hook.pytest_report_teststatus(
            report=report, config=self._config
        )
        if isinstance(word, tuple):  # a plugin already chose the markup
            return letter, word[1]
        was_xfail = hasattr(report, "wasxfail")
        if report.passed:
            markup = {"yellow": True} if was_xfail else {"green": True}  # xpass vs pass
        elif report.failed:
            markup = {"red": True}
        elif report.skipped:
            markup = {"yellow": True}  # skipped / xfailed
        else:
            markup = {}
        return letter, markup

    def finish_line(self, is_last_group: bool) -> None:
        """Keep the finished display as the permanent record.

        For a single-line group on the final group we leave the cursor on the line and tell the
        writer its visible width, so pytest's end-of-loop ``[100%]`` fills this line's right edge
        exactly as it would a normal dot row. Every other case ends with a newline so the next
        group (or pytest's summary) starts fresh.
        """
        if not self.live:
            return
        if self._is_multiline():
            if self._lines_drawn:
                self._tw.line()
            return
        if not self._width:
            return
        if is_last_group:
            self._tw._current_line = self._last_plain
        else:
            self._tw.line()
        self._width = 0

    def _is_multiline(self) -> bool:
        # More than one file, and the block fits in the terminal -> render per-file lines.
        rows = shutil.get_terminal_size((80, 24)).lines
        return len(self._files) > 1 and len(self._files) <= max(1, rows - 2)

    def _cells_for(self, children: list[Any]) -> str:
        return "".join(
            self._tw.markup(letter, **markup) for letter, markup in map(self._cell.get, children)
        )

    def _render(self) -> None:
        if not self.live:
            return
        if len(self._files) > 1:
            self._render_block()
        else:
            self._render_single()

    def _render_single(self) -> None:
        counter = f" [{len(self._done)}/{len(self._order)}]"
        letters = "".join(self._cell[c][0] for c in self._order)
        plain = self._label + letters + counter
        if len(plain) >= self._tw.fullwidth:  # too wide for one cell per test
            body = plain = f"{self._label}running {len(self._done)}/{len(self._order)}"
        else:
            body = self._label + self._cells_for(self._order) + counter
        pad = " " * max(0, self._width - len(plain))  # erase a previously longer line
        self._tw.write("\r" + body + pad, flush=True)
        self._width = len(plain)
        self._last_plain = plain

    def _render_block(self) -> None:
        if not self._is_multiline():  # too tall: collapse to a single summary line
            label = f"{len(self._files)} files "
            plain = f"{label}running {len(self._done)}/{len(self._order)}"
            pad = " " * max(0, self._width - len(plain))
            self._tw.write("\r" + plain + pad, flush=True)
            self._width = len(plain)
            self._lines_drawn = 0
            return

        lines: list[str] = []
        for label, children in self._files:
            done = sum(1 for c in children if c in self._done)
            counter = f" [{done}/{len(children)}]"
            plain = f"{label} {''.join(self._cell[c][0] for c in children)}{counter}"
            if len(plain) >= self._tw.fullwidth:
                lines.append(f"{label} running {done}/{len(children)}")
            else:
                lines.append(f"{label} {self._cells_for(children)}{counter}")

        # Repaint the block in place: carriage-return, move up to the first line, then rewrite
        # each line and clear to end-of-line. hasmarkup is True here, so ANSI is safe.
        if self._lines_drawn:
            self._tw.write("\r")
            if self._lines_drawn > 1:
                self._tw.write(f"\033[{self._lines_drawn - 1}A")
        self._tw.write("\n".join(line + "\033[K" for line in lines), flush=True)
        self._lines_drawn = len(lines)


def run_group(group: AsyncioConcurrentGroup, nextgroup: AsyncioConcurrentGroup | None) -> object:
    """Run one concurrent group, fixing the plugin's event-loop handling and reporting.

    This mirrors ``pytest_asyncio_concurrent.plugin.pytest_runtest_protocol_async_group`` but
    differs in a few ways:

    * It captures the event loop **after** per-member setup instead of before. In auto mode,
      pytest-asyncio sets up its own loop while the members' fixtures are created, so the
      plugin's pre-setup capture no longer matches the loop ``asyncio.gather`` actually binds
      to -- which crashes with "future belongs to a different loop". Capturing after setup keeps
      the gather, ``run_until_complete`` and the test coroutines all on the same loop.

    * It reports each test **as it finishes** instead of all at once after the gather. The
      plugin emits ``logstart`` for every test up front and ``logreport`` for every test at the
      end, which (a) shows no progress until everything is done and (b) prints every name twice
      under ``-v``. Reporting on completion gives live progress and one clean line per test; for
      a non-verbose TTY we additionally drive a colour-coded :class:`_GroupProgress` (blue while
      running, final-status colour as each test lands), multi-line for cross-file groups.

    * It does not enforce same-parent groups: cross-module groups are intentional at the
      ``category`` / ``session`` levels (see :func:`_allow_mixed_parents`).

    Registered ``tryfirst`` for the plugin's ``firstresult`` hook, so this wins and the plugin's
    own implementation does not run. We reuse the plugin's (private) helpers for the individual
    setup/call/teardown steps so behaviour otherwise stays identical.
    """
    from _pytest.config import create_terminal_writer

    terminalreporter = group.config.pluginmanager.get_plugin("terminalreporter")
    progress = _GroupProgress(terminalreporter, group.config) if terminalreporter else None
    live = progress is not None and progress.live

    item_passed_setup: list[AsyncioConcurrentGroupMember] = []
    for child in group.children:
        report = _call_and_report(_setup_child(child), child, "setup")
        if report.passed:
            item_passed_setup.append(child)

    def _emit(child: pytest.Function, callinfo: pytest.CallInfo, report: pytest.TestReport) -> None:
        # logstart + logreport back-to-back => one clean line per test (no -v double-print).
        child.ihook.pytest_runtest_logstart(nodeid=child.nodeid, location=child.location)
        if _check_interactive_exception(call=callinfo, report=report):
            child.ihook.pytest_exception_interact(node=child, call=callinfo, report=report)
        child.ihook.pytest_runtest_logreport(report=report)

    if live:
        progress.begin(item_passed_setup)

    async def run_one(
        child: AsyncioConcurrentGroupMember,
    ) -> tuple[AsyncioConcurrentGroupMember, pytest.CallInfo, pytest.TestReport]:
        if live:
            progress.set_running(child)
        callinfo = await _call_runtest_async(child)
        report = child.ihook.pytest_runtest_makereport(item=child, call=callinfo)
        if live:
            progress.set_done(child, report)  # update the colour in place
        else:
            _emit(child, callinfo, report)  # stream the result (dots/names) as it lands
        return child, callinfo, report

    loop = _current_event_loop()
    # Tag tasks with their owning test (for leaked-task attribution) for the whole group,
    # including teardown -- that's when fail_on_leaked_tasks inspects the loop.
    with _task_ownership(loop):
        results = loop.run_until_complete(asyncio.gather(*[run_one(c) for c in item_passed_setup]))

        # In live mode the coloured display is the permanent visual, so discard the terminal
        # output of the (replayed) reports, teardown and logfinish -- stats / failures / other
        # plugins still see them via the hooks, just without a second dot row or stray line.
        original_tw = terminalreporter._tw if live else None
        if live:
            progress.finish_line(is_last_group=nextgroup is None)
            terminalreporter._tw = create_terminal_writer(terminalreporter.config, io.StringIO())
        try:
            if live:
                for child, callinfo, report in results:
                    _emit(child, callinfo, report)
            for child in group.children:
                _call_and_report(_teardown_child(child, nextgroup=nextgroup), child, "teardown")
                child.ihook.pytest_runtest_logfinish(nodeid=child.nodeid, location=child.location)
        finally:
            if original_tw is not None:
                terminalreporter._tw = original_tw

    return True


def _current_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


async def run_member_capturing(item: pytest.Function) -> object:
    """Run one concurrent test with per-task output capture.

    Replaces the plugin's ``pytest_runtest_call_async`` (registered ``tryfirst`` so this wins
    the ``firstresult`` race). We deliberately skip the plugin's ``hook_wrapper_entered`` --
    that is what re-enters pytest's broken-under-concurrency global capture -- and capture here.
    """
    if not inspect.iscoroutinefunction(item.obj):
        pytest.skip("Marking a sync function with @asyncio_concurrent is invalid.")

    testfunction = item.obj
    testargs = {arg: item.funcargs[arg] for arg in item._fixtureinfo.argnames}

    # Tag tasks this test spawns with its nodeid so fail_on_leaked_tasks can attribute leaks
    # to it even though the whole group shares one event loop.
    owner_token = _task_owner.set(item.nodeid)
    try:
        # ``-s`` / ``--capture=no``: the user asked to see output live -- don't buffer it.
        if item.config.option.capture == "no":
            return await testfunction(**testargs)

        cap = _TestCapture()
        # Wire caplog to *this* test's records so caplog-using tests stay correct (and isolated).
        if _CAPLOG_SUPPORTED and cap.log_handler is not None:
            item.stash[caplog_handler_key] = cap.log_handler
            item.stash[caplog_records_key] = {
                "setup": [],
                "call": cap.log_handler.records,
                "teardown": [],
            }

        with _controller:
            token = _active.set(cap)
            try:
                return await testfunction(**testargs)
            finally:
                _active.reset(token)
                if cap.stdout.getvalue():
                    item.add_report_section("call", "stdout", cap.stdout.getvalue())
                if cap.stderr.getvalue():
                    item.add_report_section("call", "stderr", cap.stderr.getvalue())
                if cap.log_handler is not None and cap.log_handler.records:
                    item.add_report_section(
                        "call", "log", _format_log(cap.log_handler.records, item.config)
                    )
    finally:
        _task_owner.reset(owner_token)


# --------------------------------------------------------------------------- #
# pytest hooks
#
# This module is loaded as a *global* plugin (``-p tests.concurrency`` in pyproject), not via
# conftest.py. That matters: the underlying plugin invokes ``pytest_runtest_protocol_async_group``
# through ``session.ihook``, which is scoped to the rootdir and would NOT see hooks defined in a
# subdirectory conftest like tests/conftest.py.
# --------------------------------------------------------------------------- #
def pytest_addoption(parser: Parser) -> None:
    add_toggle_option(parser)


def pytest_configure(config: pytest.Config) -> None:
    register_markers(config)
    _allow_mixed_parents()


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_pycollect_makeitem(collector: pytest.Collector, name: str, obj: object) -> Any:
    # Outermost wrapper: runs after pytest-asyncio's auto-mode conversion, so we can restore
    # the concurrent group members it converted back to ordinary sequential items.
    outcome = yield
    repromote_collected(outcome, collector.config)


@pytest.hookimpl(specname="pytest_runtest_protocol_async_group", tryfirst=True)
def pytest_runtest_protocol_async_group(
    group: AsyncioConcurrentGroup, nextgroup: AsyncioConcurrentGroup | None
) -> object:
    # Wins the plugin's firstresult hook; fixes its event-loop handling (see run_group).
    return run_group(group, nextgroup)


@pytest.hookimpl(specname="pytest_runtest_call_async", tryfirst=True)
async def pytest_runtest_call_async(item: pytest.Function) -> object:
    # Wins the plugin's firstresult hook; adds per-task output capture.
    return await run_member_capturing(item)
