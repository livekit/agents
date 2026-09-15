"""Run a module's async tests concurrently on one event loop, with correct per-test isolation.

A thin layer over the `pytest-asyncio-concurrent` plugin, which supplies the hard part:
collecting a *group*'s async tests and running them together via `asyncio.gather` (so a slow,
I/O-bound module like `test_agent_session.py` finishes in a fraction of the wall time). What the
plugin does *not* give us, and this module adds:

1. Per-task output capture. pytest's stdout/stderr/log/caplog capture is process-global, so under
   concurrency output leaks or is misattributed to whichever test finishes first. We key capture
   on a `contextvars.ContextVar` instead: each gathered test runs in its own task/context, so the
   stream proxies, root-log handler, and `caplog` records route to the right test's buffer and
   surface only when that test fails -- as normal.

2. Per-test reporting + a live progress line. The plugin reports nothing until the whole group is
   done (and double-prints names under `-v`); we report each test the moment it finishes, and on
   a non-verbose color TTY draw a transient one-cell-per-test line with a blue head orbiting the
   still-running cells, each cell freezing to its status letter on completion.

3. Per-test leaked-task attribution. On a shared loop, diffing `asyncio.all_tasks()` can't tell
   whose task is whose. A task factory tags each task with its creating test's nodeid (via a
   contextvar, inherited by spawned tasks); `fail_on_leaked_tasks` then inspects only its own
   still-pending tasks. See :func:`owned_pending_tasks`.

4. Concurrency control, via two markers and a switch:

   - `@pytest.mark.concurrent`    -- always run this async test concurrently (unless off).
   - `@pytest.mark.no_concurrent` -- never (process-global state, signal handlers, capsys, ...);
     honored even under `--concurrent`.
   - `--concurrent` (or `LK_TEST_CONCURRENCY=all`) runs *every* async test concurrently except
     `no_concurrent` ones; the default runs only `concurrent` ones; `--no-concurrent` (or
     `LK_TEST_CONCURRENCY=0`) forces everything sequential.

   Markers apply at test, class, or module level (`pytestmark = ...`); `capsys`/`capfd` users
   are auto-excluded (process-global fixtures).

5. The fixes that make pytest-asyncio's `auto` mode cooperate:

   - :func:`repromote_collected` restores the concurrent members after auto mode claims (and would
     sequentialize) every async test.
   - One concurrent group per *module*, so module-level tests and every class in the file share a
     loop (the plugin otherwise pins a group to its first member's parent).
   - The gather's loop is captured *after* per-member setup: in auto mode pytest-asyncio builds its
     own loop during fixture setup, and capturing before it crashes with "future belongs to a
     different loop".

Loaded as a *global* plugin via `-p tests.concurrency` (see pyproject.toml), not conftest.py: the
plugin drives `pytest_runtest_protocol_async_group` through the rootdir-scoped `session.ihook`,
which would not see hooks defined in a subdirectory conftest.

Limitations:
- Module-scoped groups break *class*-scoped fixtures on concurrent members (the finalizer needs a
  Class node that module grouping drops); function/module/session scopes are fine.
- The shared loop couples timing: a test that hogs it can perturb another's. If a concurrent run
  gets flaky, force it sequential with `--no-concurrent`.
- pytest-xdist distributes by item and would split a group across workers, so `-n` disables
  concurrency entirely: :func:`pytest_configure` unregisters the underlying plugin (whose
  runtestloop wrapper and collection regrouping otherwise desync xdist's per-worker item indexing,
  crashing the worker with an IndexError), and the default mode falls back to sequential with a
  one-line notice at session start (see :func:`pytest_report_header`); an explicit `--concurrent`
  under `-n` is rejected.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import io
import logging
import math
import os
import sys
import threading
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Literal

import pytest

try:
    from pytest_asyncio_concurrent.grouping import (
        AsyncioConcurrentGroup,
        AsyncioConcurrentGroupMember,
        PytestAsyncioConcurrentGroupingWarning,
    )
    from pytest_asyncio_concurrent.plugin import (
        _call_and_report,
        _call_runtest_async,
        _check_interactive_exception,
        _setup_child,
        _teardown_child,
    )

    PYTEST_ASYNCIO_CONCURRENT_INSTALLED = True
except ImportError:
    PYTEST_ASYNCIO_CONCURRENT_INSTALLED = False

if TYPE_CHECKING:
    from _pytest.config.argparsing import Parser
    from pluggy import Result

# `caplog` reads its handler/records out of these item-stash keys. We populate them per
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
_CAPTURE_FIXTURES = frozenset({"capsys", "capfd", "capsysbinary", "capfdbinary", "capteesys"})


def register_markers(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "concurrent: run this async test concurrently with its peers."
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
        action="store_true",
        default=False,
        help="Run every async test concurrently, except @pytest.mark.no_concurrent ones.",
    )
    group.addoption(
        "--no-concurrent",
        action="store_true",
        default=False,
        help="Run all tests sequentially, even @pytest.mark.concurrent ones.",
    )


def _xdist_active(config: pytest.Config) -> bool:
    """True when pytest-xdist is in play -- on the controller *and* inside each worker.

    xdist splits a run by *item*, which would scatter a concurrent group (one shared event loop)
    across workers. The controller carries `-n` / `--dist`; a worker process carries neither
    (xdist clears them so the worker doesn't itself re-distribute) but does carry `workerinput`.
    The concurrent plugin must be disabled on the worker too -- that's where its runtestloop
    wrapper fights xdist's remote loop (`items[nextitem_index]` -> IndexError). `-n0` / no
    xdist leaves this False everywhere.
    """
    if hasattr(config, "workerinput"):
        return True
    if getattr(config.option, "dist", "no") not in ("no", None):
        return True
    return bool(getattr(config.option, "numprocesses", None))


def _requested_mode(config: pytest.Config) -> Literal["default", "all", "off"]:
    """Run mode the user asked for via flags/env, before the xdist override: off/default/all."""
    env = os.environ.get(_ENV_TOGGLE, "").strip().lower()
    if config.getoption("--no-concurrent", default=False) or env in ("0", "off", "false", "no"):
        return "off"
    if config.getoption("--concurrent", default=False) or env in ("all", "1", "on", "yes"):
        return "all"
    return "default"


def concurrency_mode(config: pytest.Config) -> Literal["default", "all", "off"]:
    """Resolve the effective run mode.

    Under pytest-xdist (`-n`) concurrency can't work -- a group shares one event loop and xdist
    would split it across workers -- so any non-off request is forced to `"off"` here. The
    incompatible `--concurrent` + `-n` case is rejected outright in :func:`pytest_configure`;
    the default mode is downgraded quietly (notice in :func:`pytest_report_header`).
    """
    requested = _requested_mode(config)
    if requested != "off" and _xdist_active(config):
        return "off"
    return requested


def concurrency_enabled(config: pytest.Config) -> bool:
    """Whether any concurrent execution happens at all."""
    return concurrency_mode(config) != "off"


def should_run_concurrently(item: pytest.Function, mode: Literal["default", "all", "off"]) -> bool:
    """Whether this async test should run as a concurrent group member (see marker table above)."""
    if mode == "off":
        return False
    if item.get_closest_marker("no_concurrent") is not None:
        return False
    if _CAPTURE_FIXTURES.intersection(getattr(item._fixtureinfo, "names_closure", ())):
        return False
    if mode == "all":
        return True
    return item.get_closest_marker("concurrent") is not None


def is_concurrent_member(node: object) -> bool:
    """True if `node` is a test running as part of a concurrent group.

    Used by fixtures that can't behave correctly when tests share one event loop
    (e.g. per-test leaked-task detection). When concurrency is toggled off the tests
    are ordinary items and this returns False, so such fixtures run normally.
    """
    return PYTEST_ASYNCIO_CONCURRENT_INSTALLED and isinstance(node, AsyncioConcurrentGroupMember)


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
        # pytest's own handler type so `caplog.text` / `caplog.records` keep working.
        self.log_handler = LogCaptureHandler() if _CAPLOG_SUPPORTED else None
        if self.log_handler is not None:
            self.log_handler.setLevel(0)


class _DispatchStream:
    """A `sys.stdout`/`sys.stderr` stand-in that routes writes to the active test.

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
        # `caplog.set_level(...)` keeps working.
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


@contextlib.contextmanager
def _capture_subprocess_fds(terminalreporter: Any) -> Any:
    """Capture file-descriptor-level output for the duration of a concurrent gather.

    The per-task :class:`_CaptureController` only swaps `sys.stdout`/`sys.stderr`, so it sees
    in-process `print`s but not writes a *subprocess* (or C extension) makes straight to the
    inherited stdout/stderr file descriptors -- e.g. a spawned child's `print()`. With pytest's
    own fd capture suspended while tests run, those writes hit the real terminal and corrupt the
    live progress line (and leak into CI logs).

    So for the gather window we point fds 1/2 at a pipe and drain it on a background thread, while
    rerouting the *parent's* terminal writer to the preserved fds -- so dots, the live line and
    reports still render; only out-of-process writes are diverted. The buffer is shared by every
    member (they share the fds) and so can't be attributed to one test; it's yielded back for the
    caller to surface only if the group failed, matching pytest's "show captured output on failure
    only" behaviour. A no-op (`b""`) is yielded when there are no real fds to capture.
    """
    tw = terminalreporter._tw
    original_file = tw._file
    try:
        original_file.flush()
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # pragma: no cover - best effort flush
        pass

    try:
        saved_out, saved_err = os.dup(1), os.dup(2)
    except OSError:  # pragma: no cover - no real fds (e.g. fully in-memory streams)
        yield lambda: b""
        return

    pipe_r, pipe_w = os.pipe()
    os.dup2(pipe_w, 1)
    os.dup2(pipe_w, 2)
    os.close(pipe_w)

    # Parent-side terminal output bypasses the captured fds via the preserved stdout.
    preserved = os.fdopen(saved_out, "w", buffering=1, closefd=True)
    tw._file = preserved

    buffer = bytearray()

    def _pump() -> None:
        while True:
            try:
                chunk = os.read(pipe_r, 65536)
            except OSError:
                break
            if not chunk:  # EOF: every write end (fds 1/2) has been restored/closed
                break
            buffer.extend(chunk)

    pump = threading.Thread(target=_pump, name="lk-concurrent-fd-capture", daemon=True)
    pump.start()

    try:
        yield lambda: bytes(buffer)
    finally:
        try:
            preserved.flush()
        except Exception:  # pragma: no cover
            pass
        # Restore the real fds; this drops the last refs to the pipe's write end, so the pump
        # sees EOF and exits.
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_err)
        pump.join(timeout=2.0)
        if not pump.is_alive():
            os.close(pipe_r)
        tw._file = original_file
        preserved.close()  # closes saved_out


def _emit_captured_subprocess_output(
    terminalreporter: Any,
    results: list[tuple[Any, pytest.CallInfo, pytest.TestReport]],
    data: bytes,
) -> None:
    """Surface a concurrent group's captured fd output -- but only if a member failed.

    The output can't be attributed to a single member (they shared the fds), so it's shown as a
    group-level section rather than per test, mirroring pytest's failure-only capture display.
    """
    if not data or not any(report.failed for _, _, report in results):
        return
    text = data.decode("utf-8", "replace")
    terminalreporter.write_sep("-", "Captured subprocess stdout/stderr (concurrent group)")
    terminalreporter._tw.write(text if text.endswith("\n") else text + "\n")


def _format_log(records: list[logging.LogRecord], config: pytest.Config) -> str:
    fmt = config.getini("log_format") or _DEFAULT_LOG_FORMAT
    formatter = logging.Formatter(fmt)
    return "\n".join(formatter.format(record) for record in records)


# --------------------------------------------------------------------------- #
# Hook bodies (called from conftest.py)
# --------------------------------------------------------------------------- #
def repromote_collected(outcome: Result, config: pytest.Config) -> None:
    """Turn the async tests that should run concurrently into concurrent group members.

    In auto mode pytest-asyncio claims every async test (and would run it sequentially); after it
    has had its say we promote the ones :func:`should_run_concurrently` selects, grouping each by
    its parent (module or class) so a group's members share a parent as the plugin requires. The
    user-facing markers are `concurrent` / `no_concurrent`; `asyncio_concurrent` (which the
    underlying plugin keys on) is an internal detail we attach here.

    Meant to be called from a `pytest_pycollect_makeitem` *hookwrapper* registered
    `tryfirst` (outermost), so it runs after pytest-asyncio has had its say.
    """
    mode = concurrency_mode(config)
    if mode == "off":
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
            and should_run_concurrently(item, mode)
        ):
            member = AsyncioConcurrentGroupMember.promote_from_function(item)
            # Group by module (possible thanks to _enable_cross_parent_grouping)
            group_name = item.getparent(pytest.Module).nodeid
            member.add_marker(pytest.mark.asyncio_concurrent(group=group_name))
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
    """Still-pending tasks created by test `owner` (see module note on task ownership).

    Lets fail_on_leaked_tasks attribute leaks per test even though the whole group shares one
    event loop. Returns `[]` when no group is running (e.g. sequential mode), so callers fall
    back to their normal whole-loop check.
    """
    loop = _group_loop
    if loop is None:
        return []
    # asyncio.all_tasks() already excludes finished tasks.
    return [task for task in asyncio.all_tasks(loop) if _owner_by_task.get(task) == owner]


# --------------------------------------------------------------------------- #
# Live progress line
# --------------------------------------------------------------------------- #
_ORBIT_INTERVAL = 0.01  # seconds between frames; the only thing that wakes the shared loop
_ORBIT_TRAIL = 10  # base comet length (lit dots behind the head); widened to cover fast steps
_ORBIT_SPEED = 1.0  # default dots advanced per frame (speed); 2.0 == twice as fast, same frame rate
_RUNNING_MARKUP = {"blue": True}


# Cell sentinel: this test is in flight -- render the orbiting head for it (blank braille otherwise).
_RUNNING = object()


def _emit(child: pytest.Function, callinfo: pytest.CallInfo, report: pytest.TestReport) -> None:
    # logstart + logreport back-to-back => one clean line per test (no -v double-print).
    child.ihook.pytest_runtest_logstart(nodeid=child.nodeid, location=child.location)
    if _check_interactive_exception(call=callinfo, report=report):
        child.ihook.pytest_exception_interact(node=child, call=callinfo, report=report)
    child.ihook.pytest_runtest_logreport(report=report)


def _orbit_path(cells: list[int]) -> list[tuple[int, int]]:
    """Braille-dot positions tracing the rectangle around `cells` clockwise from the top-left.

    `cells` are the running columns, left to right; finished columns are simply absent, so the
    head jumps over them and the walls sit on the first/last running column. Per cell the dots are
    1,4 across the top, 7,8 across the bottom, 1,2,3,7 down the left and 4,5,6,8 down the right.
    Top dots run left->right, down the right wall of the last cell, bottom dots right->left, then
    up the left wall of the first cell. Shared corners are not repeated.
    """
    if not cells:
        return []
    first, last = cells[0], cells[-1]
    top = [(cell, bit) for cell in cells for bit in (0x01, 0x08)]
    right = [(last, bit) for bit in (0x10, 0x20, 0x80)]
    bottom = [(cell, bit) for cell in reversed(cells) for bit in (0x80, 0x40)][1:]
    left = [(first, 0x04), (first, 0x02)]
    return top + right + bottom + left


class _GroupProgress:
    """Owns how one concurrent group reports -- both strategies behind one interface.

    The mode is decided once in `__init__` and never leaks to the caller:

    * live (non-verbose colour TTY, capture on): a transient in-place line, one cell per test.
      While a test runs its cell is blank braille; collectively the running cells show a colored
      head orbiting just the running cells -- finished columns are dropped from the loop, so the
      head jumps over them (see :func:`_orbit_path`). Test starts and completions only mutate
      cell state; the line is redrawn on the animation timer (see :meth:`ticking`), so a cell
      freezes to its final status letter/colour on the next frame and a burst of completions
      costs one redraw rather than one per test. Per-test reports are withheld until the line is
      frozen, then replayed through a silenced terminal writer so stats/failures/other plugins
      still see them via the hooks without a second dot row; the cumulative `[ XX%]` is stamped
      on afterwards (see :meth:`_finish_line`).
    * streaming (everything else): no line; each report is emitted the moment its test finishes,
      so pytest prints dots/names live as usual.

    `run_group` drives this purely through `begin` / `on_start` / `on_finish` /
    `ticking` / `finalize` and never branches on the mode.
    """

    def __init__(self, terminalreporter: Any, config: pytest.Config) -> None:
        self._terminalreporter = terminalreporter
        self._tw = terminalreporter._tw
        self._config = config
        verbose = config.get_verbosity() > 0
        capture_off = config.option.capture == "no"
        self._live: bool = self._tw.hasmarkup and not verbose and not capture_off
        self._cell: dict[Any, Any] = {}
        self._done: set[Any] = set()
        self._order: list[Any] = []
        self._lit: dict[int, int] = {}
        self._label = ""
        self._width = 0
        self._last_plain = ""
        self._phase = 0.0
        self._step = _ORBIT_SPEED
        self._trail = max(_ORBIT_TRAIL, math.ceil(self._step))
        self._marked: dict[Any, str] = {}
        self._glyph_cache: dict[str, str] = {}
        self._count_width = 0
        self._fullwidth = 0

    # -- driven by run_group ------------------------------------------------- #
    def begin(self, children: list[Any]) -> None:
        self._order = list(children)
        self._cell = dict.fromkeys(children, _RUNNING)
        self._marked = {}
        self._count_width = len(str(len(self._order)))
        self._fullwidth = self._tw.fullwidth
        if self._order:
            self._label = self._order[0].nodeid.split("::", 1)[0] + " "
        self._render()

    def on_start(self, child: Any) -> None:
        self._cell[child] = _RUNNING

    def on_finish(
        self, child: pytest.Function, callinfo: pytest.CallInfo, report: pytest.TestReport
    ) -> None:
        letter, markup = self._letter_and_markup(report)
        if self._live:
            self._marked[child] = self._tw.markup(letter, **markup)
        self._cell[child] = (letter, markup)
        self._done.add(child)
        if not self._live:
            _emit(child, callinfo, report)

    @contextlib.asynccontextmanager
    async def ticking(self) -> Any:
        """Step the orbiting head on a timer while the group's gather runs (live only).

        Also the only place the live line is redrawn while the gather runs: on_start/on_finish
        merely mutate cell state, and the final redraw here (after the head is cancelled) freezes
        the end state that _finish_line then stamps the percentage onto.
        """
        if not self._live:
            yield
            return
        # No owner tag (created in the gather's own task context), so it can't look like a
        # leaked test task; cancelled and awaited here.
        orbit = asyncio.ensure_future(self._drive_orbit())
        try:
            yield
        finally:
            orbit.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await orbit
            self._render()

    @contextlib.contextmanager
    def finalize(
        self,
        results: list[tuple[AsyncioConcurrentGroupMember, pytest.CallInfo, pytest.TestReport]],
        *,
        is_last_group: bool,
    ) -> Any:
        """Wrap teardown: replay withheld reports silently, then stamp the line's progress (live only).

        Streaming mode already emitted each report in :meth:`on_finish`, so it just runs the body.
        The reports replay through a throwaway writer so stats/failures/other plugins still see
        them via the hooks without a second dot row -- but that throwaway writer is also where
        pytest's own `[ XX%]` would land, so the cumulative percentage is stamped onto the real
        writer afterwards (see :meth:`_finish_line`).
        """
        if not self._live:
            yield
            return
        from _pytest.config import create_terminal_writer

        original_tw = self._terminalreporter._tw
        self._terminalreporter._tw = create_terminal_writer(
            self._terminalreporter.config, io.StringIO()
        )
        try:
            for child, callinfo, report in results:
                _emit(child, callinfo, report)
            yield
        finally:
            self._terminalreporter._tw = original_tw
            self._finish_line(is_last_group=is_last_group)

    # -- rendering ----------------------------------------------------------- #
    async def _drive_orbit(self) -> None:
        while True:
            await asyncio.sleep(_ORBIT_INTERVAL)
            self._advance_orbit()
            self._render()

    def _advance_orbit(self) -> None:
        """Advance the head by `self._step` dots along the loop of still-running cells.

        Distance per frame is the speed knob: raising it moves the head faster without touching
        the frame rate (`_ORBIT_INTERVAL`), so no extra event-loop wakeups and no added load on
        the gathered tests. The head is a phase in [0, 1) so that when a finished cell drops out
        and the loop shortens it stays at the same fraction of the way round rather than
        teleporting (which flickered when a burst of completions landed between two frames). The
        trail widens to `self._trail` >= step so a fast head stays a continuous comet instead of
        a dot hopping with gaps.
        """
        running = [
            index for index, child in enumerate(self._order) if self._cell[child] is _RUNNING
        ]
        path = _orbit_path(running)
        if not path:
            self._lit = {}
            return
        period = len(path)
        self._phase = (self._phase + self._step / period) % 1.0
        head = int(self._phase * period)
        lit: dict[int, int] = {}
        for offset in range(self._trail):
            cell, dot = path[(head - offset) % period]
            lit[cell] = lit.get(cell, 0) | dot
        self._lit = lit

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

    def _finish_line(self, is_last_group: bool) -> None:
        """Stamp the cumulative `[ XX%]` onto the finished colour line and keep it as the record.

        The colour line carries embedded markup, so its raw width is wrong -- reset
        `_current_line` to the plain text first, then let pytest fill the same right-aligned
        progress message it writes after an ordinary dot row, so concurrent groups read exactly
        like the sequential lines. The final group is left untouched: pytest's own end-of-loop
        `[100%]` lands on it, and writing it here too would double.
        """
        if not (self._live and self._width):
            return
        self._tw._current_line = self._last_plain
        if is_last_group:
            return
        self._terminalreporter._write_progress_information_filling_space()
        self._tw.line()

    def _render(self) -> None:
        if not self._live:
            return
        counter = f" [{len(self._done):>{self._count_width}}/{len(self._order)}]"
        glyphs: list[str] = []
        marked: list[str] = []
        for index, child in enumerate(self._order):
            cell = self._cell[child]
            if cell is _RUNNING:
                glyph = chr(0x2800 + self._lit.get(index, 0))
                glyphs.append(glyph)
                marked.append(self._render_glyph(glyph))
            else:
                glyphs.append(cell[0])
                marked.append(self._marked[child])
        plain = self._label + "".join(glyphs) + counter
        if len(plain) >= self._fullwidth:  # too wide for one cell per test
            body = plain = f"{self._label}running {len(self._done)}/{len(self._order)}"
        else:
            body = self._label + "".join(marked) + counter
        pad = " " * max(0, self._width - len(plain))  # erase a previously longer line
        self._tw.write("\r" + body + pad, flush=True)
        self._width = len(plain)
        self._last_plain = plain

    def _render_glyph(self, glyph: str) -> str:
        cached = self._glyph_cache.get(glyph)
        if cached is None:
            cached = self._tw.markup(glyph, **_RUNNING_MARKUP)
            self._glyph_cache[glyph] = cached
        return cached


def run_group(group: AsyncioConcurrentGroup, nextgroup: AsyncioConcurrentGroup | None) -> object:
    """Run one concurrent group, fixing the plugin's event-loop handling and reporting.

    This mirrors `pytest_asyncio_concurrent.plugin.pytest_runtest_protocol_async_group` but
    differs in two ways:

    * It captures the event loop **after** per-member setup instead of before. In auto mode,
      pytest-asyncio sets up its own loop while the members' fixtures are created, so the
      plugin's pre-setup capture no longer matches the loop `asyncio.gather` actually binds
      to -- which crashes with "future belongs to a different loop". Capturing after setup keeps
      the gather, `run_until_complete` and the test coroutines all on the same loop.

    * It reports each test **as it finishes** instead of all at once after the gather, via the
      :class:`_GroupProgress` reporter -- which also drives a colour-coded live line for a
      non-verbose TTY. The plugin instead emits `logstart` for every test up front and
      `logreport` for every test at the end, which shows no progress until everything is done
      and prints every name twice under `-v`.

    Registered `tryfirst` for the plugin's `firstresult` hook, so this wins and the plugin's
    own implementation does not run. We reuse the plugin's (private) helpers for the individual
    setup/call/teardown steps so behaviour otherwise stays identical.
    """
    if not group.children_have_same_parent:
        for child in group.children:
            child.add_marker("skip")
        warnings.warn(
            PytestAsyncioConcurrentGroupingWarning(
                f"Asyncio Concurrent Group [{group.name}] has children from different parents, "
                "skipping all of its children."
            ),
            stacklevel=2,
        )

    terminalreporter = group.config.pluginmanager.get_plugin("terminalreporter")
    progress = _GroupProgress(terminalreporter, group.config)

    item_passed_setup: list[AsyncioConcurrentGroupMember] = []
    for child in group.children:
        report = _call_and_report(_setup_child(child), child, "setup")
        if report.passed:
            item_passed_setup.append(child)

    progress.begin(item_passed_setup)

    async def run_one(
        child: AsyncioConcurrentGroupMember,
    ) -> tuple[AsyncioConcurrentGroupMember, pytest.CallInfo, pytest.TestReport]:
        progress.on_start(child)
        callinfo = await _call_runtest_async(child)
        report = child.ihook.pytest_runtest_makereport(item=child, call=callinfo)
        progress.on_finish(child, callinfo, report)
        return child, callinfo, report

    async def _run_all() -> list[
        tuple[AsyncioConcurrentGroupMember, pytest.CallInfo, pytest.TestReport]
    ]:
        async with progress.ticking():
            return await asyncio.gather(*[run_one(c) for c in item_passed_setup])

    loop = _current_event_loop()
    # Under `-s` the user asked to see output live, so don't divert the subprocess fds.
    capture_subprocess = group.config.option.capture != "no"
    # Tag tasks with their owning test (for leaked-task attribution) for the whole group,
    # including teardown -- that's when fail_on_leaked_tasks inspects the loop.
    with _task_ownership(loop):
        if capture_subprocess:
            with _capture_subprocess_fds(terminalreporter) as captured:
                results = loop.run_until_complete(_run_all())
            captured_subprocess_output = captured()
        else:
            results = loop.run_until_complete(_run_all())
            captured_subprocess_output = b""
        with progress.finalize(results, is_last_group=nextgroup is None):
            for child in group.children:
                _call_and_report(_teardown_child(child, nextgroup=nextgroup), child, "teardown")
                child.ihook.pytest_runtest_logfinish(nodeid=child.nodeid, location=child.location)
        _emit_captured_subprocess_output(terminalreporter, results, captured_subprocess_output)

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

    Replaces the plugin's `pytest_runtest_call_async` (registered `tryfirst` so this wins
    the `firstresult` race). We deliberately skip the plugin's `hook_wrapper_entered` --
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
        # `-s` / `--capture=no`: the user asked to see output live -- don't buffer it.
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


# HACK: some callers invoke pytest without installing dev dependencies,
# so we might end up here without the underlying plugin installed.
# Keeping the previous behavior intact here.
if PYTEST_ASYNCIO_CONCURRENT_INSTALLED:
    # --------------------------------------------------------------------------- #
    # pytest hooks
    #
    # This module is loaded as a *global* plugin (`-p tests.concurrency` in pyproject), not via
    # conftest.py. That matters: the underlying plugin invokes `pytest_runtest_protocol_async_group`
    # through `session.ihook`, which is scoped to the rootdir and would NOT see hooks defined in a
    # subdirectory conftest like tests/conftest.py.
    # --------------------------------------------------------------------------- #
    def pytest_addoption(parser: Parser) -> None:
        add_toggle_option(parser)

    def _disable_concurrent_plugin(config: pytest.Config) -> None:
        """Unregister pytest-asyncio-concurrent so it doesn't fight xdist.

        Its runtestloop wrapper and collection regrouping assume a single in-process run; under xdist
        they desync the worker's positional item indexing -- `items[nextitem_index]` runs off the
        end and the worker dies with an IndexError (surfaced as an INTERNALERROR on the controller).
        With the plugin gone, plain pytest-asyncio (auto mode) handles the async tests and xdist
        distributes them by item as usual. Matching on the module name covers both the package and its
        `.plugin` submodule, and does not match regular `pytest_asyncio`.
        """
        pm = config.pluginmanager
        for plugin in list(pm.get_plugins()):
            module = getattr(plugin, "__name__", "") or getattr(type(plugin), "__module__", "")
            if isinstance(module, str) and module.startswith("pytest_asyncio_concurrent"):
                pm.unregister(plugin)

    def pytest_configure(config: pytest.Config) -> None:
        if not PYTEST_ASYNCIO_CONCURRENT_INSTALLED:
            return

        register_markers(config)
        if _xdist_active(config):
            # xdist distributes by item; a concurrent group can't span workers. Reject an explicit
            # request, otherwise disable concurrency (the underlying plugin included) and go sequential.
            if _requested_mode(config) == "all":
                raise pytest.UsageError(
                    "--concurrent (or LK_TEST_CONCURRENCY=all) is incompatible with pytest-xdist "
                    "(-n): a concurrent group shares a single event loop, which xdist would split "
                    "across worker processes. Drop -n to run concurrently, or drop --concurrent."
                )
            _disable_concurrent_plugin(config)
            return
        _enable_cross_parent_grouping()

    def pytest_report_header(config: pytest.Config) -> str | None:
        # Once per session, on the controller, at the top of the run: announce that -n forced the
        # default concurrent mode to sequential. (--concurrent under -n already errored in configure.)
        if hasattr(config, "workerinput"):
            return None
        if _xdist_active(config) and _requested_mode(config) != "off":
            return "concurrency: disabled under pytest-xdist (-n); async tests run sequentially."
        return None

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

    def _enable_cross_parent_grouping() -> None:
        """One concurrent group per module: module-level tests plus every class in the file.

        The plugin pins a group to the parent of its first member and skips any group whose
        members don't all share it. Here the group sits at the Module and mixed parents are
        allowed. A class-scoped fixture on a concurrent member fails at setup as a result --
        its finalizer needs the Class node on SetupState, which module grouping removes.
        Function-, module-, and session-scoped fixtures are unaffected.
        """
        if getattr(AsyncioConcurrentGroup, "_lk_cross_parent_grouping", False):
            return

        original_init = AsyncioConcurrentGroup.__init__

        def init_at_module(self, parent, originalname):
            original_init(self, parent.getparent(pytest.Module) or parent, originalname)

        def add_child_keep_group(self, item):
            item.group = self
            self.children.append(item)
            self.children_finalizer[item] = []

        AsyncioConcurrentGroup.__init__ = init_at_module
        AsyncioConcurrentGroup.add_child = add_child_keep_group
        AsyncioConcurrentGroup._lk_cross_parent_grouping = True
