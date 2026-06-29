"""Deterministic virtual-time execution for timing-coupled async tests.

Voice tests drive the agent through fake STT/TTS/VAD/audio components in *real* wall-clock
time: the agent's endpointing, interruption timeouts and audio play-out clock all schedule
against the event loop and measure durations with `time.time()` / `time.perf_counter()`.
That makes the suite sensitive to scheduler jitter -- a shared event loop, or a slow/contended
CI runner, perturbs the relative timing and the count assertions race.

A test marked `@pytest.mark.virtual_time` runs instead on a :class:`_VirtualTimeLoop`, whose
clock only advances when the loop goes idle (it jumps straight to the next scheduled timer). No
real time passes, so a 30-second scripted conversation completes in microseconds and, crucially,
*deterministically* -- the order of timer firings is fixed by their scheduled times, not by who
won a CPU slice.

The loop only virtualizes `loop.time()` (what `asyncio.sleep`/`call_later` read). The
agent and fakes also read wall time directly, so :func:`_virtual_wall_clock` additionally points
`time.time()` / `time.perf_counter()` at the running loop's virtual clock *for the duration of
the marked test only* -- a scoped patch, so no global state leaks between tests.

Wiring (conftest.py re-exports these so pytest discovers them):
- :func:`event_loop_policy` is pytest-asyncio's override point for the loop a test runs on; it
  returns the virtual-time policy for marked tests and the default policy otherwise (so non-marked
  tests behave exactly as before).
- :func:`_virtual_wall_clock` is autouse; a no-op unless the test carries the marker. It patches
  the `time` module for inline reads and, for `Field(default_factory=time.time)` references
  captured at import (which a module patch cannot reach), rewrites the validators in place via
  :func:`_patch_model_factories` so `created_at` timestamps also run on virtual time.
"""

from __future__ import annotations

import asyncio
import contextlib
import selectors
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any
from unittest import mock

import pytest

# A fixed, realistic wall-clock base so patched `time.time()` still looks like a real epoch.
# Only differences are ever asserted under virtual time, so the exact value is irrelevant.
_WALL_CLOCK_EPOCH = 1_700_000_000.0

# The real clocks, captured at import (before any patch is installed) so the fallbacks below can
# hand real wall time to callers that aren't on the virtual loop, and as the signature for
# discovering which captured `default_factory` references to redirect.
_REAL_TIME = time.time
_REAL_PERF = time.perf_counter

# Smallest separation between two wall-clock reads at the same loop instant. The clock does not
# move between reads within one tick, but synchronously-created items still need distinct, ordered
# `created_at` values (ChatContext reconciliation orders by it), so reads are nudged to stay
# strictly increasing (see _VirtualClock.read). Tiny enough that accumulation across a test stays
# far below any timing assertion's tolerance.
_TICK_EPSILON = 1e-6


def _virtual_loop() -> _VirtualTimeLoop | None:
    """The virtual-time loop driving the current call, or None.

    Returns None when there is no running loop (e.g. a background thread such as the OpenTelemetry
    metrics exporter) or the running loop is an ordinary real one. The clock patch lives on the
    `time` module process-wide for the duration of a virtual-time test, so it is reachable from
    any thread; only callers actually on the virtual loop should see virtual time -- everyone else
    must get the real wall clock, otherwise a stray read freezes at the epoch (which, for the
    metrics exporter, stamps batches with a 2023 timestamp the gateway then rejects).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return loop if isinstance(loop, _VirtualTimeLoop) else None


def _now() -> float:
    # Off the virtual loop (no running loop, or a real one) fall back to the real clock, so stray
    # reads from other threads see true wall time.
    loop = _virtual_loop()
    return _WALL_CLOCK_EPOCH + loop.monotonic_read() if loop is not None else _REAL_TIME()


def _perf() -> float:
    loop = _virtual_loop()
    return loop.monotonic_read() if loop is not None else _REAL_PERF()


# Maps a captured real clock to its virtual replacement. A model field built with
# `Field(default_factory=time.time)` captures the bare `time.time` at import; patching the
# `time` module later cannot reach that captured reference, so :func:`_patch_model_factories`
# rewrites it in the validator instead.
_VIRTUAL_FOR = {_REAL_TIME: _now, _REAL_PERF: _perf}


def is_virtual_time(node: pytest.Item) -> bool:
    # `--real-time` disables virtual time globally; the `real_time` marker opts out a single test.
    if node.config.getoption("--real-time", default=False):
        return False
    return (
        node.get_closest_marker("virtual_time") is not None
        and node.get_closest_marker("real_time") is None
    )


def _to_plain(obj: Any) -> Any:
    """Materialize a (possibly lazy `MockCoreSchema`) core schema into a fresh plain dict tree.

    pydantic exposes `__pydantic_core_schema__` as a `Mapping` proxy whose `deepcopy` stays
    a proxy, but `SchemaValidator` needs real dicts. Rebuilding every mapping/list also gives an
    independent copy we can edit without touching the class's schema.
    """
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def _model_node(schema: dict[str, Any]) -> dict[str, Any]:
    """Descend `definitions` / validator wrappers to the `model` node holding the fields."""
    node = schema
    while node.get("type") != "model":
        node = node["schema"]
    return node


# Built once, lazily: `model -> (patched_validator, original_validator)`. The patched validator
# is invariant across tests (its `default_factory` is `_now`, which resolves the running loop
# at call time), so we pay the schema-edit + validator construction a single time and then only
# swap references per test.
_PATCHED: dict[type, tuple[Any, Any]] = {}


def _build_patch_cache() -> None:
    """Construct, once, a virtual-clock validator for every model with a captured real clock."""
    if _PATCHED:
        return
    from pydantic_core import SchemaValidator

    for model, fields in _discover_factory_patches().items():
        schema = _to_plain(model.__pydantic_core_schema__)
        node = _model_node(schema)
        field_schemas = node["schema"]["fields"]
        for name, factory in fields.items():
            field_schemas[name]["schema"]["default_factory"] = factory
        # _use_prebuilt=False forces a build from our edited schema instead of reusing the
        # validator cached on the class.
        patched = SchemaValidator(schema, node.get("config"), _use_prebuilt=False)
        _PATCHED[model] = (patched, model.__pydantic_validator__)


@contextlib.contextmanager
def _patched_model_clocks() -> Iterator[None]:
    """Swap every captured-clock model onto its virtual-clock validator, then restore.

    All such models move together so `created_at`-ordered structures (e.g. `ChatContext`,
    which inserts by `created_at`) stay consistent -- a partial swap would interleave virtual
    and real timestamps and scramble item order. The per-test cost is just the reference swaps.
    """
    _build_patch_cache()
    try:
        for model, (patched, _original) in _PATCHED.items():
            model.__pydantic_validator__ = patched  # type: ignore[attr-defined]
        yield
    finally:
        for model, (_patched, original) in _PATCHED.items():
            model.__pydantic_validator__ = original  # type: ignore[attr-defined]


def _discover_factory_patches() -> dict[type, dict[str, Callable[[], Any]]]:
    """Find every pydantic model field whose `default_factory` is a real clock, grouped by model.

    Self-maintaining: any new event/chat-context model with a `default_factory=time.time` field
    is picked up automatically, no registry to update. Imported lazily so importing this module
    stays cheap and free of livekit import-order constraints.
    """
    from pydantic import BaseModel

    from livekit.agents.inference import interruption
    from livekit.agents.llm import chat_context
    from livekit.agents.voice import events

    patches: dict[type, dict[str, Callable[[], Any]]] = {}
    for module in (events, chat_context, interruption):
        for obj in vars(module).values():
            if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
                continue
            fields = {
                name: _VIRTUAL_FOR[info.default_factory]  # type: ignore[index]
                for name, info in obj.model_fields.items()
                if info.default_factory in _VIRTUAL_FOR
            }
            if fields:
                patches[obj] = fields
    return patches


# Sub-tick ordering granularity. The autojumping clock jumps straight onto the next scheduled
# time, so two timers due at the same virtual instant fire in asyncio's heap order (which reflects
# heap structure, not scheduling order) -- and that can invert the order real time would produce
# for causally-ordered-but-simultaneous events. We give each timer a strictly increasing
# sub-microsecond nudge so same-instant timers fire in *scheduling* (FIFO) order, matching the
# causal order real wall-clock produces. The nudge needs a finer clock resolution than the default
# 1e-6 to be representable; accumulation across a test stays far below any timing tolerance.
_TIE_BREAK = 1e-9
_FINE_RESOLUTION = 1e-12


class _VirtualClock:
    """The loop's virtual clock.

    `time()` only moves when `advance()` is called (by the selector), and is what asyncio reads for
    scheduling -- it must stay raw so the sub-nanosecond timer tie-break (see _VirtualTimeLoop) is
    not swamped. `read()` is the wall-clock view: the same virtual time, but nudged to stay
    strictly increasing within a tick so synchronously-created items get distinct, ordered
    `created_at`. The high-water mark lives here (per loop), so it resets naturally with each test's
    fresh loop -- no global state.
    """

    __slots__ = ("_time", "_read")

    def __init__(self) -> None:
        self._time = 0.0
        self._read = 0.0

    def time(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        if seconds > 0:
            self._time += seconds

    def read(self) -> float:
        self._read = self._time if self._time > self._read else self._read + _TICK_EPSILON
        return self._read


class _AutojumpSelector(selectors.BaseSelector):
    """A selector that never really blocks: instead of waiting `timeout` seconds for the next
    timer, it jumps the virtual clock there and returns immediately.

    It delegates fd registration to a real selector and still polls it non-blockingly, so the
    loop's own self-pipe (and thus `call_soon_threadsafe` wakeups from executor threads) keep
    working. A `None` timeout means the loop has nothing scheduled and is genuinely waiting on
    such a wakeup, so we block on the real selector -- matching stock asyncio.
    """

    def __init__(self, clock: _VirtualClock) -> None:
        self._clock = clock
        self._real = selectors.DefaultSelector()

    def register(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._real.register(fileobj, events, data)

    def unregister(self, fileobj: Any) -> selectors.SelectorKey:
        return self._real.unregister(fileobj)

    def modify(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._real.modify(fileobj, events, data)

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        # Hot path: only the loop's self-pipe is registered (no real I/O). A real poll could only
        # ever report that self-pipe, and a `call_soon_threadsafe` wakeup's callback is already
        # queued in the loop's `_ready` (so the next iteration runs it regardless) -- skipping the
        # poll therefore can't drop work. It just avoids an epoll syscall on every loop iteration,
        # which is what keeps virtual time as fast as a fully virtual selector.
        if len(self._real.get_map()) <= 1:
            if timeout is None:
                return self._real.select(None)  # genuinely idle: block for an off-loop wakeup
            if timeout > 0:
                self._clock.advance(timeout)
            return []
        # Real I/O is registered: poll without blocking and jump only if nothing is ready.
        ready = self._real.select(0)
        if ready or timeout == 0:
            return ready
        if timeout is None:
            return self._real.select(None)
        self._clock.advance(timeout)
        return []

    def get_map(self) -> Any:
        return self._real.get_map()

    def close(self) -> None:
        self._real.close()


class _VirtualTimeLoop(asyncio.SelectorEventLoop):
    """A `SelectorEventLoop` whose clock only advances when the loop would otherwise sleep.

    The selector jumps the virtual clock straight to the next scheduled callback (see
    :class:`_AutojumpSelector`), so timing-coupled tests run in ~0 wall time and deterministically.
    Same-instant timers are nudged into scheduling (FIFO) order so they fire in the causal order
    real wall time would produce.
    """

    def __init__(self) -> None:
        self._virtual_clock = _VirtualClock()
        super().__init__(selector=_AutojumpSelector(self._virtual_clock))
        # Resolve finely enough that the sub-nanosecond tie-break is representable and only the
        # exact-same-instant timer fires per jump.
        self._clock_resolution = _FINE_RESOLUTION
        self._tie_seq = 0

    def time(self) -> float:
        return self._virtual_clock.time()

    def monotonic_read(self) -> float:
        """Strictly-increasing wall-clock read off this loop's clock (see _VirtualClock.read)."""
        return self._virtual_clock.read()

    def call_at(self, when: float, callback: Any, *args: Any, **kwargs: Any) -> Any:
        self._tie_seq += 1
        return super().call_at(when + self._tie_seq * _TIE_BREAK, callback, *args, **kwargs)


class _VirtualTimePolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        return _VirtualTimeLoop()


@pytest.fixture
def event_loop_policy(request: pytest.FixtureRequest) -> asyncio.AbstractEventLoopPolicy:
    """Loop policy pytest-asyncio builds the test loop from.

    Virtual-time tests get a :class:`_VirtualTimeLoop` (autojumping virtual clock); everything else
    gets the default policy, i.e. exactly what pytest-asyncio would use without this override, so
    non-marked tests are unaffected. `real_time` opts a test out even under a module-level mark.
    """
    if is_virtual_time(request.node):
        return _VirtualTimePolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(autouse=True)
def _virtual_wall_clock(request: pytest.FixtureRequest) -> Iterator[None]:
    """Point wall-clock reads at the running loop's virtual clock for a `virtual_time` test.

    No-op unless the test is marked `virtual_time`. Two kinds of reads are covered:

    - inline `time.time()` / `time.perf_counter()` -- redirected by patching the `time`
      module (these re-read the attribute at call time, so a scoped patch reaches them);
    - `default_factory=time.time` captured on event/chat-context models at import -- redirected
      by :func:`_patched_model_clocks`, since a module patch cannot reach a captured reference.

    Both resolve the loop at call time, so reads inside the loop see virtual time while any stray
    read from outside it (no running loop) falls back to a stable constant. Scoped to the test.
    """
    if not is_virtual_time(request.node):
        yield
        return

    with (
        mock.patch.object(time, "time", _now),
        mock.patch.object(time, "perf_counter", _perf),
        _patched_model_clocks(),
    ):
        yield


def add_realtime_option(parser: pytest.Parser) -> None:
    group = parser.getgroup("virtual time")
    group.addoption(
        "--real-time",
        action="store_true",
        default=False,
        help="Disable virtual-time patching entirely; virtual_time tests run on the real clock.",
    )


def register_marker(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "virtual_time: run the test on a deterministic autojumping virtual-time event loop.",
    )
    config.addinivalue_line(
        "markers",
        "real_time: opt a test out of a module-level virtual_time mark (runs on the real loop).",
    )
