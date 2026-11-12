"""Suite-wide fixtures. Categories are folders: `e2e/` needs live services, the rest is hermetic."""

import asyncio
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest

from livekit.agents import _core
from livekit.agents.cli import NOISY_LOGGERS

_E2E = Path(__file__).parent / "e2e"
# highest fd soft limit the room tests survive on macOS
_MAX_OPEN_FILES = 8192


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--e2e", action="store_true", help="also run tests_new/e2e (live services)")


def pytest_configure(config: pytest.Config) -> None:
    if sys.platform == "darwin":
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft > _MAX_OPEN_FILES:
            resource.setrlimit(resource.RLIMIT_NOFILE, (_MAX_OPEN_FILES, hard))


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if not config.getoption("--e2e") and collection_path.is_relative_to(_E2E):
        return True
    return None


@pytest.fixture(autouse=True)
def _quiet_noisy_loggers() -> None:
    for name in NOISY_LOGGERS:
        noisy = logging.getLogger(name)
        if noisy.level == logging.NOTSET:
            noisy.setLevel(logging.WARN)


@pytest.fixture(autouse=True)
def _core_logs() -> Iterator[None]:
    """The core queues its log lines: a test's lines reach `logging` before the next test."""
    _core.flush_logs()
    yield
    _core.flush_logs()


class _CoreCaplog(pytest.LogCaptureFixture):  # type: ignore[misc]  # final in pytest; reads must flush the core first
    """`caplog` whose reads first flush the core's queued log lines."""

    @property
    def records(self) -> list[logging.LogRecord]:
        _core.flush_logs()
        return super().records

    @property
    def text(self) -> str:
        _core.flush_logs()
        return super().text

    @property
    def record_tuples(self) -> list[tuple[str, int, str]]:
        _core.flush_logs()
        return super().record_tuples

    @property
    def messages(self) -> list[str]:
        _core.flush_logs()
        return super().messages

    def get_records(self, when: Literal["setup", "call", "teardown"]) -> list[logging.LogRecord]:
        _core.flush_logs()
        return super().get_records(when)


@pytest.fixture
def caplog(request: pytest.FixtureRequest) -> Iterator[pytest.LogCaptureFixture]:
    result = _CoreCaplog(request.node, _ispytest=True)
    yield result
    result._finalize()


@pytest.fixture(scope="session", autouse=True)
def _local_threads(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Local SQLite threads stay out of the real `~/.livekit`."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LIVEKIT_AGENT_THREADS_DIR", str(tmp_path_factory.mktemp("agent-threads")))
        yield


@pytest.fixture(scope="module", autouse=True)
def _restore_logging() -> Iterator[None]:
    """Revert global logging a module mutated (e.g. via `cli.setup_logging`)."""
    loggers = [logging.getLogger()] + [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if name.startswith("livekit")
    ]
    saved = [(lg, lg.level, lg.handlers[:], lg.propagate) for lg in loggers]
    yield
    for lg, level, handlers, propagate in saved:
        lg.setLevel(level)
        lg.handlers[:] = handlers
        lg.propagate = propagate


def _is_ignorable_task(task: asyncio.Task[object]) -> bool:
    coro = task.get_coro()
    name = getattr(coro, "__qualname__", "") or type(coro).__name__
    if "async_generator_athrow" in name:  # GC finalizing an async generator
        return True
    return any("pytest" in frame.f_code.co_filename for frame in task.get_stack())


@pytest.fixture(autouse=True)
async def _fail_on_leaked_tasks() -> object:
    before = set(asyncio.all_tasks())
    yield
    leaked = [
        t for t in set(asyncio.all_tasks()) - before if not t.done() and not _is_ignorable_task(t)
    ]
    if leaked:
        pytest.fail("Test leaked tasks:\n" + "\n".join(repr(t) for t in leaked))
