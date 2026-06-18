import asyncio
import dataclasses
import logging
import re
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import NamedTuple

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, utils
from livekit.agents.cli import log

from . import concurrency
from .toxic_proxy import Toxiproxy
from .virtual_time import (  # noqa: F401  (re-exported so pytest discovers the fixtures)
    _virtual_wall_clock,
    add_realtime_option as _add_realtime_option,
    event_loop_policy,
    register_marker as _register_virtual_time_marker,
)

TEST_CONNECT_OPTIONS = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0)

# Category names are the pytest markers declared in pyproject.toml — the single
# source of truth. They're read from the file rather than via config.getini()
# because the --<category> options are registered in pytest_addoption, which runs
# before pytest has parsed the ini file.
_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _load_categories() -> tuple[str, ...]:
    block = re.search(r"(?ms)^markers\s*=\s*\[(.*?)^\s*\]", _PYPROJECT.read_text(encoding="utf-8"))
    names = re.findall(r'^\s*"(\w+)', block.group(1), re.MULTILINE) if block else []
    if not names:
        raise RuntimeError(f"no pytest markers found in {_PYPROJECT}")
    return tuple(dict.fromkeys(names))


CATEGORIES = _load_categories()

# matches `pytest.mark.<category>` anywhere in a module's source (module-level
# `pytestmark = ...` or per-test `@pytest.mark.<category>` decorators).
_CATEGORY_RE = re.compile(r"pytest\.mark\.(" + "|".join(CATEGORIES) + r")\b")
# a module "has tests" if it declares any (non-commented) test function or Test class.
_HAS_TESTS_RE = re.compile(r"^\s*(?:async\s+)?def test|^\s*class Test", re.MULTILINE)

_CATEGORY_HINT = (
    "Every test module must declare its category with a module-level marker, one of:\n\n"
    + "\n".join(f"    pytestmark = pytest.mark.{c}" for c in CATEGORIES)
    + "\n\nRun pytest with the --allow-uncategorized option to temporarily disable this rule."
)


def pytest_addoption(parser: pytest.Parser) -> None:
    _add_realtime_option(parser)
    group = parser.getgroup("categories", "test category selection")
    for category in CATEGORIES:
        group.addoption(
            f"--{category}",
            nargs="?",
            const=True,
            default=None,
            metavar="PROVIDER",
            help=f"select only `{category}` tests; optionally narrow to a single "
            f"provider/target (e.g. --{category} <name>). Repeatable categories are unioned.",
        )
    group.addoption(
        "--list-categories",
        action="store_true",
        default=False,
        help="list every test module grouped by category, then exit (no tests run).",
    )
    group.addoption(
        "--allow-uncategorized",
        action="store_true",
        default=False,
        help="allow test modules without a category marker, temporarily disabling "
        "the categorization rule (CI keeps it on by default).",
    )


class _ModuleFacts(NamedTuple):
    categories: frozenset[str]
    has_tests: bool


@cache
def _module_facts(path: Path) -> _ModuleFacts:
    """
    Retrieve information about the module without importing it.
    Cache to avoid redundant filesystem reads.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return _ModuleFacts(frozenset(), False)
    return _ModuleFacts(
        frozenset(_CATEGORY_RE.findall(source)),
        _HAS_TESTS_RE.search(source) is not None,
    )


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _iter_test_files(config: pytest.Config) -> Iterator[Path]:
    rootdir = Path(str(config.rootdir))
    for testpath in config.getini("testpaths") or ["tests"]:
        base = rootdir / testpath
        if base.is_dir():
            yield from sorted(base.rglob("test_*.py"))
        elif base.is_file() and base.name.startswith("test_"):
            yield base


def _uncategorized_modules(config: pytest.Config) -> list[Path]:
    """Test files that contain tests but declare no category marker."""
    offenders: list[Path] = []
    for path in _iter_test_files(config):
        facts = _module_facts(path)
        if not facts.has_tests:
            continue  # empty / fully-commented module: nothing to categorize
        if not facts.categories:
            offenders.append(path)
    return offenders


def pytest_configure(config: pytest.Config) -> None:
    """Handle `--list-categories` before any collection/import happens."""
    _module_facts.cache_clear()
    _register_virtual_time_marker(config)

    if not config.getoption("--list-categories"):
        return

    rootdir = Path(str(config.rootdir))
    by_category: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    uncategorized: list[str] = []
    for path in _iter_test_files(config):
        facts = _module_facts(path)
        if not facts.has_tests:
            continue
        rel = str(path.relative_to(rootdir))
        if not facts.categories:
            uncategorized.append(rel)
        for cat in facts.categories:
            by_category[cat].append(rel)

    lines = ["", "Test categories (select with --<category>):", ""]
    for category in CATEGORIES:
        modules = by_category[category]
        lines.append(f"  {category:<8} {_plural(len(modules), 'module')}")
        lines.extend(f"             - {rel}" for rel in modules)
    if uncategorized:
        blocked = _plural(len(uncategorized), "module")
        lines.append(f"\n  UNCATEGORIZED {blocked} — run is blocked:")
        lines.extend(f"             - {rel}" for rel in uncategorized)
    pytest.exit("\n".join(lines), returncode=0)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Enforce category markers, then apply provider-argument filtering."""
    if not config.getoption("--allow-uncategorized"):
        offenders = _uncategorized_modules(config)
        if offenders:
            rootdir = Path(str(config.rootdir))
            listed = "\n".join(f"  - {p.relative_to(rootdir)}" for p in offenders)
            raise pytest.UsageError(
                f"Found {_plural(len(offenders), 'test module')} without a "
                f"category marker:\n\n{listed}\n\n{_CATEGORY_HINT}"
            )

    _filter_by_provider(config, items)


def _selected_categories(config: pytest.Config) -> dict[str, object]:
    # category -> True (whole category) or a provider string (single target)
    selected: dict[str, object] = {}
    for category in CATEGORIES:
        value = config.getoption(f"--{category}")
        if value is not None:
            selected[category] = value
    return selected


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip non-selected modules *before* importing them.

    Marker selection alone can't do this: pytest must import a module to read its
    markers, and some modules fail to import without optional/cloud deps. So we
    detect the category marker by a plain substring scan of the source instead.
    """
    selected = _selected_categories(config)
    if not selected:
        return None
    if collection_path.is_dir() or collection_path.suffix != ".py":
        return None
    if not collection_path.name.startswith("test_"):
        return None
    if _module_facts(collection_path).categories.intersection(selected):
        return None

    return True


def _filter_by_provider(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply provider-argument filtering once modules are imported.

    `--plugin anthropic` keeps only items whose `plugin` marker carries "anthropic"
    in its args; a bare `--plugin` keeps the whole category.
    """
    selected = _selected_categories(config)
    if not selected:
        return

    kept, deselected = [], []
    for item in items:
        keep = False
        for category, target in selected.items():
            for marker in item.iter_markers(category):
                if target is True or target in marker.args:
                    keep = True
                    break
            if keep:
                break
        (kept if keep else deselected).append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


@pytest.fixture
async def job_process():
    utils.http_context._new_session_ctx()
    yield
    await utils.http_context._close_http_ctx()


@pytest.fixture(autouse=True)
def configure_test():
    log._silence_noisy_loggers()


@pytest.fixture(scope="session", autouse=True)
def _logging_baseline():
    """Pristine logging state, snapshotted once before any test runs."""
    loggers = [logging.getLogger()]  # root
    loggers += [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if name.startswith("livekit")
    ]
    return [(logger, logger.level, logger.handlers[:], logger.propagate) for logger in loggers]


@pytest.fixture(scope="module", autouse=True)
def _restore_logging(_logging_baseline):
    """Revert global logging a test mutated (e.g. via cli.log.setup_logging)."""
    yield
    for logger, level, handlers, propagate in _logging_baseline:
        logger.setLevel(level)
        logger.handlers[:] = handlers
        logger.propagate = propagate


@pytest.fixture
def toxiproxy():
    toxiproxy = Toxiproxy()
    yield toxiproxy
    if toxiproxy.running():
        toxiproxy.destroy_all()


@pytest.fixture()
def logger():
    logger = logging.getLogger("livekit.tests")
    logger.setLevel(logging.DEBUG)
    return logger


def _is_ignorable_task(task) -> bool:
    try:
        coro = task.get_coro()
        # async_generator_athrow tasks are created by Python's GC when finalizing
        # async generators (e.g. httpx/httpcore streaming generators)
        coro_name = getattr(coro, "__qualname__", "") or type(coro).__name__
        if "async_generator_athrow" in coro_name:
            return True
        mod = getattr(coro, "__module__", "")
        if "pytest" in mod or "pytest_asyncio" in mod:
            return True
        for frame in task.get_stack():
            if "pytest" in frame.f_code.co_filename or "pytest_asyncio" in frame.f_code.co_filename:
                return True
    except Exception:
        pass
    return False


def format_task(task) -> str:
    try:
        name = task.get_name() if hasattr(task, "get_name") else "<unknown name>"
        coro = task.get_coro()
        coro_name = getattr(coro, "__qualname__", None) or type(coro).__name__
        frame = getattr(coro, "cr_frame", None)
        if frame:
            location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        else:
            location = "no frame available"
        return (
            f"Task Name   : {name}\n"
            f"Coroutine   : {coro_name}\n"
            f"Location    : {location}\n"
            f"State       : {'pending' if not task.done() else 'done'}"
        )
    except Exception:
        return repr(task)


@pytest.fixture(autouse=True)
async def fail_on_leaked_tasks(request):
    if concurrency.is_concurrent_member(request.node):
        # Concurrent group members share one event loop, so we can't diff asyncio.all_tasks()
        # (it would flag other still-running tests). Instead ask which still-pending tasks were
        # created by *this* test -- tagged via contextvars, see tests/concurrency.py.
        yield
        leaked_tasks = [
            task
            for task in concurrency.owned_pending_tasks(request.node.nodeid)
            if not _is_ignorable_task(task)
        ]
        if leaked_tasks:
            tasks_msg = "\n\n".join(format_task(task) for task in leaked_tasks)
            pytest.fail("Test leaked tasks:\n\n" + tasks_msg)
        return

    tasks_before = set(asyncio.all_tasks())

    yield

    tasks_after = set(asyncio.all_tasks())

    leaked_tasks = [
        task
        for task in tasks_after - tasks_before
        if not task.done() and not _is_ignorable_task(task)
    ]

    if leaked_tasks:
        tasks_msg = "\n\n".join(format_task(task) for task in leaked_tasks)
        pytest.fail("Test leaked tasks:\n\n" + tasks_msg)
