import asyncio
import dataclasses
import logging

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, utils
from livekit.agents.cli import log

from .toxic_proxy import Toxiproxy

TEST_CONNECT_OPTIONS = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0)


@pytest.fixture
async def job_process():
    utils.http_context._new_session_ctx()
    yield
    await utils.http_context._close_http_ctx()


@pytest.fixture(autouse=True)
def configure_test():
    log._silence_noisy_loggers()


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
async def fail_on_leaked_tasks():
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
