import dataclasses
import types
import asyncio
import gc
import logging
import inspect
import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, utils
from livekit.agents.cli import log
from .toxic_proxy import Toxiproxy

TEST_CONNECT_OPTIONS = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0)


@pytest.fixture
def job_process(event_loop):
    utils.http_context._new_session_ctx()
    yield
    event_loop.run_until_complete(utils.http_context._close_http_ctx())


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


def safe_is_async_generator(obj):
    # For whatever reason, OpenAI complains about this.

    # .venv/lib/python3.9/site-packages/openai/_extras/pandas_proxy.py:20: in __load__
    #   import pandas
    #   ModuleNotFoundError: No module named 'pandas'
    try:
        return isinstance(obj, types.AsyncGeneratorType)
    except Exception:
        return False


def safe_is_async_generator(obj):
    try:
        return isinstance(obj, types.AsyncGeneratorType)
    except Exception:
        return False


def live_async_generators_ids() -> set:
    return {
        id(obj)
        for obj in gc.get_objects()
        if safe_is_async_generator(obj) and getattr(obj, "ag_frame", None) is not None
    }


def is_pytest_asyncio_task(task) -> bool:
    try:
        coro = task.get_coro()
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


def format_async_generator_by_id(gen_id: int) -> str:
    for obj in gc.get_objects():
        if id(obj) == gen_id and safe_is_async_generator(obj):
            try:
                frame = getattr(obj, "ag_frame", None)
                if frame:
                    filename = frame.f_code.co_filename  # type: ignore[attr-defined]
                    lineno = frame.f_lineno  # type: ignore[attr-defined]
                    func_name = frame.f_code.co_name  # type: ignore[attr-defined]
                    stack_summary = "\n".join(
                        f'    File "{frm.f_code.co_filename}", line {frm.f_lineno}, in {frm.f_code.co_name}'
                        for frm in inspect.getouterframes(frame)
                    )
                    return (
                        f"AsyncGenerator id: {gen_id}\n"
                        f"  Created/paused in: {func_name}\n"
                        f"  Location: {filename}:{lineno}\n"
                        f"  Frame stack:\n{stack_summary}"
                    )
                else:
                    return f"AsyncGenerator id: {gen_id} (closed)"
            except Exception as e:
                return f"AsyncGenerator id: {gen_id} (failed to introspect: {e})"
    return f"AsyncGenerator id: {gen_id} (object not found)"


@pytest.fixture(autouse=True)
async def fail_on_leaked_tasks():
    tasks_before = set(asyncio.all_tasks())
    async_gens_before = live_async_generators_ids()

    yield

    # gc.collect()

    tasks_after = set(asyncio.all_tasks())
    async_gens_after = live_async_generators_ids()

    leaked_tasks = [
        task
        for task in tasks_after - tasks_before
        if not task.done() and not is_pytest_asyncio_task(task)
    ]
    leaked_async_gens = async_gens_after - async_gens_before

    error_messages = []
    if leaked_tasks:
        tasks_msg = "\n\n".join(format_task(task) for task in leaked_tasks)
        error_messages.append("Leaked tasks detected:\n" + tasks_msg)
    if leaked_async_gens:
        gens_msg = "\n\n".join(format_async_generator_by_id(gen_id) for gen_id in leaked_async_gens)
        error_messages.append("Leaked async generators detected:\n" + gens_msg)

    if error_messages:
        final_msg = "Test leaked resources:\n\n" + "\n\n".join(error_messages)
        pytest.fail(final_msg)
