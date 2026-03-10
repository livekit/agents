from __future__ import annotations

import logging
import os
import socket
import threading

import pytest

from livekit.agents.cli.log import (
    _has_user_configured_handlers,
    setup_logging,
)
from livekit.agents.ipc.log_queue import LogQueueHandler, LogQueueListener
from livekit.agents.log import _LOG_JSON_ENV, configure_logging, logger
from livekit.agents.utils.aio.duplex_unix import _Duplex


@pytest.fixture(autouse=True)
def _clean_loggers():
    """Remove all handlers added during a test so state doesn't leak."""
    root = logging.getLogger()
    lk = logging.getLogger("livekit.agents")

    root_before = list(root.handlers)
    lk_before = list(lk.handlers)
    lk_level_before = lk.level
    env_before = os.environ.get(_LOG_JSON_ENV)

    yield

    root.handlers = root_before
    lk.handlers = lk_before
    lk.level = lk_level_before
    if env_before is None:
        os.environ.pop(_LOG_JSON_ENV, None)
    else:
        os.environ[_LOG_JSON_ENV] = env_before


class TestNullHandler:
    def test_logger_has_null_handler(self):
        assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)

    def test_no_output_without_configuration(self, capsys):
        """Logging to the livekit.agents logger should not produce output
        when no additional handlers have been configured."""
        logger.info("should be silent")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestHasUserConfiguredHandlers:
    def test_null_handler_only(self):
        # The autouse fixture restores original handlers which includes the
        # NullHandler added at import time.  No StreamHandler should be present.
        root = logging.getLogger()
        lk = logging.getLogger("livekit.agents")

        # Temporarily strip everything except NullHandlers
        saved_root = root.handlers[:]
        saved_lk = lk.handlers[:]
        root.handlers = [h for h in root.handlers if isinstance(h, logging.NullHandler)]
        lk.handlers = [h for h in lk.handlers if isinstance(h, logging.NullHandler)]

        try:
            assert _has_user_configured_handlers() is False
        finally:
            root.handlers = saved_root
            lk.handlers = saved_lk

    def test_stream_handler_on_root(self):
        root = logging.getLogger()
        h = logging.StreamHandler()
        root.addHandler(h)
        try:
            assert _has_user_configured_handlers() is True
        finally:
            root.removeHandler(h)

    def test_handler_on_livekit_agents(self):
        lk = logging.getLogger("livekit.agents")
        h = logging.StreamHandler()
        lk.addHandler(h)
        try:
            assert _has_user_configured_handlers() is True
        finally:
            lk.removeHandler(h)


class TestConfigureLogging:
    def test_default_handler(self):
        root = logging.getLogger()
        before = len(root.handlers)

        configure_logging(level=logging.WARNING)

        assert len(root.handlers) == before + 1
        new_handler = root.handlers[-1]
        assert isinstance(new_handler, logging.StreamHandler)

    def test_custom_handler(self):
        root = logging.getLogger()
        custom = logging.StreamHandler()

        configure_logging(handler=custom)

        assert custom in root.handlers

    def test_json_formatter(self):
        from livekit.agents.cli.log import JsonFormatter

        root = logging.getLogger()

        configure_logging(json=True)

        new_handler = root.handlers[-1]
        assert isinstance(new_handler.formatter, JsonFormatter)

    def test_custom_formatter(self):
        fmt = logging.Formatter("%(message)s")

        configure_logging(formatter=fmt)

        root = logging.getLogger()
        new_handler = root.handlers[-1]
        assert new_handler.formatter is fmt

    def test_json_overrides_custom_formatter(self):
        from livekit.agents.cli.log import JsonFormatter

        fmt = logging.Formatter("%(message)s")

        configure_logging(formatter=fmt, json=True)

        root = logging.getLogger()
        new_handler = root.handlers[-1]
        assert isinstance(new_handler.formatter, JsonFormatter)

    def test_level_setting(self):
        configure_logging(level=logging.ERROR)

        root = logging.getLogger()
        assert root.level == logging.ERROR


class TestSetupLoggingGuard:
    def test_skips_handler_when_user_configured(self):
        root = logging.getLogger()
        user_handler = logging.StreamHandler()
        root.addHandler(user_handler)

        before_count = len(root.handlers)
        setup_logging("INFO", devmode=False, console=False)

        # Should NOT have added another handler
        assert len(root.handlers) == before_count

    def test_adds_handler_when_not_configured(self):
        root = logging.getLogger()
        lk = logging.getLogger("livekit.agents")

        # Strip all non-NullHandler handlers
        root.handlers = [h for h in root.handlers if isinstance(h, logging.NullHandler)]
        lk.handlers = [h for h in lk.handlers if isinstance(h, logging.NullHandler)]

        before_count = len(root.handlers)
        setup_logging("DEBUG", devmode=True, console=False)

        assert len(root.handlers) == before_count + 1


class TestIPCLogForwarding:
    def test_log_record_forwarded_through_socket(self):
        s1, s2 = socket.socketpair()

        child_duplex = _Duplex.open(s1)
        parent_duplex = _Duplex.open(s2)

        received: list[logging.LogRecord] = []
        ready = threading.Event()

        def prepare(record: logging.LogRecord) -> None:
            pass

        listener = LogQueueListener(parent_duplex, prepare_fnc=prepare)

        # Patch handle to capture records instead of dispatching
        def capture_handle(record: logging.LogRecord) -> None:
            received.append(record)
            ready.set()

        listener.handle = capture_handle  # type: ignore[assignment]
        listener.start()

        handler = LogQueueHandler(child_duplex)

        test_record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello from child",
            args=None,
            exc_info=None,
        )
        handler.emit(test_record)

        assert ready.wait(timeout=5), "Timed out waiting for log record"
        assert len(received) == 1
        assert received[0].getMessage() == "hello from child"
        assert received[0].name == "test.logger"

        handler.close()
        handler.thread.join(timeout=5)
        listener.stop()
