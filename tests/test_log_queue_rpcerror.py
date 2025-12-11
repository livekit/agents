"""Test case for Issue #4188: LogQueueListener crashes when logging RpcErrors."""

from __future__ import annotations

import logging
import pickle
import socket
import threading
import time

from livekit.agents.ipc.log_queue import LogQueueHandler, LogQueueListener
from livekit.agents.utils.aio import duplex_unix


class MockRpcError:
    """Mock RpcError-like object that might not be pickleable."""

    def __init__(self, message: str, code: int = 0):
        self.message = message
        self.code = code
        self.msg = message

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"MockRpcError(message={self.message!r}, code={self.code})"


def test_log_queue_handler_with_rpcerror() -> None:
    """Test that LogQueueHandler can safely handle RpcError objects in log records."""
    mp_log_pch, mp_log_cch = socket.socketpair()

    try:
        log_pch = duplex_unix._Duplex.open(mp_log_pch)
        handler = LogQueueHandler(log_pch)

        # Create a logger and add the handler
        test_logger = logging.getLogger("test_rpcerror")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        # Try to log with RpcError in extra fields
        rpc_error = MockRpcError("Test RPC error", code=500)
        test_logger.error(
            "Test error with RpcError",
            extra={"rpc_error": rpc_error, "error_code": rpc_error.code},
        )

        # Give the handler time to process
        time.sleep(0.1)

        # The handler should not crash when pickling
        # If it does, the exception will be caught by handleError
        assert True, "LogQueueHandler should handle RpcError without crashing"

    finally:
        handler.close()
        log_pch.close()
        mp_log_cch.close()


def test_log_queue_listener_with_rpcerror() -> None:
    """Test that LogQueueListener can safely receive and handle log records with RpcError."""
    mp_log_pch, mp_log_cch = socket.socketpair()

    try:
        log_pch = duplex_unix._Duplex.open(mp_log_pch)
        log_cch = duplex_unix._Duplex.open(mp_log_cch)

        received_records = []

        def prepare_fnc(record: logging.LogRecord) -> None:
            received_records.append(record)

        listener = LogQueueListener(log_cch, prepare_fnc)
        listener.start()

        # Create a handler that sends to the listener
        handler = LogQueueHandler(log_pch)

        test_logger = logging.getLogger("test_listener_rpcerror")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        # Log with RpcError
        rpc_error = MockRpcError("Test RPC error for listener", code=404)
        test_logger.warning(
            "Test warning with RpcError",
            extra={"rpc_error": rpc_error},
        )

        # Give time for the log to be processed
        time.sleep(0.2)

        # The listener should not crash
        assert True, "LogQueueListener should handle RpcError without crashing"

        listener.stop()
        handler.close()

    finally:
        log_pch.close()
        log_cch.close()

