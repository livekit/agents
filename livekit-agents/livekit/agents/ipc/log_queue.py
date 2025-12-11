from __future__ import annotations

import copy
import logging
import pickle
import queue
import sys
import threading
from typing import Any, Callable, Optional

from .. import utils
from ..utils.aio import duplex_unix


def _safe_to_log(obj: Any) -> Any:
    """Safely convert an object to a pickleable format.

    Handles RpcError and other objects that may not be directly pickleable
    by converting them to a dictionary representation.
    """
    try:
        # Try to identify RpcError-like objects
        if hasattr(obj, "__class__") and "RpcError" in obj.__class__.__name__:
            # Convert RpcError to a safe dict representation
            return {
                "_type": "RpcError",
                "message": str(obj) if hasattr(obj, "__str__") else repr(obj),
                "code": getattr(obj, "code", None),
                "msg": getattr(obj, "msg", None),
            }
        return obj
    except Exception:
        # If anything goes wrong, return a safe string representation
        return str(obj) if obj is not None else None


class LogQueueListener:
    def __init__(
        self,
        duplex: utils.aio.duplex_unix._Duplex,
        prepare_fnc: Callable[[logging.LogRecord], None],
    ):
        self._thread: threading.Thread | None = None
        self._duplex = duplex
        self._prepare_fnc = prepare_fnc

    def start(self) -> None:
        self._thread = threading.Thread(target=self._monitor, name="ipc_log_listener")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return

        self._duplex.close()
        self._thread.join()
        self._thread = None

    def handle(self, record: logging.LogRecord) -> None:
        self._prepare_fnc(record)

        lger = logging.getLogger(record.name)
        if not lger.isEnabledFor(record.levelno):
            return

        lger.callHandlers(record)

    def _monitor(self) -> None:
        while True:
            try:
                data = self._duplex.recv_bytes()
            except utils.aio.duplex_unix.DuplexClosed:
                break

            record = pickle.loads(data)
            self.handle(record)


class LogQueueHandler(logging.Handler):
    _sentinal = None

    def __init__(self, duplex: utils.aio.duplex_unix._Duplex) -> None:
        super().__init__()
        self._duplex = duplex
        self._send_q = queue.SimpleQueue[Optional[bytes]]()
        self._send_thread = threading.Thread(target=self._forward_logs, name="ipc_log_forwarder")
        self._send_thread.start()

    @property
    def thread(self) -> threading.Thread:
        return self._send_thread

    def _forward_logs(self) -> None:
        while True:
            serialized_record = self._send_q.get()
            if serialized_record is None:
                break

            try:
                self._duplex.send_bytes(serialized_record)
            except duplex_unix.DuplexClosed:
                break

        self._duplex.close()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Check if Python is shutting down
            if sys.is_finalizing():
                return

            # from https://github.com/python/cpython/blob/91b7f2e7f6593acefda4fa860250dd87d6f849bf/Lib/logging/handlers.py#L1453
            msg = self.format(record)
            record = copy.copy(record)
            record.message = msg
            record.msg = msg
            record.args = None
            record.exc_info = None
            # pass formatted exc_text since stack trace is not pickleable
            record.exc_text = record.exc_text
            record.stack_info = None

            # https://websockets.readthedocs.io/en/stable/topics/logging.html#logging-to-json
            # webosckets library add "websocket" attribute to log records, which is not pickleable
            if hasattr(record, "websocket"):
                record.websocket = None

            # Safely handle RpcError and other non-pickleable objects
            # RpcError objects might be in extra dict fields added via logger calls
            # We need to sanitize these before pickling
            try:
                # Get all non-standard attributes (those added via extra= in logging calls)
                # Standard LogRecord attributes are safe, we only need to check custom ones
                standard_attrs = {
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "message", "pathname", "process", "processName", "relativeCreated",
                    "thread", "threadName", "exc_info", "exc_text", "stack_info",
                    "getMessage", "websocket"
                }

                # Check custom attributes that might contain RpcError
                for attr_name in dir(record):
                    if attr_name.startswith("_") or attr_name in standard_attrs:
                        continue
                    try:
                        attr_value = getattr(record, attr_name, None)
                        if attr_value is not None:
                            # Recursively sanitize dict values (common for extra= parameters)
                            if isinstance(attr_value, dict):
                                safe_dict = {}
                                for key, value in attr_value.items():
                                    safe_dict[key] = _safe_to_log(value)
                                setattr(record, attr_name, safe_dict)
                            else:
                                safe_value = _safe_to_log(attr_value)
                                if safe_value is not attr_value:
                                    setattr(record, attr_name, safe_value)
                    except (AttributeError, TypeError):
                        # Skip attributes that can't be accessed or modified
                        pass
            except Exception:
                # If sanitization fails, continue anyway - the original exception handler will catch pickle errors
                pass

            self._send_q.put_nowait(pickle.dumps(record))

        except Exception:
            self.handleError(record)

    def close(self) -> None:
        super().close()
        self._send_q.put_nowait(self._sentinal)
