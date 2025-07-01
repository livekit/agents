from __future__ import annotations

import copy
import logging
import pickle
import queue
import sys
import threading
from typing import Callable, Optional

from .. import utils
from ..utils.aio import duplex_unix


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
            record.exc_text = None
            record.stack_info = None

            # https://websockets.readthedocs.io/en/stable/topics/logging.html#logging-to-json
            # webosckets library add "websocket" attribute to log records, which is not pickleable
            if hasattr(record, "websocket"):
                record.websocket = None

            self._send_q.put_nowait(pickle.dumps(record))

        except Exception:
            self.handleError(record)

    def close(self) -> None:
        super().close()
        self._send_q.put_nowait(self._sentinal)
