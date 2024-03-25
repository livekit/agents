# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal
import asyncio
from contextlib import ExitStack
import ctypes
import importlib.resources
import logging
import os
import platform
import atexit
import threading
from typing import Generic, List, Optional, TypeVar

from ._proto import ffi_pb2 as proto_ffi
from ._utils import Queue, classproperty

logger = logging.getLogger("livekit")

_resource_files = ExitStack()
atexit.register(_resource_files.close)


def get_ffi_lib():
    # allow to override the lib path using an env var
    libpath = os.environ.get("LIVEKIT_LIB_PATH", "").strip()
    if libpath:
        return ctypes.CDLL(libpath)

    if platform.system() == "Linux":
        libname = "liblivekit_ffi.so"
    elif platform.system() == "Darwin":
        libname = "liblivekit_ffi.dylib"
    elif platform.system() == "Windows":
        libname = "livekit_ffi.dll"
    else:
        raise Exception(
            f"no ffi library found for platform {platform.system()}. \
                Set LIVEKIT_LIB_PATH to specify a the lib path"
        )

    res = importlib.resources.files("livekit.rtc.resources") / libname
    ctx = importlib.resources.as_file(res)
    path = _resource_files.enter_context(ctx)
    return ctypes.CDLL(str(path))


ffi_lib = get_ffi_lib()
ffi_cb_fnc = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t)

# C function types
ffi_lib.livekit_ffi_initialize.argtypes = [ffi_cb_fnc, ctypes.c_bool]

ffi_lib.livekit_ffi_request.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
    ctypes.POINTER(ctypes.c_size_t),
]
ffi_lib.livekit_ffi_request.restype = ctypes.c_uint64

ffi_lib.livekit_ffi_drop_handle.argtypes = [ctypes.c_uint64]
ffi_lib.livekit_ffi_drop_handle.restype = ctypes.c_bool

INVALID_HANDLE = 0


class FfiHandle:
    def __init__(self, handle: int) -> None:
        self.handle = handle
        self._disposed = False

    def __del__(self):
        self.dispose()

    @property
    def disposed(self) -> bool:
        return self._disposed

    def dispose(self) -> None:
        if self.handle != INVALID_HANDLE and not self._disposed:
            self._disposed = True
            assert ffi_lib.livekit_ffi_drop_handle(ctypes.c_uint64(self.handle))


T = TypeVar("T")


class FfiQueue(Generic[T]):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: List[tuple[Queue[T], asyncio.AbstractEventLoop]] = []

    def put(self, item: T) -> None:
        with self._lock:
            for queue, loop in self._subscribers:
                loop.call_soon_threadsafe(queue.put_nowait, item)

    def subscribe(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> Queue[T]:
        with self._lock:
            queue = Queue[T]()
            loop = loop or asyncio.get_event_loop()
            self._subscribers.append((queue, loop))
            return queue

    def unsubscribe(self, queue: Queue[T]) -> None:
        with self._lock:
            # looping here is ok, since we don't expect a lot of subscribers
            for i, (q, _) in enumerate(self._subscribers):
                if q == queue:
                    self._subscribers.pop(i)
                    break


@ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t)
def ffi_event_callback(
    data_ptr: ctypes.POINTER(ctypes.c_uint8),  # type: ignore
    data_len: ctypes.c_size_t,
) -> None:
    event_data = bytes(data_ptr[: int(data_len)])
    event = proto_ffi.FfiEvent()
    event.ParseFromString(event_data)

    which = event.WhichOneof("message")
    if which == "logs":
        for record in event.logs.records:
            level = to_python_level(record.level)
            rtc_debug = os.environ.get("LIVEKIT_WEBRTC_DEBUG", "").strip()
            if (
                record.target == "libwebrtc"
                and level == logging.DEBUG
                and rtc_debug.lower() not in ("true", "1")
            ):
                continue

            if level is not None:
                logger.log(
                    level,
                    "%s:%s:%s - %s",
                    record.target,
                    record.line,
                    record.module_path,
                    record.message,
                )

        return  # no need to queue the logs
    elif which == "panic":
        logger.critical("Panic: %s", event.panic.message)
        # We are in a unrecoverable state, terminate the process
        os.kill(os.getpid(), signal.SIGTERM)
        return

    FfiClient.instance.queue.put(event)


def to_python_level(level: proto_ffi.LogLevel.ValueType) -> Optional[int]:
    if level == proto_ffi.LogLevel.LOG_ERROR:
        return logging.ERROR
    elif level == proto_ffi.LogLevel.LOG_WARN:
        return logging.WARN
    elif level == proto_ffi.LogLevel.LOG_INFO:
        return logging.INFO
    elif level == proto_ffi.LogLevel.LOG_DEBUG:
        return logging.DEBUG
    elif level == proto_ffi.LogLevel.LOG_TRACE:
        # Don't show TRACE logs inside DEBUG, it is too verbos
        # Python's logging doesn't have a TRACE level
        # return logging.DEBUG
        pass

    return None


class FfiClient:
    _instance: Optional["FfiClient"] = None

    @classproperty
    def instance(self):
        if self._instance is None:
            self._instance = FfiClient()
        return self._instance

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._queue = FfiQueue[proto_ffi.FfiEvent]()

        ffi_lib.livekit_ffi_initialize(ffi_event_callback, True)

    @property
    def queue(self) -> FfiQueue[proto_ffi.FfiEvent]:
        return self._queue

    def request(self, req: proto_ffi.FfiRequest) -> proto_ffi.FfiResponse:
        proto_data = req.SerializeToString()
        proto_len = len(proto_data)
        data = (ctypes.c_ubyte * proto_len)(*proto_data)

        resp_ptr = ctypes.POINTER(ctypes.c_ubyte)()
        resp_len = ctypes.c_size_t()
        handle = ffi_lib.livekit_ffi_request(
            data, proto_len, ctypes.byref(resp_ptr), ctypes.byref(resp_len)
        )
        assert handle != INVALID_HANDLE

        resp_data = bytes(resp_ptr[: resp_len.value])
        resp = proto_ffi.FfiResponse()
        resp.ParseFromString(resp_data)

        FfiHandle(handle)
        return resp
