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

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

T = TypeVar("T")

# loaded models are shared by every instance in the process (one copy per job process,
# or one copy per worker when jobs run in a thread executor)
_loaded: dict[Hashable, Any] = {}
_load_locks: dict[Hashable, threading.Lock] = {}
# guards the two dicts only; loads hold their key's own lock so they don't block others
_lock = threading.Lock()


def get_or_load(key: Hashable, load: Callable[[], T]) -> T:
    """Return the object cached under ``key``, loading it once per process."""
    with _lock:
        if key in _loaded:
            return _loaded[key]  # type: ignore[no-any-return]
        load_lock = _load_locks.setdefault(key, threading.Lock())

    with load_lock:
        with _lock:
            if key in _loaded:
                return _loaded[key]  # type: ignore[no-any-return]
        value = load()
        with _lock:
            _loaded[key] = value
        return value


def not_cached_error(model: str) -> RuntimeError:
    return RuntimeError(
        f"model {model!r} is not in the local Hugging Face cache. Download it ahead of "
        "time with `python <your_agent>.py download-files` (set LIVEKIT_HF_MODELS to the "
        "models you use) or `livekit.plugins.hf.download_model()`, or pass "
        "local_files_only=False"
    )
