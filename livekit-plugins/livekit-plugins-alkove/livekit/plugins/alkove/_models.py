# Copyright 2026 Eddy Bordi
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

import alkove

_lock = threading.Lock()
_models: alkove.Models | None = None


def models(*, download: bool = True) -> alkove.Models:
    """Focus and Cue, loaded once for the process and shared by every session.

    They are read from the cache `download-files` filled and checked by their sha256.

    Args:
        download: Download the models that are not in the cache. Pass `False` on the
            event loop.
    """
    global _models
    with _lock:
        if _models is None:
            _models = alkove.Models.load(["focus", "cue"], download=download)
        return _models
