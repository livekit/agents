# Copyright 2025 LiveKit, Inc.
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

import blockguard as _cext

from .version import __version__


def install(threshold_ms: float = 5000, poll_ms: float = 500) -> None:
    """Start the watchdog. Must be called from the event-loop thread."""
    _cext.install(threshold_ms=threshold_ms, poll_ms=poll_ms)


def uninstall() -> None:
    """Stop the watchdog thread."""
    _cext.uninstall()


__all__ = [
    "install",
    "uninstall",
    "__version__",
]
