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

"""Krisp SDK process-singleton manager with reference counting.

The Krisp ``globalInit`` / ``globalDestroy`` calls are process-global, so this
manager keeps a single backend alive across multiple frame processors. The
active backend is selected by the first :meth:`KrispSDKManager.acquire` call;
subsequent acquires must use a provider with the same ``name``.
"""

from __future__ import annotations

from threading import Lock

from .auth import KrispAuthProvider, KrispBackend
from .log import logger


class KrispSDKManager:
    """Process-singleton with reference counting for the Krisp SDK.

    Acquire a reference when constructing a frame processor; release it when
    destroying the processor. The first acquire selects the auth provider
    and initializes the backend; the last release tears it down.
    """

    _provider: KrispAuthProvider | None = None
    _backend: KrispBackend | None = None
    _reference_count: int = 0
    _lock = Lock()

    @classmethod
    def acquire(cls, provider: KrispAuthProvider) -> KrispBackend:
        """Acquire a reference, initializing on first call.

        Raises:
            RuntimeError: if a previously-acquired provider has a different
                ``name`` (mixing auth modes in one process is not supported).
        """
        with cls._lock:
            if cls._reference_count == 0:
                backend = provider.init_sdk()
                cls._backend = backend
                cls._provider = provider
            else:
                assert cls._provider is not None and cls._backend is not None
                if cls._provider.name != provider.name:
                    raise RuntimeError(
                        f"Krisp SDK already initialized with provider="
                        f"{cls._provider.name!r}; cannot mix with "
                        f"{provider.name!r} in the same process."
                    )
                if type(cls._provider) is not type(provider):
                    logger.warning(
                        "Krisp SDK reacquired with a different provider instance "
                        "of the same kind; the original credentials are kept "
                        "(globalInit is process-global)."
                    )

            cls._reference_count += 1
            logger.debug("Krisp SDK reference count: %d", cls._reference_count)
            assert cls._backend is not None
            return cls._backend

    @classmethod
    def release(cls) -> None:
        """Release a reference, destroying the SDK on the last release."""
        with cls._lock:
            if cls._reference_count == 0:
                return
            cls._reference_count -= 1
            logger.debug("Krisp SDK reference count: %d", cls._reference_count)

            if cls._reference_count == 0:
                assert cls._provider is not None and cls._backend is not None
                try:
                    cls._provider.destroy_sdk(cls._backend)
                    logger.debug("Krisp Audio SDK destroyed (all references released)")
                except Exception as e:
                    logger.error(f"Error during Krisp SDK cleanup: {e}")
                finally:
                    cls._backend = None
                    cls._provider = None

    @classmethod
    def is_initialized(cls) -> bool:
        with cls._lock:
            return cls._backend is not None

    @classmethod
    def get_reference_count(cls) -> int:
        with cls._lock:
            return cls._reference_count
