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

"""Authentication providers for the Krisp plugin.

These types are plain configuration holders — :class:`KrispVivaFilterFrameProcessor`
dispatches on ``isinstance(...)`` to pick the matching internal FrameProcessor
implementation.

- :class:`LiveKitCloudAuthProvider` (default): selects the closed-source
  ``krisp_audio_livekit_internal`` wheel. The wheel bundles the model and
  authenticates against LiveKit Cloud using the room's JWT, which the agent
  framework hands to the FrameProcessor via the standard
  ``_on_credentials_updated`` callback.
- :class:`KrispLicenseAuthProvider`: selects the public ``krisp_audio`` wheel
  with a Krisp license key + ``.kef`` model file.

Recommended call sites use the snake_case aliases via the ``auth`` namespace::

    from livekit.plugins import krisp

    krisp.auth.livekit_cloud()
    krisp.auth.krisp_license(license_key="...", model_path="...")
"""

from __future__ import annotations

import os


class LiveKitCloudAuthProvider:
    """Marker for the LiveKit Cloud-bundled Krisp backend.

    Auth + metering happen inside ``krisp_audio_livekit_internal``, which
    receives the room JWT via the standard FrameProcessor
    ``_on_credentials_updated`` callback (forwarded by the facade). No
    constructor arguments — there is nothing to configure on this side.
    """


class KrispLicenseAuthProvider:
    """Krisp-direct auth using a license key + ``.kef`` model file.

    Defaults to reading ``KRISP_VIVA_SDK_LICENSE_KEY`` and
    ``KRISP_VIVA_FILTER_MODEL_PATH`` from the environment.
    """

    def __init__(
        self,
        *,
        license_key: str | None = None,
        model_path: str | None = None,
    ) -> None:
        resolved_license_key = license_key or os.getenv("KRISP_VIVA_SDK_LICENSE_KEY") or ""
        resolved_model_path = model_path or os.getenv("KRISP_VIVA_FILTER_MODEL_PATH")

        if not resolved_model_path:
            raise ValueError(
                "Krisp model path is required. Pass model_path=... or set "
                "KRISP_VIVA_FILTER_MODEL_PATH."
            )
        if not resolved_model_path.endswith(".kef"):
            raise ValueError("Krisp model must have .kef extension")
        if not os.path.isfile(resolved_model_path):
            raise FileNotFoundError(f"Krisp model file not found: {resolved_model_path}")

        self.license_key: str = resolved_license_key
        self.model_path: str = resolved_model_path


# Snake_case aliases — preferred call sites use ``krisp.auth.livekit_cloud()``
# and ``krisp.auth.krisp_license(...)``. Classes are callable, so these are
# just renames, not factory wrappers (cf. ``numpy.array``, ``dataclasses.field``).
livekit_cloud = LiveKitCloudAuthProvider
krisp_license = KrispLicenseAuthProvider

__all__ = [
    "LiveKitCloudAuthProvider",
    "KrispLicenseAuthProvider",
    "livekit_cloud",
    "krisp_license",
]
