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

"""Helpers for constructing LiveKit Egress upload configurations from GCP credentials."""

from __future__ import annotations

import asyncio
import json
import os

__all__ = ["build_gcp_upload", "build_gcp_upload_async"]


def build_gcp_upload(
    bucket: str,
    *,
    credentials_info: dict | None = None,
    credentials_file: str | None = None,
) -> livekit.api.GCPUpload:  # type: ignore[name-defined]  # noqa: F821
    """Build a :class:`livekit.api.GCPUpload` message for use in LiveKit Egress requests.

    The ``GCPUpload.credentials`` field must contain a service account key JSON string.
    This helper resolves that string from (in priority order):

    1. ``credentials_info`` — a service account key dict passed directly.
    2. ``credentials_file`` — a path to a service account key JSON file.
    3. The ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable, if it points to a
       service account key file.

    .. important::

        ``GCPUpload.credentials`` is consumed by the **LiveKit Egress Server**, not by the
        agent process itself.  When you pass the resulting message to
        :meth:`livekit.api.EgressService.start_room_composite_egress` (or any other egress
        RPC), LiveKit forwards the credentials string to the Egress Server, which uses it to
        write to GCS.

        **GKE Workload Identity**: Short-lived token-based credentials obtained from the GKE
        metadata server cannot be serialised into the JSON format expected by the Egress
        Server.  For GKE deployments the recommended approach is to run a **self-hosted
        LiveKit Egress Server** whose pod has a Workload Identity binding with GCS write
        permissions, and pass an empty ``credentials`` string so that the Egress Server
        authenticates using its own ambient credentials:

        .. code-block:: python

            from livekit.api import GCPUpload

            # Egress Server pod must have GCS write permissions via Workload Identity.
            gcp_upload = GCPUpload(bucket="my-bucket")  # credentials="" → Egress uses its own ADC

    Args:
        bucket: GCS bucket name (without the ``gs://`` prefix).
        credentials_info: Service account key as a dict (e.g. loaded with ``json.load``).
            When provided, takes precedence over ``credentials_file`` and the environment
            variable.
        credentials_file: Path to a service account JSON key file.  When provided, takes
            precedence over the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable.

    Returns:
        A :class:`livekit.api.GCPUpload` protobuf message ready to be embedded in an egress
        output configuration such as :class:`livekit.protocol.egress.EncodedFileOutput`.

    Raises:
        ImportError: If ``livekit-api`` is not installed.
        FileNotFoundError: If ``credentials_file`` or ``GOOGLE_APPLICATION_CREDENTIALS``
            points to a file that does not exist.
        ValueError: If no credentials source is available or the resolved credentials file
            does not contain a service account key.

    Example — explicit service account key dict::

        import json
        from livekit.plugins.google.egress import build_gcp_upload

        with open("/path/to/sa-key.json") as f:
            sa_info = json.load(f)

        gcp_upload = build_gcp_upload("my-bucket", credentials_info=sa_info)

    Example — service account key file path::

        from livekit.plugins.google.egress import build_gcp_upload

        gcp_upload = build_gcp_upload("my-bucket", credentials_file="/path/to/sa-key.json")

    Example — via ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable::

        # $ export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
        from livekit.plugins.google.egress import build_gcp_upload

        gcp_upload = build_gcp_upload("my-bucket")

    Example — self-hosted Egress on GKE with Workload Identity (no credentials in agent)::

        from livekit.api import GCPUpload

        # The Egress Server pod handles auth directly via its Workload Identity binding.
        gcp_upload = GCPUpload(bucket="my-bucket")
    """
    try:
        from livekit.api import GCPUpload
    except ImportError as exc:
        raise ImportError(
            "livekit-api is required to use build_gcp_upload. "
            "Install it with: pip install livekit-api"
        ) from exc

    credentials_json = _resolve_credentials_json(
        credentials_info=credentials_info,
        credentials_file=credentials_file,
    )
    return GCPUpload(credentials=credentials_json, bucket=bucket)


async def build_gcp_upload_async(
    bucket: str,
    *,
    credentials_info: dict | None = None,
    credentials_file: str | None = None,
) -> livekit.api.GCPUpload:  # type: ignore[name-defined]  # noqa: F821
    """Async variant of :func:`build_gcp_upload`.

    Runs the blocking file I/O in a thread-pool executor so that it does not block the
    event loop.  Prefer this variant inside ``async def`` entrypoints.

    Args:
        bucket: GCS bucket name (without the ``gs://`` prefix).
        credentials_info: Service account key as a dict.
        credentials_file: Path to a service account JSON key file.

    Returns:
        A :class:`livekit.api.GCPUpload` protobuf message.
    """
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: build_gcp_upload(
            bucket,
            credentials_info=credentials_info,
            credentials_file=credentials_file,
        ),
    )


def _resolve_credentials_json(
    *,
    credentials_info: dict | None,
    credentials_file: str | None,
) -> str:
    """Return a service account key JSON string.

    Resolution order:
    1. credentials_info dict
    2. credentials_file path
    3. GOOGLE_APPLICATION_CREDENTIALS environment variable
    """
    if credentials_info is not None:
        _validate_service_account_info(credentials_info)
        return json.dumps(credentials_info)

    path = credentials_file or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if path:
        with open(path) as f:
            raw = f.read()
        info = json.loads(raw)
        _validate_service_account_info(info, path=path)
        return raw

    raise ValueError(
        "No GCP service account credentials found. Provide one of:\n"
        "  1. credentials_info — a service account key dict\n"
        "  2. credentials_file — a path to a service account key JSON file\n"
        "  3. Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of "
        "a service account key file\n"
        "\n"
        "If your agent runs on GKE with Workload Identity, pass GCPUpload(bucket=...) "
        "with empty credentials and grant your self-hosted Egress Server pod GCS write "
        "permissions instead.  See the livekit-plugins-google README for details."
    )


def _validate_service_account_info(info: dict, *, path: str | None = None) -> None:
    location = f" at {path!r}" if path else ""
    if info.get("type") != "service_account":
        raise ValueError(
            f"The credentials{location} are of type {info.get('type')!r}, not "
            "'service_account'.  GCPUpload.credentials requires a service account key JSON.  "
            "See the livekit-plugins-google README for the GKE Workload Identity alternative."
        )
