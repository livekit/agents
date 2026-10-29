"""Explicit controls for first-party workers; application secrets stay outside."""

import os


def worker_environment() -> dict[str, str]:
    keys = (
        "LIVEKIT_BOUNDARY_PACKAGE_ROOT",
        "LIVEKIT_BOUNDARY_FAKE_INIT_READY",
        "LIVEKIT_BOUNDARY_FAKE_READY",
        "LIVEKIT_BOUNDARY_FAKE_DELAY",
        "LIVEKIT_BOUNDARY_FAKE_ADDON",
        "LIVEKIT_BOUNDARY_NATIVE_DIR",
        "LIVEKIT_BOUNDARY_NATIVE_HANG_LOG",
        "LIVEKIT_BOUNDARY_TRANSPORT",
        "LIVEKIT_BOUNDARY_TRANSPORT_ONCE",
        "LIVEKIT_NAPI_FAKE_MODE",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}
