# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
# and the `#[host(export)]` / `#[host(object)]` items in `EXPORT_SOURCES`.
#
# The stub of every `_core` class and function the binding generates rather than
# writes. `_core.pyi` re-exports this file.
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

WARM_TRANSFER_IDENTITY: str
TWILIO_RINGING_TIMEOUT: float
def rtc_wire_names() -> tuple[list[str], list[str], list[str], list[str], list[str]]: ...
def video_queue_frames() -> int: ...
TELEMETRY_MAX_DEPTH: int
TELEMETRY_MAX_EVENTS: int
SDK_RECORD_BYTES: int
SDK_PENDING_RECORDS: int
SDK_BATCH_ITEMS: int
SDK_SCHEDULE_DELAY_MILLIS: int
SDK_EXPORT_TIMEOUT_MILLIS: int
SDK_MAX_PROVIDERS: int
SDK_SHUTDOWN_TIMEOUT_SECONDS: float
UPLOAD_TIMEOUT_SECONDS: int
REDACTED_EXCEPTION: str
REDACTED_LOG_BODY: str


class StaleHandle(RuntimeError):
    """A handle whose originating host no longer owns a live entry."""


class SessionClosing(RuntimeError):
    """`wait_for_idle()`/`wait_for_idle_and_hold()` refused: the session is closing."""
