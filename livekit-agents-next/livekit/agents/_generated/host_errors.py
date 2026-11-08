# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/errors/host_capture.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from builtins import (
        BaseException as _HostProviderErrorMetadata,
        BaseException as _HostRealtimeErrorMetadata,
    )

ProviderErrorMetadata: TypeAlias = "_HostProviderErrorMetadata"


RealtimeErrorMetadata: TypeAlias = "_HostRealtimeErrorMetadata"
