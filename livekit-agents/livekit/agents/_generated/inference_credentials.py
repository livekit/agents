# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/inference/mod.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations


@dataclass(frozen=True)
class Credentials:
    """The inference gateway's credentials and identification. The core reads
    `LIVEKIT_INFERENCE_URL`, `LIVEKIT_INFERENCE_API_KEY`, `LIVEKIT_INFERENCE_API_SECRET` and
    `LIVEKIT_WORKER_TOKEN` from the environment, and the server's `LIVEKIT_URL` and key pair from
    the running server's settings (in a job, its `AgentServer`'s), else the environment.
    Where a gateway client connects, and with what.
    """

    base_url: str
    api_key: str
    api_secret: str


Credentials.__annotations__ = get_annotations(Credentials, eval_str=True)
Credentials.__module__ = "livekit.agents.inference"
