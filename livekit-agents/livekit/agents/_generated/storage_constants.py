# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/storage/constants.rs

from __future__ import annotations

from typing import Final

from .. import _core

DEFAULT_PARTITION_KEY: Final[str] = _core.DEFAULT_PARTITION_KEY
"""The protocol's partition key when an application omits one.
The framework resolves this default before issuing storage operations.
"""
