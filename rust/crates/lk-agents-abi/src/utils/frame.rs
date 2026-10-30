//! The ambient-frame protocol: the host reads its own ambient storage and passes the id in; Rust
//! never reads host state, so Node's FFI-callback context bug (`warm_transfer.ts:176`) cannot occur.

use std::fmt;

/// `i64`, not `u64`: napi maps `u64` to a `BigInt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameId(i64);

impl FrameId {
    /// "No ambient frame", and not an error; only `AgentTask.complete()`/`__await__` reject it.
    pub const ROOT: FrameId = FrameId(0);

    /// Never invalid, only stale or unknown to this session.
    pub const fn from_raw(raw: i64) -> Self {
        FrameId(raw)
    }

    pub const fn raw(self) -> i64 {
        self.0
    }

    pub const fn is_root(self) -> bool {
        self.0 == FrameId::ROOT.0
    }
}

impl Default for FrameId {
    fn default() -> Self {
        FrameId::ROOT
    }
}

impl fmt::Display for FrameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_root() { f.write_str("root") } else { write!(f, "frame:{}", self.0) }
    }
}
