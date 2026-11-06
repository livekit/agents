//! The ambient-frame protocol: the host reads its own ambient storage and passes the id in; Rust
//! never reads host state, so Node's FFI-callback context bug (`warm_transfer.ts:176`) cannot occur.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// `i64`, not `u64`: napi maps `u64` to a `BigInt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameId(i64);

impl FrameId {
    /// "No ambient frame", and not an error; only `AgentTask.complete()`/`__await__` reject it.
    pub const ROOT: FrameId = FrameId(0);

    /// Never invalid, only stale or [`InboundFrame::Unknown`].
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

/// `stale` is rejected once at the door rather than per call site (cf. `agent_activity.py:3081`).
#[derive(Debug)]
pub struct Frame {
    pub traceparent: Option<String>,
    pub activity: i64,
    pub speech: Option<i64>,
    pub generation: Option<i64>,
    pub function_call: Option<String>,
    /// False unless the call is a tool or hook dispatch.
    pub inline_ok: bool,
    /// `agent_activity.py:1911`.
    pub idle_hold: bool,
    stale: AtomicBool,
}

impl Frame {
    pub fn root() -> Self {
        Frame {
            traceparent: None,
            activity: 0,
            speech: None,
            generation: None,
            function_call: None,
            inline_ok: false,
            idle_hold: false,
            stale: AtomicBool::new(false),
        }
    }

    /// Nothing can mint a frame born retired.
    pub fn new(
        activity: i64,
        speech: Option<i64>,
        function_call: Option<String>,
        inline_ok: bool,
        traceparent: Option<String>,
    ) -> Self {
        Frame { activity, speech, function_call, inline_ok, traceparent, ..Frame::root() }
    }

    /// A method because `stale` is private, so struct-update syntax is refused outside this module.
    pub fn with_generation(mut self, generation: i64) -> Self {
        self.generation = Some(generation);
        self
    }

    /// `agent_activity.py:1911`.
    pub fn with_idle_hold(mut self, idle_hold: bool) -> Self {
        self.idle_hold = idle_hold;
        self
    }

    pub fn is_stale(&self) -> bool {
        self.stale.load(Ordering::Acquire)
    }
}

/// Its own lock, guarding only a map read: an inbound lookup during an outbound call must not touch
/// the driver's state lock.
pub struct FrameTable {
    frames: Mutex<HashMap<FrameId, Arc<Frame>>>,
    /// Never reused, so a stale id always means "that frame's speech is done".
    next_id: AtomicI64,
    current: AtomicI64,
}

impl Default for FrameTable {
    fn default() -> Self {
        FrameTable::new()
    }
}

impl FrameTable {
    pub fn new() -> Self {
        FrameTable {
            frames: Mutex::new(HashMap::new()),
            // Ids start at 1: 0 is `FrameId::ROOT` and is never a slab entry.
            next_id: AtomicI64::new(1),
            current: AtomicI64::new(FrameId::ROOT.raw()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<FrameId, Arc<Frame>>> {
        self.frames.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn alloc(&self, frame: Frame) -> FrameId {
        let id = FrameId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.lock().insert(id, Arc::new(frame));
        id
    }

    pub fn get(&self, id: FrameId) -> Option<Arc<Frame>> {
        if id.is_root() {
            return None;
        }
        self.lock().get(&id).cloned()
    }

    /// Later inbound calls carrying it are stale.
    pub fn retire(&self, id: FrameId) {
        if let Some(f) = self.get(id) {
            f.stale.store(true, Ordering::Release);
        }
    }

    pub fn current(&self) -> FrameId {
        FrameId(self.current.load(Ordering::Acquire))
    }

    /// Restores the previous id on drop, even on unwind.
    pub fn enter(&self, id: FrameId) -> FrameScope<'_> {
        let previous = FrameId(self.current.swap(id.raw(), Ordering::AcqRel));
        FrameScope { table: self, previous }
    }

    /// There is deliberately no "read the ambient frame" entry point.
    pub fn resolve_inbound(&self, id: FrameId) -> InboundFrame {
        if id.is_root() {
            return InboundFrame::Root;
        }
        match self.get(id) {
            None => InboundFrame::Unknown,
            Some(f) if f.is_stale() => InboundFrame::Stale(f),
            Some(f) => InboundFrame::Live(f),
        }
    }

    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// `Unknown` is an id minted by a different session.
pub enum InboundFrame {
    Root,
    Live(Arc<Frame>),
    Stale(Arc<Frame>),
    Unknown,
}

impl InboundFrame {
    /// The spellings both bindings report: the `lk.frame.resolution` values of the registry.
    pub fn kind(&self) -> &'static str {
        use crate::telemetry::semconv::lk_frame_resolution as resolution;
        match self {
            InboundFrame::Root => resolution::ROOT,
            InboundFrame::Live(_) => resolution::LIVE,
            InboundFrame::Stale(_) => resolution::STALE,
            InboundFrame::Unknown => resolution::UNKNOWN,
        }
    }
}

#[must_use = "binding the scope to `_` ends it immediately; bind it to `_f`"]
pub struct FrameScope<'a> {
    table: &'a FrameTable,
    previous: FrameId,
}

impl Drop for FrameScope<'_> {
    fn drop(&mut self) {
        self.table.current.store(self.previous.raw(), Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speech_frame(speech: i64) -> Frame {
        Frame { speech: Some(speech), inline_ok: true, ..Frame::root() }
    }

    #[test]
    fn root_is_the_default_and_is_not_a_slab_entry() {
        let t = FrameTable::new();
        assert!(matches!(t.resolve_inbound(FrameId::ROOT), InboundFrame::Root));
        assert!(t.get(FrameId::ROOT).is_none());
        assert!(t.is_empty());
    }

    #[test]
    fn ids_are_monotonic_and_never_reused() {
        let t = FrameTable::new();
        let a = t.alloc(speech_frame(1));
        let b = t.alloc(speech_frame(2));
        assert!(b > a, "ids must be monotonic");
        t.retire(a);
        let c = t.alloc(speech_frame(3));
        assert!(c > b, "retiring must not free the id for reuse");
    }

    #[test]
    fn a_retired_frame_is_stale_not_invalid() {
        let t = FrameTable::new();
        let id = t.alloc(speech_frame(1));
        assert!(matches!(t.resolve_inbound(id), InboundFrame::Live(_)));
        t.retire(id);
        match t.resolve_inbound(id) {
            InboundFrame::Stale(f) => assert_eq!(f.speech, Some(1), "the frame is still readable"),
            _ => panic!("a retired frame must resolve to Stale, never to an error"),
        }
    }

    #[test]
    fn an_id_from_another_session_is_unknown_not_a_panic() {
        let a = FrameTable::new();
        let b = FrameTable::new();
        let id = a.alloc(speech_frame(1));
        assert!(matches!(b.resolve_inbound(id), InboundFrame::Unknown));
    }


    #[test]
    fn the_scope_restores_the_previous_frame_and_nests() {
        let t = FrameTable::new();
        let outer = t.alloc(speech_frame(1));
        let inner = t.alloc(speech_frame(2));
        assert_eq!(t.current(), FrameId::ROOT);
        {
            let _f = t.enter(outer);
            assert_eq!(t.current(), outer);
            {
                let _g = t.enter(inner);
                assert_eq!(t.current(), inner);
            }
            assert_eq!(t.current(), outer, "the inner scope must restore, not clear");
        }
        assert_eq!(t.current(), FrameId::ROOT);
    }

    #[test]
    fn the_scope_restores_on_unwind() {
        let t = FrameTable::new();
        let id = t.alloc(speech_frame(1));
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _f = t.enter(id);
            panic!("boom");
        }));
        assert!(r.is_err());
        assert_eq!(
            t.current(),
            FrameId::ROOT,
            "a panic inside a frame scope must not pin the frame"
        );
    }


}
