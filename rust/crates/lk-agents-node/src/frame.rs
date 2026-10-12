//! The napi projection of §8e's ambient frame, and nothing else.
//!
//! **The protocol itself moved to [`lk_agents_abi::utils::frame`]** -- the slab, the RAII
//! [`FrameScope`](lk_agents_abi::utils::frame::FrameScope), the monotonic
//! [`FrameId`] and the four-way
//! [`InboundFrame`](lk_agents_abi::utils::frame::InboundFrame) resolution. It was written here first,
//! deliberately napi-free, precisely so it could be lifted once `lk-agents-py` needed the same
//! thing; two private copies of a protocol whose job is to make the two hosts agree would be the
//! divergence the whole project exists to remove. Read that module for the design; this file is the
//! last 40 lines of it, the part that is genuinely Node's.
//!
//! What is genuinely Node's is exactly one thing: [`FrameView`], the record the shim reads to
//! install its own `AsyncLocalStorage` store. Python builds the same projection from the same
//! `&Frame` with pyo3, into its own `_FrameVar` plus the four legacy `ContextVar`s.
//!
//! The rule the core enforces and this file inherits: **the host reads its own ambient storage and
//! passes the id in; Rust never reads host state.** Nothing here touches `AsyncLocalStorage`, and
//! nothing here can -- napi gives no way to read it, which is the point.
//!
//! # The Node-specific hazard this design is immune to
//!
//! `agents-js` documents the failure at `workflows/warm_transfer.ts:176-180`: a callback arriving
//! from a native FFI layer runs with the `AsyncLocalStorage` context that was live when the
//! *callback* was created, not the caller's. napi-rs `ThreadsafeFunction` has exactly that
//! property. So a binding that lets context *propagate* across the boundary reproduces the bug by
//! construction; one that passes the id as **data** and has the shim `als.run(...)` it cannot,
//! because the context is never inherited -- it is always installed. `node/@livekit-agents`'s
//! `frame.test.ts` builds the tsfn inside a *deliberately wrong* `als.run` and asserts the
//! callback still sees the frame Rust sent.
//!
//! # There is no pid guard here, and that is not an oversight
//!
//! `lk-agents-py`'s R13 `check_pid()` exists because a forked Python child inherits a core whose
//! threads did not survive. Node's `child_process.fork()` is a fresh process with a fresh addon
//! load, `worker_threads` get their own `AgentCore`, and `posix.fork()` is not reachable from JS.
//! The asymmetry is real and belongs in the bindings; the core stays free of it.

use lk_agents_abi::FrameId;
use lk_agents_abi::utils::frame::Frame;
use napi_derive::napi;

/// What the host shim needs to install its own ambient storage, and nothing more.
///
/// Python's `_bridge.py:_frame` sets `_FrameVar` plus the four legacy `ContextVar`s from this;
/// Node's `frame.ts:withFrame` does the same with `AsyncLocalStorage`. Both read these fields;
/// neither can reach the `Frame` itself.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct FrameView {
    pub id: i64,
    pub activity: i64,
    pub speech: Option<i64>,
    pub generation: Option<i64>,
    pub function_call: Option<String>,
    pub inline_ok: bool,
    pub idle_hold: bool,
    pub traceparent: Option<String>,
    /// True once the frame's speech finished or its activity was swapped. The shim installs a
    /// stale frame anyway -- rejecting it here would break the audio pump, which §8e says "warns
    /// at most and continues".
    pub stale: bool,
}

impl FrameView {
    pub fn root() -> Self {
        FrameView {
            id: FrameId::ROOT.raw(),
            activity: 0,
            speech: None,
            generation: None,
            function_call: None,
            inline_ok: false,
            idle_hold: false,
            traceparent: None,
            stale: false,
        }
    }

    pub fn of(id: FrameId, frame: &Frame) -> Self {
        FrameView {
            id: id.raw(),
            activity: frame.activity,
            speech: frame.speech,
            generation: frame.generation,
            function_call: frame.function_call.clone(),
            inline_ok: frame.inline_ok,
            idle_hold: frame.idle_hold,
            traceparent: frame.traceparent.clone(),
            stale: frame.is_stale(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::utils::frame::FrameTable;

    #[test]
    fn the_view_carries_everything_a_shim_installs_and_nothing_else() {
        let t = FrameTable::new();
        let id = t.alloc(
            Frame::new(
                7,
                Some(9),
                Some("call-1".to_owned()),
                true,
                Some("00-aaaa-bbbb-01".to_owned()),
            )
            .with_generation(11)
            .with_idle_hold(true),
        );
        let v = FrameView::of(id, &t.get(id).unwrap());
        assert_eq!(v.id, id.raw());
        assert_eq!(v.activity, 7);
        assert_eq!(v.speech, Some(9));
        assert_eq!(v.function_call.as_deref(), Some("call-1"));
        assert!(v.inline_ok && v.idle_hold);
        assert!(!v.stale);
        t.retire(id);
        assert!(FrameView::of(id, &t.get(id).unwrap()).stale);
    }

    #[test]
    fn the_root_view_is_the_one_the_shim_installs_when_there_is_no_frame() {
        // `FrameId::ROOT` is the default, not an error: `session.say()` from a job entrypoint has
        // no ambient frame and the shim still needs a store to install.
        let v = FrameView::root();
        assert_eq!(v.id, FrameId::ROOT.raw());
        assert!(!v.stale && !v.inline_ok);
        assert!(v.speech.is_none() && v.traceparent.is_none());
    }
}
