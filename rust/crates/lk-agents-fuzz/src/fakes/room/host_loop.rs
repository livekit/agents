//! The host's event loop as the room's SDK objects see it (rust-core.md "Simulated host
//! execution"). An SDK media stream binds the loop it is created on, and a stream writer a call
//! returns is taken in some context, its calls running there. The generated adapters run a
//! declared `await` on the host's loop in its call's context, and a declared `call` (or the read of
//! a reply) on the core's thread, where no loop runs. The fakes bind their loop-bound objects where
//! the real adapters bind them, and each binding is recorded for `host_objects_bind_on_their_loop`.
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{Closed, StreamOutput, Ts};
use std::sync::{Arc, Mutex};

/// Where a host object was bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Thread {
    /// On the host's event loop, in the context of the call that produced it.
    Loop,
    /// On the core's thread: inside a declared `call`, or as a reply was read off the loop.
    Core,
}

/// Told of each loop-bound host object, by its host class, and where it was bound.
pub type BindObserver = dyn Fn(&'static str, Thread) + Send + Sync;

/// One room's host loop: who is told of the objects bound to it.
#[derive(Default)]
pub struct HostLoop(Mutex<Option<Arc<BindObserver>>>);

impl HostLoop {
    pub fn observe(&self, observer: Arc<BindObserver>) {
        *self.0.lock().unwrap() = Some(observer);
    }

    pub fn bind(&self, object: &'static str, thread: Thread) {
        let observer = self.0.lock().unwrap().clone();
        if let Some(observer) = observer {
            observer(object, thread);
        }
    }
}

/// A host stream whose SDK object its first read creates, on the loop: the core opens it with a
/// declared `call`, on its own thread (`room_io._AudioStream`, `_VideoStream`).
pub(super) struct OpenedOnRead<S> {
    pub(super) object: &'static str,
    pub(super) host: Arc<HostLoop>,
    pub(super) stream: S,
    pub(super) opened: bool,
}

impl<T: Send + 'static, S: StreamOutput<T>> StreamOutput<T> for OpenedOnRead<S> {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<T>, ProviderError>> {
        if !std::mem::replace(&mut self.opened, true) {
            self.host.bind(self.object, Thread::Loop);
        }
        self.stream.next()
    }

    fn cancel_safe(&self) -> bool {
        self.stream.cancel_safe()
    }

    fn close(self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::new(self.stream).close(deadline)
    }
}
