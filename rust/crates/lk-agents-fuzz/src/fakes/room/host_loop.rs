use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{Closed, StreamOutput, Ts};
use std::sync::{Arc, Mutex};

// An SDK media stream binds the loop it is created on. Generated adapters run a declared `await`
// on the host's loop and a declared `call` (or a reply read) on the core's thread, where no loop
// runs. The fakes bind where the real adapters do, recorded for `host_objects_bind_on_their_loop`.

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

/// A host stream opened by a declared `call` on the core's thread, whose SDK object its first
/// read creates on the loop (`room_io._AudioStream`, `_VideoStream`).
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
