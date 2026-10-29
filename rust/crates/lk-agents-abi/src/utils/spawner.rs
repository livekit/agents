//! The task-ownership contract: an injected [`Spawner`] hands back an [`OwnedTask`]. The
//! executor-agnostic implementation and the Tokio executor live in `lk_agents_net::task`.

use std::any::Any;
use std::future::Future;
use std::panic::Location;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// A boxed `Send` future, the shape futures cross the ABI in.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Which way a task ended; the first of completion and cancellation to happen wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOutcome {
    Completed,
    Cancelled,
    Panicked,
}

/// Why an [`OwnedTask`] produced no value.
#[derive(Debug)]
pub enum TaskError {
    /// Cancelled by its owner, or dropped unfinished by its executor.
    Cancelled,
    /// The work panicked, with the panic's message. The panic itself still unwinds into the
    /// executor once the outcome is reported, so a harness judging its tasks sees it.
    Panicked(String),
}

impl TaskError {
    pub fn outcome(&self) -> TaskOutcome {
        match self {
            TaskError::Cancelled => TaskOutcome::Cancelled,
            TaskError::Panicked(_) => TaskOutcome::Panicked,
        }
    }
}

impl std::fmt::Display for TaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskError::Cancelled => f.write_str("task cancelled"),
            TaskError::Panicked(message) => write!(f, "task panicked: {message}"),
        }
    }
}

impl std::error::Error for TaskError {}

/// The executor's side of an [`OwnedTask`].
pub trait TaskControl: Send + Sync {
    /// Requests cancellation; idempotent.
    fn cancel(&self);
    /// Settled: the work ended or was dropped and the outcome was reported.
    fn is_finished(&self) -> bool;
}

/// The one way to own spawned work: dropping it requests cancellation, never detaches; awaiting
/// it yields the work's value or why there is none, only after the work and everything it owned
/// were dropped and its settle step ran.
#[must_use = "dropping an OwnedTask cancels it"]
pub struct OwnedTask<T = ()> {
    /// `None` only once handed to an [`AbortHandle`].
    control: Option<Arc<dyn TaskControl>>,
    /// `Sync` so an owner shared by reference (an `Arc`'d session, a callback) can hold it.
    output: TaskOutput<T>,
}

/// What an executor resolves once the task settled.
pub type TaskOutput<T> = Pin<Box<dyn Future<Output = Result<T, TaskError>> + Send + Sync>>;

impl<T: 'static> OwnedTask<T> {
    /// For executors: `output` resolves once the task settled.
    pub fn new(control: Arc<dyn TaskControl>, output: TaskOutput<T>) -> Self {
        OwnedTask { control: Some(control), output }
    }

    /// Requests cancellation; idempotent. The outcome is reported by the task, not implied here.
    pub fn cancel(&self) {
        if let Some(control) = &self.control {
            control.cancel();
        }
    }

    pub fn is_finished(&self) -> bool {
        self.control.as_ref().is_none_or(|control| control.is_finished())
    }

    /// Gives up ownership for a cancel-only handle; the work then ends on its own.
    pub fn into_abort_handle(mut self) -> AbortHandle {
        AbortHandle(self.control.take().expect("an owned task has its control"))
    }

    fn map<U: 'static>(mut self, f: impl FnOnce(T) -> U + Send + Sync + 'static) -> OwnedTask<U> {
        let output = std::mem::replace(&mut self.output, Box::pin(std::future::pending()));
        OwnedTask {
            control: self.control.take(),
            output: Box::pin(async move { output.await.map(f) }),
        }
    }
}

impl<T> Future for OwnedTask<T> {
    type Output = Result<T, TaskError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.output.as_mut().poll(cx)
    }
}

impl<T> Drop for OwnedTask<T> {
    fn drop(&mut self) {
        if let Some(control) = &self.control {
            control.cancel();
        }
    }
}

impl<T> std::fmt::Debug for OwnedTask<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let finished = self.control.as_ref().is_none_or(|control| control.is_finished());
        f.debug_struct("OwnedTask").field("finished", &finished).finish()
    }
}

/// A cancel-only handle on work nobody owns, from [`SpawnerExt::spawn_unboxed`]; call sites
/// still on it are being moved to [`OwnedTask`].
#[derive(Clone)]
pub struct AbortHandle(Arc<dyn TaskControl>);

impl AbortHandle {
    pub fn abort(&self) {
        self.0.cancel();
    }

    pub fn is_finished(&self) -> bool {
        self.0.is_finished()
    }
}

impl std::fmt::Debug for AbortHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AbortHandle").field("finished", &self.is_finished()).finish()
    }
}

/// Runs once a task's work was dropped, with how it ended; also when the owner cancelled it.
pub type Settle = Box<dyn FnOnce(TaskOutcome) -> BoxFuture<'static, ()> + Send>;

/// What a [`Spawner`] is asked to run: the work, its settle step and the line that spawned it
/// (so an executor can place a task it reports as stuck).
pub struct Launch {
    pub work: BoxFuture<'static, Box<dyn Any + Send>>,
    pub settled: Option<Settle>,
    pub site: &'static Location<'static>,
}

/// `tokio::spawn` panics on a host thread with no reactor, so the executor is injected. Every
/// executor (Tokio, the host loop's virtual time, the DST's scheduler) implements the same
/// [`OwnedTask`] contract, through `lk_agents_net::task::launch`.
pub trait Spawner: Send + Sync {
    fn launch(&self, task: Launch) -> OwnedTask<Box<dyn Any + Send>>;
}

/// Blanket-implemented, including for `dyn Spawner`.
pub trait SpawnerExt: Spawner {
    #[track_caller]
    fn spawn<T, F>(&self, work: F) -> OwnedTask<T>
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        let work = Box::pin(async move { Box::new(work.await) as Box<dyn Any + Send> });
        let task = self.launch(Launch { work, settled: None, site: Location::caller() });
        task.map(|value| *value.downcast::<T>().expect("a task yields its own output type"))
    }

    /// Runs `settled` with the outcome once `work` is dropped, also when the owner cancelled it.
    #[track_caller]
    fn spawn_settled<W, S, F>(&self, work: W, settled: S) -> OwnedTask
    where
        W: Future<Output = ()> + Send + 'static,
        S: FnOnce(TaskOutcome) -> F + Send + 'static,
        F: Future<Output = ()> + Send + 'static,
    {
        let work = Box::pin(async move { Box::new(work.await) as Box<dyn Any + Send> });
        let settled: Settle = Box::new(move |outcome| Box::pin(settled(outcome)));
        let task = self.launch(Launch { work, settled: Some(settled), site: Location::caller() });
        task.map(|_| ())
    }

    /// Work nobody owns, by choice: it ends on its own and nothing waits for it or cancels it.
    /// Each call site says why.
    #[track_caller]
    fn spawn_detached<F>(&self, work: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let work = Box::pin(async move { Box::new(work.await) as Box<dyn Any + Send> });
        drop(
            self.launch(Launch { work, settled: None, site: Location::caller() })
                .into_abort_handle(),
        );
    }

    /// Unowned work with a cancel-only handle.
    #[track_caller]
    fn spawn_unboxed<F>(&self, work: F) -> AbortHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let work = Box::pin(async move { Box::new(work.await) as Box<dyn Any + Send> });
        self.launch(Launch { work, settled: None, site: Location::caller() }).into_abort_handle()
    }
}

impl<S: Spawner + ?Sized> SpawnerExt for S {}
