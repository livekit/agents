//! On-demand native stream consumption and owned source/resource settlement.

pub(crate) mod input;

#[cfg(test)]
#[path = "../run_stream_tests.rs"]
mod tests;

use crate::agent_run::resources::RunResources;
use input::{Input, Message};
use lk_agents_abi::agent::resources::RunResources as _;
use lk_agents_abi::agent::run::InvalidRun;
use lk_agents_abi::agent::stream::PresentedEvent;
use lk_agents_abi::agent::stream::{self, StreamError, StreamOutcome, StreamRefused};
use lk_agents_abi::text_session::TextSessionDelivery;
use lk_agents_abi::usage::Usage;
use lk_agents_abi::{BoxFuture, Spawner};
use lk_agents_core::run_stream::{
    Stream as Policy, View,
    projection::{Item, Projection},
};
use lk_agents_net::{env::ProcessEnv, task::TaskSet};
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock};
use tokio::sync::{Mutex as AsyncMutex, watch};

type Outcome = Result<StreamOutcome, StreamError>;
pub struct RunStream(Arc<Inner>);

struct Inner {
    env: ProcessEnv,
    policy: Mutex<Policy>,
    projection: Mutex<Projection>,
    source: AsyncMutex<Option<Box<dyn Input>>>,
    resources: OnceLock<Arc<RunResources>>,
    spawner: OnceLock<Arc<dyn Spawner>>,
    usage: Mutex<Option<watch::Receiver<Usage>>>,
    terminal: Mutex<Option<Outcome>>,
    closing: watch::Sender<bool>,
    closed: watch::Sender<bool>,
    tasks: Arc<TaskSet>,
}

impl RunStream {
    /// Reserve the binding before starting a runner. Failure still owns resource settlement.
    pub(crate) fn bind(
        &self,
        resources: Arc<RunResources>,
        build: impl FnOnce() -> Result<Box<dyn Input>, InvalidRun>,
    ) -> Result<(), InvalidRun> {
        let spawner = (self.0.env.spawner)().map_err(InvalidRun)?;
        self.0.policy.lock().attach().map_err(|e| InvalidRun(e.to_string()))?;
        let _ = self.0.spawner.set(spawner);
        let _ = self.0.resources.set(resources);
        match build() {
            Ok(input) => {
                let usage = input.usage();
                let retired = self.0.usage.lock().replace(usage);
                drop(retired);
                *self.0.source.try_lock().expect("unready stream cannot be read") = Some(input);
                self.0.policy.lock().ready();
                Ok(())
            }
            Err(error) => {
                self.0.finish(Err(StreamError::Start(error.clone())));
                Err(error)
            }
        }
    }
}

impl Inner {
    fn finish(self: &Arc<Self>, outcome: Outcome) {
        if !self.policy.lock().close() {
            return;
        }
        self.closing.send_replace(true);
        let inner = self.clone();
        let spawner = self.spawner.get().expect("bound stream owns its spawner");
        self.tasks.spawn(&**spawner, "run stream settlement", async move {
            let source = inner.source.lock().await.take();
            if let Some(source) = source {
                let deadline =
                    inner.env.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
                let closed = source.close(deadline).await;
                if !matches!(closed, lk_agents_abi::Closed::Clean) {
                    log::error!("the run's source cleanup missed its deadline: {closed:?}");
                }
            }
            let cleanup = match inner.resources.get() {
                Some(resources) => resources.aclose().await.map_err(StreamError::Resources),
                None => Ok(()),
            };
            let projection = std::mem::take(&mut *inner.projection.lock());
            drop(projection);
            let result = cleanup.and(outcome);
            let retired = inner.terminal.lock().replace(result);
            drop(retired);
            inner.policy.lock().settled();
            inner.closed.send_replace(true);
        });
    }
    async fn settled(&self) -> Outcome {
        let mut closed = self.closed.subscribe();
        let _ = closed.wait_for(|done| *done).await;
        self.terminal.lock().as_ref().cloned().ok_or(StreamError::OutcomeTaken)?
    }
    async fn exhausted(&self) -> Result<(), StreamError> {
        match self.settled().await {
            Ok(_) | Err(StreamError::OutcomeTaken) => Ok(()),
            Err(error) => Err(error),
        }
    }
    async fn pull(
        self: &Arc<Self>,
        consumer: u64,
        cursor_closed: Option<&watch::Sender<bool>>,
    ) -> Result<Option<Item>, StreamError> {
        let (view, _reading) = if cursor_closed.is_some() {
            (self.policy.lock().read_view(consumer)?, None)
        } else {
            let view = self.policy.lock().begin_read(consumer)?;
            (view, Some(Reading { inner: self.clone(), consumer }))
        };
        let mut cancelling = self.closing.subscribe();
        loop {
            if cursor_closed.is_some_and(|closed| *closed.borrow()) {
                return Ok(None);
            }
            if *cancelling.borrow_and_update() {
                self.exhausted().await?;
                return Ok(None);
            }
            let mut input = self.source.lock().await;
            // A cancelled bridge waiter leaves its item pending for the next consumer. Check
            // under source ownership too: an older in-flight read may have just completed.
            let pending = self.projection.lock().retry(view, consumer);
            if pending.is_some() {
                return Ok(pending);
            }
            let Some(source) = input.as_mut() else {
                drop(input);
                self.exhausted().await?;
                return Ok(None);
            };
            let next = tokio::select! {
                biased;
                _ = signal(cursor_closed) => return Ok(None),
                _ = cancelling.changed() => { drop(input); self.exhausted().await?; return Ok(None); },
                next = source.next() => next,
            };
            match next {
                Ok(Some(Message::Event(event))) => {
                    let selected = self.projection.lock().select(event, view, consumer);
                    if selected.is_some() {
                        return Ok(selected);
                    }
                }
                Ok(Some(Message::Release(invocation_id))) => {
                    let messages = self.projection.lock().release();
                    if let Err(error) =
                        source.release(TextSessionDelivery { invocation_id, messages })
                    {
                        drop(input);
                        self.finish(Err(StreamError::Start(error)));
                        self.exhausted().await?;
                        return Ok(None);
                    }
                }
                Ok(Some(Message::Incomplete(pending))) => {
                    log::error!("the run's cleanup missed its deadline: {pending:?}");
                }
                other => {
                    drop(input);
                    let outcome = match other {
                        Ok(Some(Message::Terminal(outcome))) => Ok(outcome),
                        Err(error) => Err(StreamError::Provider(error)),
                        _ => Err(StreamError::Cancelled),
                    };
                    self.finish(outcome);
                    self.exhausted().await?;
                    return Ok(None);
                }
            }
        }
    }
    fn release(&self, consumer: u64) {
        self.policy.lock().release(consumer);
    }
}

async fn signal(sender: Option<&watch::Sender<bool>>) {
    match sender {
        Some(sender) => {
            let _ = sender.subscribe().wait_for(|closed| *closed).await;
        }
        None => std::future::pending::<()>().await,
    }
}
struct Reading {
    inner: Arc<Inner>,
    consumer: u64,
}
impl Drop for Reading {
    fn drop(&mut self) {
        self.inner.policy.lock().end_read(self.consumer);
    }
}
impl Drop for RunStream {
    fn drop(&mut self) {
        self.0.finish(Err(StreamError::Cancelled));
    }
}

impl stream::RunStream for RunStream {
    type Env = ProcessEnv;
    type Events = Events;
    type Text = Text;
    fn new(env: &ProcessEnv) -> Self {
        Self(Arc::new(Inner {
            env: env.clone(),
            policy: Mutex::default(),
            projection: Mutex::default(),
            source: AsyncMutex::new(None),
            resources: OnceLock::new(),
            spawner: OnceLock::new(),
            usage: Mutex::default(),
            terminal: Mutex::default(),
            closing: watch::channel(false).0,
            closed: watch::channel(false).0,
            tasks: Arc::default(),
        }))
    }
    fn enter(&self) -> Result<(), StreamRefused> {
        self.0.policy.lock().enter()
    }
    fn events(&self) -> Result<Events, StreamRefused> {
        let consumer = self.0.policy.lock().claim(View::Events)?;
        Ok(Events(Cursor::new(self.0.clone(), consumer)))
    }
    fn text(&self, delta: bool) -> Result<Text, StreamRefused> {
        let consumer = self.0.policy.lock().claim(View::Text { delta })?;
        Ok(Text(Cursor::new(self.0.clone(), consumer)))
    }
    fn begin_result(&self) -> Result<u64, StreamRefused> {
        self.0.policy.lock().claim(View::Result)
    }
    fn end_result(&self, consumer: u64, hydrated: bool) {
        self.0.release(consumer);
        if hydrated {
            let retired = self.0.terminal.lock().take();
            drop(retired);
        }
    }
    fn result(&self, consumer: u64) -> BoxFuture<'static, Outcome> {
        let inner = self.0.clone();
        Box::pin(async move {
            inner.pull(consumer, None).await?;
            inner.settled().await
        })
    }
    fn cancel(&self) {
        self.0.finish(Err(StreamError::Cancelled));
    }
    fn aclose(&self) -> BoxFuture<'static, Result<(), StreamError>> {
        if !self.0.policy.lock().attached() {
            return Box::pin(async { Ok(()) });
        }
        self.0.finish(Err(StreamError::Cancelled));
        let inner = self.0.clone();
        Box::pin(async move {
            match inner.settled().await {
                Err(StreamError::Cancelled | StreamError::OutcomeTaken) | Ok(_) => Ok(()),
                Err(error) => Err(error),
            }
        })
    }
    fn rollback(&self) -> BoxFuture<'static, ()> {
        let closing = <Self as stream::RunStream>::aclose(self);
        Box::pin(async move {
            if let Err(error) = closing.await {
                log::warn!("the failed stream could not close: {error}");
            }
        })
    }
    fn done(&self) -> bool {
        self.0.policy.lock().done()
    }
    fn cancelled(&self) -> bool {
        matches!(&*self.0.terminal.lock(), Some(Err(StreamError::Cancelled)))
    }
    fn usage(&self) -> Usage {
        self.0.usage.lock().as_ref().map(|u| u.borrow().clone()).unwrap_or_default()
    }
}

// A cursor owns consumer admission even before its first `next()` is polled.

struct Cursor {
    inner: Arc<Inner>,
    consumer: u64,
    closed: watch::Sender<bool>,
}
impl Cursor {
    fn new(inner: Arc<Inner>, consumer: u64) -> Self {
        Self { inner, consumer, closed: watch::channel(false).0 }
    }
    fn close(&self) {
        self.closed.send_replace(true);
        self.inner.release(self.consumer);
    }
    fn begin_read(&self) -> Result<bool, StreamError> {
        if *self.closed.borrow() {
            return Ok(false);
        }
        self.inner.policy.lock().begin_read(self.consumer)?;
        Ok(true)
    }
    fn end_read(&self, accepted: bool) {
        self.inner.projection.lock().acknowledge(self.consumer, accepted);
        self.inner.policy.lock().end_read(self.consumer);
    }
    fn close_idle(&self) -> Result<(), StreamError> {
        self.inner.policy.lock().close_cursor(self.consumer)?;
        self.closed.send_replace(true);
        Ok(())
    }
    async fn next(&self) -> Result<Option<Item>, StreamError> {
        if *self.closed.borrow() {
            return Ok(None);
        }
        let item = self.inner.pull(self.consumer, Some(&self.closed)).await;
        if !matches!(item, Ok(Some(_))) {
            self.close();
        }
        item
    }
}
impl Drop for Cursor {
    fn drop(&mut self) {
        self.close();
    }
}
pub struct Events(Cursor);
pub struct Text(Cursor);
impl stream::StreamEvents for Events {
    fn begin_read(&self) -> Result<bool, StreamError> {
        self.0.begin_read()
    }
    fn end_read(&self, accepted: bool) {
        self.0.end_read(accepted);
    }
    async fn recv(&self) -> Result<Option<PresentedEvent>, StreamError> {
        match self.0.next().await? {
            Some(Item::Event(event)) => Ok(Some(*event)),
            _ => Ok(None),
        }
    }
    async fn aclose(&self) -> Result<(), StreamError> {
        self.0.close_idle()
    }
}
impl stream::StreamText for Text {
    fn begin_read(&self) -> Result<bool, StreamError> {
        self.0.begin_read()
    }
    fn end_read(&self, accepted: bool) {
        self.0.end_read(accepted);
    }
    async fn recv(&self) -> Result<Option<String>, StreamError> {
        match self.0.next().await? {
            Some(Item::Text(text)) => Ok(Some(text)),
            _ => Ok(None),
        }
    }
    async fn aclose(&self) -> Result<(), StreamError> {
        self.0.close_idle()
    }
}
