use super::execution::Report;
use lk_agents_abi::agent::run::SayOutcome;
use lk_agents_abi::errors::{ApiError, ErrorKind, HostError, ProviderError};
use lk_agents_abi::{BoxFuture, Spawner};
use lk_agents_core::agent_run::{CallId, ToolOutputId};
use lk_agents_net::{signal, task::TaskSet};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::{Arc, Weak};
use tokio::sync::mpsc;

const MAX_PENDING_REPORTS: usize = 128;
const MAX_PENDING_BYTES: usize = crate::a2a::door::MAX_PAYLOAD_BYTES;
#[cfg(test)]
#[path = "progress_tests.rs"]
mod tests;

/// An admitted report's settlement: progress settles `Delivered` once admitted, exact output
/// with its outcome at the output boundary; `Err` is a report that was never admitted.
type Settled = Result<SayOutcome, ProviderError>;
type Completion = signal::Cell<Option<Settled>>;
type Key = (String, u64);

#[derive(Clone)]
pub struct Progress {
    invocation: String,
    events: mpsc::Sender<(String, Report)>,
    owner: Weak<TaskSet>,
    spawner: Arc<dyn Spawner>,
    state: Arc<Mutex<State>>,
    /// Whether this execution has a live output boundary for exact text.
    exact_output: bool,
}

#[derive(Default)]
struct State {
    next: u64,
    pending: usize,
    bytes: usize,
    operations: BTreeMap<Key, Operation>,
}

enum Operation {
    Open { call: CallId, tail: Option<Completion> },
    Closing { tail: Option<Completion> },
}

struct Permit {
    state: Arc<Mutex<State>>,
    bytes: usize,
}
impl Drop for Permit {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.pending -= 1;
        state.bytes -= self.bytes;
    }
}

pub(crate) struct Receipt {
    done: signal::Producer<Option<Settled>>,
    permit: Permit,
}
impl Receipt {
    pub(crate) fn finish(self, result: Settled) {
        let Self { done, permit } = self;
        // Return capacity before a resumed host can enqueue its next awaited update.
        drop(permit);
        done.resolve(result);
    }
}

pub(crate) struct OperationScope {
    progress: Progress,
    operation: u64,
}
impl Drop for OperationScope {
    fn drop(&mut self) {
        self.progress
            .state
            .lock()
            .operations
            .remove(&(self.progress.invocation.clone(), self.operation));
    }
}
impl OperationScope {
    /// Normal body completion fences all previously admitted subwork. Failure belongs to the
    /// optional reader, not to an unawaiting business result; producer loss still ends the fence.
    pub(crate) async fn settle(&self) {
        let key = (self.progress.invocation.clone(), self.operation);
        let tail = {
            let mut state = self.progress.state.lock();
            let Some(operation) = state.operations.get_mut(&key) else {
                return;
            };
            let old = std::mem::replace(operation, Operation::Closing { tail: None });
            let tail = match old {
                Operation::Open { tail, .. } | Operation::Closing { tail } => tail,
            };
            *operation = Operation::Closing { tail: tail.clone() };
            tail
        };
        if let Some(tail) = tail {
            match tail.wait().await {
                Ok(Ok(_)) | Ok(Err(_)) | Err(signal::Ended) => {}
            }
        }
    }
}

fn ended() -> ProviderError {
    HostError::new("the tool's native report owner has ended").into()
}
pub(crate) fn no_output() -> ProviderError {
    HostError::new("exact tool output requires a live voice, text stream or A2A output").into()
}

impl Progress {
    pub(crate) fn new(
        invocation: String,
        events: mpsc::Sender<(String, Report)>,
        owner: &Arc<TaskSet>,
        spawner: Arc<dyn Spawner>,
        exact_output: bool,
    ) -> Self {
        let owner = Arc::downgrade(owner);
        Self { invocation, events, owner, spawner, state: Arc::default(), exact_output }
    }
    pub(crate) fn for_invocation(&self, invocation: String) -> Self {
        Self { invocation, ..self.clone() }
    }
    pub(crate) fn open(&self, operation: u64, call: CallId) -> OperationScope {
        let previous = self
            .state
            .lock()
            .operations
            .insert((self.invocation.clone(), operation), Operation::Open { call, tail: None });
        debug_assert!(previous.is_none(), "one native host operation owns each report scope");
        OperationScope { progress: self.clone(), operation }
    }
    pub fn ended(&self) -> bool {
        self.events.is_closed()
    }
    pub(super) fn spawn_output(
        &self,
        producer: signal::Producer<Option<SayOutcome>>,
        future: BoxFuture<'static, SayOutcome>,
    ) -> Option<lk_agents_net::task::ScopedTask> {
        self.owner.upgrade()?.try_scoped(&*self.spawner, "tool exact output", async move {
            producer.resolve(future.await);
        })
    }

    pub fn enqueue(
        &self,
        operation: u64,
        message: String,
        silent: bool,
    ) -> BoxFuture<'static, Result<(), ProviderError>> {
        match self.submit(operation, message, false, silent) {
            Ok(receipt) => {
                Box::pin(async move { receipt.wait().await.map_err(|_| ended())?.map(drop) })
            }
            Err(error) => Box::pin(async move { Err(error) }),
        }
    }
    /// Refused before admission without a live output boundary or capacity. Once admitted, the
    /// reader settles with its outcome; an owner ending first interrupted the output.
    pub fn say(
        &self,
        operation: u64,
        message: String,
    ) -> Result<BoxFuture<'static, SayOutcome>, ProviderError> {
        if !self.exact_output {
            return Err(no_output());
        }
        let receipt = self.submit(operation, message, true, false)?;
        Ok(Box::pin(async move {
            match receipt.wait().await {
                Ok(Ok(outcome)) => outcome,
                Ok(Err(_)) => SayOutcome::Failed,
                Err(signal::Ended) => SayOutcome::Interrupted,
            }
        }))
    }
    /// Compatibility for native callers that always await admission.
    pub async fn report(&self, operation: u64, message: String) -> Result<(), ProviderError> {
        self.enqueue(operation, message, false).await
    }

    fn submit(
        &self,
        operation: u64,
        text: String,
        exact: bool,
        silent: bool,
    ) -> Result<Completion, ProviderError> {
        let Some(owner) = self.owner.upgrade().filter(|_| !self.events.is_closed()) else {
            return Err(ended());
        };
        if text.len() > MAX_PENDING_BYTES {
            return Err(HostError {
                kind: lk_agents_abi::errors::HostErrorKind::TooLarge,
                ..HostError::new("tool output exceeds the local output payload limit")
            }
            .into());
        }
        let (done, receipt) = signal::once();
        let bytes = text.len();
        let (output, call, before) = {
            let mut state = self.state.lock();
            if state.pending >= MAX_PENDING_REPORTS
                || state.bytes.saturating_add(bytes) > MAX_PENDING_BYTES
            {
                return Err(ApiError::new(
                    ErrorKind::RateLimit,
                    "the native tool report admission budget is full",
                )
                .into());
            }
            let key = (self.invocation.clone(), operation);
            let Some(Operation::Open { call, tail }) = state.operations.get_mut(&key) else {
                return Err(ended());
            };
            let call = *call;
            let before = tail.replace(receipt.clone());
            state.next += 1;
            state.pending += 1;
            state.bytes += bytes;
            (ToolOutputId(state.next), call, before)
        };
        let done = Receipt { done, permit: Permit { state: self.state.clone(), bytes } };
        let events = self.events.clone();
        let invocation = self.invocation.clone();
        let work = async move {
            if let Some(before) = before {
                match before.wait().await {
                    Ok(_) => {}
                    // The owner ended before this report's turn: an exact output was cut off.
                    Err(signal::Ended) => {
                        done.finish(if exact { Ok(SayOutcome::Interrupted) } else { Err(ended()) });
                        return;
                    }
                }
            }
            if exact {
                let report = Report::ToolOutputRequested { output, call, text, done };
                if let Err(mpsc::error::SendError((_, Report::ToolOutputRequested { done, .. }))) =
                    events.send((invocation, report)).await
                {
                    done.finish(Ok(SayOutcome::Interrupted));
                }
            } else {
                let report = Report::Progress { call, message: text, silent };
                let result = events.send((invocation, report)).await.map_err(|_| ended());
                done.finish(result.map(|()| SayOutcome::Delivered));
            }
        };
        if !owner.try_spawn(&*self.spawner, "owned tool report admission", work) {
            return Err(ended());
        }
        Ok(receipt)
    }
}
