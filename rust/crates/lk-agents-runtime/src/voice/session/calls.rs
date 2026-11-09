use crate::voice::VoiceRuntimeEvent;
use crate::voice::session::Tx;
use crate::voice::speech_handle::{SpeechHandle, SpeechSignals};
use lk_agents_voice::agent_session::{
    Answer, Command, QueueClosed, RequestId, Sent, VoiceCommandOrigin, VoiceEvent,
};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::oneshot;

/// Where an awaited input's answer goes. It travels with the input that names its call; the
/// loop keeps it until the core answers that call, and drops it unanswered when it ends.
pub enum ReplyTo {
    Answer(oneshot::Sender<Answer>),
    /// A `say()` / `generate_reply()`: [`Answer::Speech`] marks the speech created; another
    /// answer, or none, marks it lost.
    Speech(SpeechReply),
}

impl ReplyTo {
    pub fn settle(self, answer: Answer) {
        match self {
            ReplyTo::Answer(tx) => {
                let _ = tx.send(answer);
            }
            ReplyTo::Speech(speech) => speech.answer(None, matches!(answer, Answer::Speech)),
        }
    }
}

impl std::fmt::Debug for ReplyTo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplyTo::Answer(_) => f.write_str("ReplyTo::Answer"),
            ReplyTo::Speech(_) => f.write_str("ReplyTo::Speech"),
        }
    }
}

/// Lost unless answered first. Carries the speech's writers to the loop, which takes them as it
/// receives the call.
pub struct SpeechReply {
    handle: Option<Arc<SpeechHandle>>,
    signals: Option<SpeechSignals>,
}

impl SpeechReply {
    pub fn new(handle: Arc<SpeechHandle>, signals: SpeechSignals) -> Self {
        SpeechReply { handle: Some(handle), signals: Some(signals) }
    }

    /// The writers, under the speech's reserved id, for the loop's table.
    pub fn take_signals(
        &mut self,
    ) -> Option<(
        lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId,
        SpeechSignals,
    )> {
        let id = self.handle.as_ref()?.reserved()?;
        Some((id, self.signals.take()?))
    }

    pub fn reserved(
        &self,
    ) -> Option<lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId> {
        self.handle.as_ref()?.reserved()
    }

    /// The loop answered: `signals` are the speech's writers from its table.
    pub fn answer(mut self, signals: Option<&SpeechSignals>, created: bool) {
        if let Some(handle) = self.handle.take() {
            handle.answered(signals.or(self.signals.as_ref()), created);
        }
    }
}

impl Drop for SpeechReply {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.answered(None, false);
        }
    }
}

/// An awaited call: its id goes in the input, its reply is sent with it.
#[derive(Debug)]
pub struct Call {
    pub id: RequestId,
    pub reply_to: ReplyTo,
}

/// The answer to an awaited call, picked by kind.
pub struct Awaited<T> {
    rx: oneshot::Receiver<Answer>,
    pick: fn(Answer) -> Option<T>,
}

/// The session ended before answering, or answered with another kind.
#[derive(Debug)]
pub struct Unanswered;

impl<T> std::future::Future for Awaited<T> {
    type Output = Result<T, Unanswered>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let pick = self.pick;
        std::pin::Pin::new(&mut self.rx)
            .poll(cx)
            .map(|answer| answer.ok().and_then(pick).ok_or(Unanswered))
    }
}

/// A call awaiting the answer `pick` takes.
pub fn open<T>(id: RequestId, pick: fn(Answer) -> Option<T>) -> (Call, Awaited<T>) {
    let (tx, rx) = oneshot::channel();
    (Call { id, reply_to: ReplyTo::Answer(tx) }, Awaited { rx, pick })
}

/// `input`, naming `call`, queued with its reply; refused, the reply is dropped unanswered.
pub fn send(queue: &Tx, input: VoiceEvent, call: Call) -> Result<(), QueueClosed> {
    queue.send(VoiceRuntimeEvent::Awaited(input, call))
}

/// `command` from `origin`, with the reply of the call it names when it is awaited.
pub fn send_command(
    queue: &Tx,
    origin: VoiceCommandOrigin,
    command: Command,
    call: Option<Call>,
) -> Result<(), QueueClosed> {
    let input = VoiceEvent::Host(Sent { origin, command });
    match call {
        Some(call) => send(queue, input, call),
        None => queue.send(input),
    }
}

/// The replies of the calls the loop received and the core has not answered yet: the loop's
/// own, so its end, however it comes, drops them unanswered.
#[derive(Default)]
pub struct Calls(BTreeMap<RequestId, ReplyTo>);

impl Calls {
    /// Kept until the core answers `call`.
    pub fn register(&mut self, call: Call) {
        self.0.insert(call.id, call.reply_to);
    }

    pub fn take(&mut self, id: RequestId) -> Option<ReplyTo> {
        self.0.remove(&id)
    }

    /// Every call still waiting is dropped unanswered.
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

/// Runtime-owned tool bodies remain registered until the host future actually settles.
/// A semantic progress/final notification may release the caller earlier; retaining this entry
/// lets normal cancellation and close cleanup claim the same single cancellation request.
#[derive(Clone, Default)]
pub(crate) struct ToolBodies(Arc<Mutex<BTreeMap<(String, u64), ToolBody>>>);

struct ToolBody {
    call_id: String,
    first_progress: Option<oneshot::Sender<lk_agents_abi::llm::tool_executor::ToolReport>>,
    cancel_requested: bool,
}

impl ToolBodies {
    pub(crate) fn tool_body(
        &self,
        invocation: &lk_agents_abi::agent::Invocation,
    ) -> tokio::sync::oneshot::Receiver<lk_agents_abi::llm::tool_executor::ToolReport> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let key = (invocation.invocation_id.clone(), invocation.operation);
        let call_id =
            invocation.function_call.as_ref().expect("a tool body has a call").call_id.clone();
        let previous = self
            .0
            .lock()
            .insert(key, ToolBody { call_id, first_progress: Some(tx), cancel_requested: false });
        drop(previous);
        rx
    }

    pub(crate) fn end_tool_body(&self, invocation: &lk_agents_abi::agent::Invocation) {
        let pending =
            self.0.lock().remove(&(invocation.invocation_id.clone(), invocation.operation));
        drop(pending);
    }

    /// Claim, then invoke the host after this method releases the lock. Preparation/read-back
    /// callbacks do not have a body entry and retain their existing cancellation path.
    #[must_use]
    pub(crate) fn claim_cancellation(&self, invocation: &lk_agents_abi::agent::Invocation) -> bool {
        let mut bodies = self.0.lock();
        let Some(body) = bodies.get_mut(&(invocation.invocation_id.clone(), invocation.operation))
        else {
            return true;
        };
        !std::mem::replace(&mut body.cancel_requested, true)
    }

    /// Observational: a cancelled body must finish cleanup without starting a fresh host
    /// validation callback under the already-cancelled operation identity.
    pub(crate) fn cancellation_requested(
        &self,
        invocation: &lk_agents_abi::agent::Invocation,
    ) -> bool {
        self.0
            .lock()
            .get(&(invocation.invocation_id.clone(), invocation.operation))
            .is_some_and(|body| body.cancel_requested)
    }

    /// The first report releases only its captured body's waiter. Every later/missing-body
    /// report goes to core with that same identity; core alone decides whether it is current.
    /// Sending reports happens after the lock.
    pub(crate) fn first_tool_progress(
        &self,
        invocation_id: &str,
        operation: u64,
        call_id: &str,
        report: lk_agents_abi::llm::tool_executor::ToolReport,
    ) -> Result<(), lk_agents_abi::llm::tool_executor::ToolReport> {
        let (first, accepted) = {
            let mut bodies = self.0.lock();
            match bodies.get_mut(&(invocation_id.to_owned(), operation)) {
                Some(body) if body.call_id == call_id => (body.first_progress.take(), true),
                _ => (None, false),
            }
        };
        if !accepted {
            return Err(report);
        }
        match first {
            Some(tx) => tx.send(report),
            None => Err(report),
        }
    }
}

#[cfg(test)]
#[path = "calls_tests.rs"]
mod tool_body_tests;
