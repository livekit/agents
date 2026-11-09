
use crate::voice::VoiceRuntimeEvent;
use crate::voice::session::Tx;
use crate::voice::speech_handle::SpeechHandle;
use lk_agents_voice::agent_session::{
    Answer, Command, QueueClosed, RequestId, Sent, VoiceCommandOrigin, VoiceEvent,
};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::oneshot;

/// Where an awaited input's answer goes. It travels with the input that names its call; the
/// runtime keeps it until the core answers that call, and drops it unanswered when the session
/// ends.
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
            ReplyTo::Speech(speech) => speech.0.answered(matches!(answer, Answer::Speech)),
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

/// Lost unless answered first.
pub struct SpeechReply(pub Arc<SpeechHandle>);

impl Drop for SpeechReply {
    fn drop(&mut self) {
        self.0.answered(false);
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

/// The replies of the calls sent to the loop and not answered yet. A sender registers its reply
/// before the input naming the call is queued; the queue owner takes it to answer; the session's
/// end drops the rest unanswered.
#[derive(Default)]
pub struct Calls(Mutex<BTreeMap<RequestId, ReplyTo>>);

impl Calls {
    /// `input`, naming `call`, queued with its reply kept; refused, the reply is dropped.
    pub fn send(&self, queue: &Tx, input: VoiceEvent, call: Call) -> Result<(), QueueClosed> {
        let id = self.register(call);
        queue.send(VoiceRuntimeEvent::Awaited(input, id)).inspect_err(|_| drop(self.take(id)))
    }

    /// Kept until the core answers `call` or the session ends.
    pub fn register(&self, call: Call) -> RequestId {
        self.0.lock().insert(call.id, call.reply_to);
        call.id
    }

    /// `command` from `origin`, with the reply of the call it names when it is awaited.
    pub fn send_command(
        &self,
        queue: &Tx,
        origin: VoiceCommandOrigin,
        command: Command,
        call: Option<Call>,
    ) -> Result<(), QueueClosed> {
        let input = VoiceEvent::Host(Sent { origin, command });
        match call {
            Some(call) => self.send(queue, input, call),
            None => queue.send(input),
        }
    }

    pub fn take(&self, id: RequestId) -> Option<ReplyTo> {
        self.0.lock().remove(&id)
    }

    /// The session ended: every call still waiting is dropped unanswered, outside the lock.
    pub fn clear(&self) {
        drop(std::mem::take(&mut *self.0.lock()));
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
