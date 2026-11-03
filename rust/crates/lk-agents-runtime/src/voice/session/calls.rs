//! An awaited input's call: its id goes in the input, its answer comes back to the caller that
//! holds the [`Awaited`] half; a `say()` / `generate_reply()` is answered on its speech handle.
//! [`Calls`] keeps each reply until the core answers its call.

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
mod tool_body_tests {
    use super::ToolBodies;
    use lk_agents_abi::agent::Invocation;
    use lk_agents_abi::llm::tool_executor::{ToolOutcome, ToolReport};

    fn invocation(operation: u64) -> Invocation {
        Invocation {
            invocation_id: "run".into(),
            operation,
            function_call: Some(lk_agents_abi::FunctionCall {
                id: format!("call-{operation}"),
                call_id: "reused-call".into(),
                name: "lookup".into(),
                arguments: "{}".into(),
                created_at: std::time::UNIX_EPOCH,
                extra: Default::default(),
                group_id: None,
            }),
            agent_name: "agent".into(),
            execution_id: None,
            frame_id: None,
            context_scope: None,
            application_context: None,
            history: Vec::new(),
            approval: None,
            speech: None,
            host_context: None,
            span: None,
        }
    }

    #[test]
    fn releasing_first_progress_preserves_single_cancellation_until_body_settlement() {
        let bodies = ToolBodies::default();
        let invocation = invocation(1);
        let mut progress = bodies.tool_body(&invocation);
        let report: ToolReport = ToolOutcome::Nothing.into();
        assert!(
            bodies
                .first_tool_progress(
                    &invocation.invocation_id,
                    invocation.operation,
                    "reused-call",
                    report
                )
                .is_ok()
        );
        assert!(progress.try_recv().is_ok());
        assert!(bodies.claim_cancellation(&invocation));
        assert!(!bodies.claim_cancellation(&invocation), "core and close must not cancel twice");
        bodies.end_tool_body(&invocation);
        assert!(bodies.0.lock().is_empty());
    }

    #[test]
    fn an_older_body_cannot_remove_or_cancel_a_reused_call_ids_new_invocation() {
        let bodies = ToolBodies::default();
        let old = invocation(1);
        let new = invocation(2);
        let _old_progress = bodies.tool_body(&old);
        assert!(
            bodies
                .first_tool_progress(
                    &old.invocation_id,
                    old.operation,
                    "reused-call",
                    ToolOutcome::Nothing.into()
                )
                .is_ok()
        );
        let mut new_progress = bodies.tool_body(&new);
        assert!(bodies.claim_cancellation(&old));
        assert!(
            bodies.claim_cancellation(&new),
            "old cancellation did not cancel the new invocation"
        );
        bodies.end_tool_body(&old);
        assert!(
            !bodies.claim_cancellation(&new),
            "old completion did not remove the new cancellation claim"
        );
        assert!(
            bodies
                .first_tool_progress(
                    &new.invocation_id,
                    new.operation,
                    "reused-call",
                    ToolOutcome::Nothing.into()
                )
                .is_ok()
        );
        assert!(new_progress.try_recv().is_ok());
        bodies.end_tool_body(&new);
        assert!(bodies.0.lock().is_empty());
    }

    #[test]
    fn late_old_progress_cannot_consume_a_replacement_invocations_sender() {
        let bodies = ToolBodies::default();
        let old = invocation(1);
        let new = invocation(2);
        let mut old_progress = bodies.tool_body(&old);
        assert!(
            bodies
                .first_tool_progress(
                    &old.invocation_id,
                    old.operation,
                    "reused-call",
                    ToolOutcome::Nothing.into()
                )
                .is_ok()
        );
        assert!(old_progress.try_recv().is_ok());
        let mut new_progress = bodies.tool_body(&new);
        assert!(
            bodies
                .first_tool_progress(
                    &old.invocation_id,
                    old.operation,
                    "reused-call",
                    ToolOutcome::Update {
                        message: "old cleanup".into(),
                        raw: "old cleanup".into()
                    }
                    .into()
                )
                .is_err(),
            "old progress is routed only to its captured core owner"
        );
        assert!(matches!(
            new_progress.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        assert!(
            bodies
                .first_tool_progress(
                    &new.invocation_id,
                    new.operation,
                    "reused-call",
                    ToolOutcome::Update { message: "new work".into(), raw: "new work".into() }
                        .into()
                )
                .is_ok()
        );
        let report = new_progress.try_recv().expect("the replacement receives only its own update");
        assert!(
            matches!(report.outcome, ToolOutcome::Update { message, .. } if message == "new work")
        );
    }

    #[test]
    fn updates_after_the_first_keep_their_owner_until_the_actual_body_finishes() {
        let bodies = ToolBodies::default();
        let invocation = invocation(1);
        let mut first = bodies.tool_body(&invocation);
        let report = || ToolOutcome::Update { message: "working".into(), raw: "working".into() };
        assert!(
            bodies
                .first_tool_progress(
                    &invocation.invocation_id,
                    invocation.operation,
                    "reused-call",
                    report().into()
                )
                .is_ok()
        );
        assert!(first.try_recv().is_ok());
        // Core decides whether this is an expert's live question or an ordinary terminal.
        assert!(
            bodies
                .first_tool_progress(
                    &invocation.invocation_id,
                    invocation.operation,
                    "reused-call",
                    ToolOutcome::InputRequired {
                        question: "Which city?".into(),
                        verbatim: false,
                        tool: None
                    }
                    .into()
                )
                .is_err()
        );
        assert!(
            bodies
                .first_tool_progress(
                    &invocation.invocation_id,
                    invocation.operation,
                    "reused-call",
                    report().into()
                )
                .is_err()
        );
        assert!(
            bodies
                .first_tool_progress(
                    &invocation.invocation_id,
                    invocation.operation,
                    "reused-call",
                    ToolOutcome::Nothing.into()
                )
                .is_err()
        );
        bodies.end_tool_body(&invocation);
        assert!(
            bodies
                .first_tool_progress(
                    &invocation.invocation_id,
                    invocation.operation,
                    "reused-call",
                    report().into()
                )
                .is_err(),
            "finished-body receipts go only to their captured core owner for rejection"
        );
    }
    #[test]
    fn simultaneous_same_call_ids_route_first_progress_to_the_exact_invocation() {
        let bodies = ToolBodies::default();
        let old = invocation(1);
        let new = invocation(2);
        let mut old_progress = bodies.tool_body(&old);
        let mut new_progress = bodies.tool_body(&new);
        assert!(
            bodies
                .first_tool_progress(
                    &new.invocation_id,
                    new.operation,
                    "reused-call",
                    ToolOutcome::Nothing.into()
                )
                .is_ok()
        );
        assert!(matches!(
            old_progress.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        assert!(new_progress.try_recv().is_ok());
        assert!(
            bodies
                .first_tool_progress(
                    &old.invocation_id,
                    old.operation,
                    "reused-call",
                    ToolOutcome::Nothing.into()
                )
                .is_ok()
        );
        assert!(old_progress.try_recv().is_ok());
    }
}
