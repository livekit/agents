//! An awaited input's call: its id goes in the input, its answer comes back to the caller that
//! holds the [`Awaited`] half; a `say()` / `generate_reply()` is answered on its speech handle.
//! [`Calls`] keeps each reply until the core answers its call.

use crate::voice::session::Tx;
use lk_agents_voice::agent_session::{Answer, Command, Event, Input, Origin, QueueClosed, RequestId, Sent};
use crate::voice::speech_handle::SpeechHandle;
use std::collections::BTreeMap;
use parking_lot::Mutex;
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
    pub fn send(&self, queue: &Tx, input: Input, call: Call) -> Result<(), QueueClosed> {
        let id = self.register(call);
        queue.send(Event::Awaited(input, id)).inspect_err(|_| drop(self.take(id)))
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
        origin: Origin,
        command: Command,
        call: Option<Call>,
    ) -> Result<(), QueueClosed> {
        let input = Input::Host(Sent { origin, command });
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
