//! An awaited input's call: its id goes in the input, its answer comes back to the caller that
//! holds the [`Awaited`] half; a `say()` / `generate_reply()` is answered on its speech handle.

use crate::agent_session::commands::{Answer, RequestId};
use crate::speech_handle::SpeechHandle;
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

impl RequestId {
    /// The calls a session's own tasks await take ids with this bit set, apart from the host's.
    const TASK: u64 = 1 << 63;

    /// The id of the `n`th call the session's tasks await.
    pub(crate) fn task(n: u64) -> Self {
        RequestId(n | Self::TASK)
    }
}
