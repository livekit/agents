use super::{Observed, Recording};

/// One fact about delegation to the remote A2A peer: what the peer received, asked and did, and
/// what each remote tool body handed the session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoteStep {
    /// The peer received a business message. `answering` is the question it was waiting on, by
    /// message id, when the message reached a task visibly waiting for input.
    Received { task: String, message: String, answering: Option<String>, text: String },
    /// The peer asked `text` on `task`, as its message `question`.
    Asked { task: String, question: String, text: String },
    /// The peer committed `task`'s result.
    Finished { task: String, result: String },
    /// The peer applied `message`, then its acknowledgement was lost.
    AckLost { task: String, message: String },
    /// The peer received a `CancelTask` for `task`.
    CancelReceived { task: String },
    StreamOpened { stream: u32 },
    StreamClosed { stream: u32 },
    /// A remote tool body started for the call item `call`.
    BodyStarted { call: String },
    /// The body's delegation handed the session a question (`DelegationHost::question`).
    Presented { call: String, text: String },
    /// The body returned this outcome to the session; `None` when it was cancelled first.
    Returned { call: String, outcome: Option<String> },
    /// The body's delegation finished its local cleanup.
    BodySettled { call: String },
}

impl Recording {
    pub fn record_remote(&mut self, at: Observed, step: RemoteStep) {
        self.remote.push((at, step));
    }

    /// Every remote fact, in order.
    pub fn remote(&self) -> &[(Observed, RemoteStep)] {
        &self.remote
    }
}
