//! What a realtime socket's fake server received from the session, and what it refused.

use super::{Observed, Recording};

/// A command the session sent a realtime socket that the AMD rules judge: a response request with
/// its per-response instructions, the session's tools, and the input buffer's commit and clear.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RtSent {
    Response {
        req_id: String,
        instructions: Option<String>,
    },
    Tools(Vec<String>),
    /// The session's own instructions (`UpdateInstructions`).
    Instructions(String),
    CommitAudio,
    ClearAudio,
}

impl RtSent {
    pub fn of(cmd: &lk_agents_abi::RealtimeCommand) -> Option<Self> {
        use lk_agents_abi::RealtimeCommand as C;
        Some(match cmd {
            C::GenerateReply(g) => {
                RtSent::Response { req_id: g.req_id.clone(), instructions: g.instructions.clone() }
            }
            C::UpdateTools(t) => RtSent::Tools(t.tools.iter().map(|t| t.name.clone()).collect()),
            C::UpdateInstructions(i) => RtSent::Instructions(i.instructions.clone()),
            C::CommitAudio(_) => RtSent::CommitAudio,
            C::ClearAudio(_) => RtSent::ClearAudio,
            _ => return None,
        })
    }
}

/// A client frame the realtime server refused, or a request it answered from the wrong state
/// (`fakes::realtime::server`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Breach {
    /// Refused with an `error` event. `knowing`: when the client sent it, it had already been told
    /// what makes it invalid (the response it cancels was done, the one it asks over had been
    /// created, the item it names never was); otherwise it raced an event still on the wire.
    Refused { event: String, code: &'static str, why: &'static str, knowing: bool },
    /// A response asked for while the conversation ended in a function call without its output,
    /// whose output the client sent afterwards: the reply read the conversation without it.
    AskedBeforeOutput { response: String, request: Option<String>, call_id: String },
    /// A client `response.cancel` stopped the server's answer to the person's turn (its VAD ended
    /// the turn and replied), before any of its audio and with no onset told over it: the server
    /// saw no reason for it.
    /// `generation` is the reply as the run knows it (`Kind::Generation`), once the client was
    /// told of it; `None` is a cancel landing on a reply the client never heard of.
    CancelledUnprompted { response: String, generation: Option<u32> },
}

impl Recording {
    /// The session sent the n-th realtime socket a command the AMD rules judge.
    pub fn record_rt_sent(&mut self, at: Observed, socket: u32, sent: RtSent) {
        self.rt_sent.push((at, socket, sent));
    }

    /// What the session sent each realtime socket, in order.
    pub fn rt_sent(&self) -> &[(Observed, u32, RtSent)] {
        &self.rt_sent
    }

    /// The n-th realtime socket's server refused a frame, or answered from the wrong state.
    pub fn record_breach(&mut self, at: Observed, socket: u32, breach: Breach) {
        self.breaches.push((at, socket, breach));
    }
}
