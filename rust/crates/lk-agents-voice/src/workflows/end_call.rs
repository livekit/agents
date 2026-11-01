//! `EndCallTool`: the session closes once the speech that called `end_call` is done, so the
//! goodbye the tool asked for is heard. A realtime model that answers tool results by itself
//! speaks that answer in a speech of its own: the session waits for it to start (five seconds at
//! most) and closes once it is done.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::Driver;
use crate::agent_session::commands::Close;
use crate::agent_session::effects::Out;
use crate::events::SessionEvent;
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::CloseReason;
use std::time::Duration;

/// How long a realtime model's own reply to the tool result has to start.
pub const REPLY_WAIT: Duration = Duration::from_secs(5);

/// Where an ended call is in its goodbye.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EndCall {
    /// The speech whose tool ended the call is still playing.
    AfterSpeech(SpeechId),
    /// It is done; the realtime model's reply has not started yet.
    AwaitingReply,
    /// The realtime model's reply is playing.
    AfterReply(SpeechId),
}

impl Driver {
    /// `end_call` from the tool running in `speech`.
    pub(crate) fn end_call(&mut self, speech: SpeechId) {
        self.activity.request_end_call(speech);
    }

    /// The speeches an ended call waits on, off the events on their way to the host.
    pub(crate) fn end_call_on_event(&mut self, now: Ts, ev: &SessionEvent, out: &mut Out) {
        if self.activity.end_call_event(now, ev, out) {
            self.close_ended_call(now, out);
        }
    }

    /// The realtime model's reply never started.
    pub(crate) fn end_call_reply_timed_out(&mut self, now: Ts, out: &mut Out) {
        if self.activity.end_call_timed_out() {
            log::warn!(target: "livekit.agents", "tool reply timed out, shutting down session");
            self.close_ended_call(now, out);
        }
    }

    /// `session.shutdown()`: closed after what is left to say.
    fn close_ended_call(&mut self, now: Ts, out: &mut Out) {
        let close = Close { reason: CloseReason::UserInitiated, drain: true, error: None };
        self.begin_closing(now, close, out);
    }
}
