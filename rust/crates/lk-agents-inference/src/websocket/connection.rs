use std::time::Duration;

use lk_agents_abi::Ts;
use lk_agents_abi::errors::{ApiError, ErrorKind, ProviderSaid, Timeout};
use lk_agents_abi::utils::websocket::{WebSocketSignal, WsFrame};

/// What one connection runs under, read when it opens.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ConnectionOptions {
    /// The protocol's keepalive frame, sent once nothing was sent for this long.
    pub(super) keepalive: Option<(Duration, WsFrame)>,
    /// Replaced at the stream's first moment it can be after this long.
    pub(super) max_session_duration: Option<Duration>,
    /// How long the wait for the provider's reply after the input's end, or for its close after
    /// a failed write, lasts.
    pub(super) reply_limit: Duration,
    /// Pinged this often, the socket's own liveness.
    pub(super) ping: Option<Duration>,
    /// Whether input waits for the provider to say it is ready.
    pub(super) readiness: Readiness,
    /// The stream's input-idle frames go once no input went out for this long.
    pub(super) input_idle: Option<Duration>,
    /// Response silence, measured after both the latest input and received response data.
    pub(super) response_idle: Option<Duration>,
    /// The provider session's own end, sent when a replacement is due and when the input ends,
    /// then drained for up to `reply_limit`. Empty: the connection is left at once.
    pub(super) goodbye: Vec<WsFrame>,
}

/// When a connection takes input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Readiness {
    /// As soon as it opens.
    Open,
    /// Once the provider signalled `ready`, by `by`.
    Awaited { by: Ts },
}

/// What the connection's loops do next, in order.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum WebSocketEffect {
    /// Send this frame (a keepalive).
    Send(WsFrame),
    /// No input went out for `input_idle`: send the stream's input-idle frames.
    InputIdle,
    ResponseIdle,
    /// The connection is done: close it, bounded.
    Finish(Finish),
    /// The connection failed: close it and fail the attempt.
    Fail(ApiError),
}

/// Why a connection finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Finish {
    /// The input ended and the provider owes nothing more.
    Ended,
    /// Replaced: connect again at once within the same attempt.
    Replaced,
    /// The provider finished its session: the next connection opens for the next input.
    SessionFinished,
}

#[derive(Debug, Clone, PartialEq)]
enum Phase {
    /// Waiting for the provider to signal `ready`, until the deadline.
    Held {
        until: Ts,
    },
    /// Taking input.
    Open,
    /// The input ended: waiting for the provider's reply or close, until the deadline.
    AwaitingReply {
        until: Ts,
    },
    /// The goodbye went: the provider's session finished or its clean close, or `until`, finishes
    /// the connection with `then` (replaced at a due replacement, ended at the input's end).
    Draining {
        until: Ts,
        then: Finish,
    },
    /// A write failed: the peer's close says why, by `until`, else `error` fails the connection.
    /// `awaiting_reply`: the input had ended, so a clean close ends the stream.
    WriteFailed {
        until: Ts,
        error: ApiError,
        awaiting_reply: bool,
    },
    Finished,
}

/// The state the loops of one connection share: when it sends its keepalive, is replaced,
/// waits for the provider and ends.
#[derive(Debug)]
pub(super) struct Connection {
    options: ConnectionOptions,
    phase: Phase,
    last_sent: Ts,
    /// Input went out on this connection: the input's end may owe a reply.
    input_sent: bool,
    /// When input last went out, and whether the input-idle frames went since.
    last_input: Ts,
    input_idle_sent: bool,
    last_response: Option<Ts>,
    response_idle_sent: bool,
    /// Items between their first frame and their last: a replacement waits for them.
    writing: usize,
    /// A replacement asked for while an item was being written runs once it is.
    replace_after_write: bool,
    replace_at: Option<Ts>,
    replacement_due: bool,
}

impl Connection {
    pub(super) fn open(options: ConnectionOptions, now: Ts) -> Self {
        let replace_at = options.max_session_duration.map(|limit| now.saturating_add(limit));
        let phase = match options.readiness {
            Readiness::Open => Phase::Open,
            Readiness::Awaited { by } => Phase::Held { until: by },
        };
        Connection {
            options,
            phase,
            last_sent: now,
            input_sent: false,
            last_input: now,
            input_idle_sent: false,
            last_response: None,
            response_idle_sent: false,
            writing: 0,
            replace_after_write: false,
            replace_at,
            replacement_due: false,
        }
    }

    /// Whether the executor takes the next input item or control message.
    pub(super) fn taking_input(&self) -> bool {
        self.phase == Phase::Open
    }

    /// Whether the connection finished or failed: nothing more is done on it.
    pub(super) fn is_finished(&self) -> bool {
        self.phase == Phase::Finished
    }

    /// An input item's first frame is about to go.
    pub(super) fn writing_input(&mut self, now: Ts) {
        self.writing += 1;
        self.input_sent = true;
        self.last_input = now;
        self.input_idle_sent = false;
        self.response_idle_sent = false;
    }

    /// The input item's last frame went.
    pub(super) fn input_written(&mut self, now: Ts, effects: &mut Vec<WebSocketEffect>) {
        self.writing = self.writing.saturating_sub(1);
        self.last_sent = now;
        if self.writing == 0
            && self.replace_after_write
            && matches!(self.phase, Phase::Open | Phase::Held { .. })
        {
            self.finish(WebSocketEffect::Finish(Finish::Replaced), effects);
        }
    }

    /// Frames that are not input (a control message, the input-idle frames) went.
    pub(super) fn wrote(&mut self, now: Ts) {
        self.last_sent = now;
    }

    pub(super) fn received_response(&mut self, now: Ts) {
        self.last_response = Some(now);
        self.response_idle_sent = false;
    }

    fn response_idle_at(&self) -> Option<Ts> {
        if self.response_idle_sent {
            return None;
        }
        let idle = self.options.response_idle?;
        Some(self.last_response?.max(self.last_input).saturating_add(idle))
    }

    /// A write failed, the socket's peer likely gone: its close decides, within the reply limit.
    pub(super) fn write_failed(&mut self, now: Ts, error: ApiError) {
        let awaiting_reply = match &self.phase {
            // a draining connection is replaced by its own limit, whatever the peer says
            Phase::Finished | Phase::WriteFailed { .. } | Phase::Draining { .. } => return,
            Phase::AwaitingReply { .. } => true,
            Phase::Held { .. } | Phase::Open => false,
        };
        let until = now.saturating_add(self.options.reply_limit);
        self.phase = Phase::WriteFailed { until, error, awaiting_reply };
    }

    /// The input ended, its end item sent. A provider session with a goodbye is sent it and
    /// drained, whatever input went: its answer is owed. Else `owes_reply`: the stream owes the
    /// provider's reply to what it sent, so a connection that sent input waits for it; else the
    /// stream ends now.
    pub(super) fn input_ended(
        &mut self,
        now: Ts,
        owes_reply: bool,
        effects: &mut Vec<WebSocketEffect>,
    ) {
        if self.phase != Phase::Open {
            return;
        }
        if !self.options.goodbye.is_empty() {
            self.say_goodbye(now, Finish::Ended, effects);
        } else if self.input_sent && owes_reply {
            let until = now.saturating_add(self.options.reply_limit);
            self.phase = Phase::AwaitingReply { until };
        } else {
            self.finish(WebSocketEffect::Finish(Finish::Ended), effects);
        }
    }

    /// `reconnect(reason)`: replace the connection now, once the item being written went.
    pub(super) fn replace(&mut self, effects: &mut Vec<WebSocketEffect>) {
        if !matches!(self.phase, Phase::Open | Phase::Held { .. }) {
            return;
        }
        if self.writing > 0 {
            self.replace_after_write = true;
        } else {
            self.finish(WebSocketEffect::Finish(Finish::Replaced), effects);
        }
    }

    /// `end_session()`: the stream read the provider's session as finished.
    pub(super) fn end_session(&mut self, effects: &mut Vec<WebSocketEffect>) {
        if !self.is_finished() {
            self.finish(WebSocketEffect::Finish(Finish::SessionFinished), effects);
        }
    }

    /// Whether `max_session_duration` passed and nothing is being written: the stream is asked
    /// whether it can be replaced now.
    pub(super) fn replacement_due(&self) -> bool {
        self.replacement_due && self.writing == 0 && self.phase == Phase::Open
    }

    /// The due replacement the stream allows: the goodbye goes first, when the provider session
    /// has one, and the connection drains it for up to the reply limit; else it is replaced now.
    pub(super) fn replace_due(&mut self, now: Ts, effects: &mut Vec<WebSocketEffect>) {
        if !self.replacement_due() {
            return;
        }
        if self.options.goodbye.is_empty() {
            self.finish(WebSocketEffect::Finish(Finish::Replaced), effects);
            return;
        }
        self.say_goodbye(now, Finish::Replaced, effects);
    }

    /// Sends the goodbye and drains the provider session it ends, then finishes with `then`.
    fn say_goodbye(&mut self, now: Ts, then: Finish, effects: &mut Vec<WebSocketEffect>) {
        self.phase = Phase::Draining { until: now.saturating_add(self.options.reply_limit), then };
        effects.extend(self.options.goodbye.iter().cloned().map(WebSocketEffect::Send));
    }

    /// What a received message signalled.
    pub(super) fn signal(
        &mut self,
        signal: WebSocketSignal,
        now: Ts,
        effects: &mut Vec<WebSocketEffect>,
    ) {
        match signal {
            WebSocketSignal::Ready => {
                if matches!(self.phase, Phase::Held { .. }) {
                    self.phase = Phase::Open;
                    self.last_sent = now;
                    self.last_input = now;
                }
            }
            WebSocketSignal::SessionFinished => {
                let then = match &self.phase {
                    // the input ended and the provider finished: the stream ends
                    Phase::AwaitingReply { .. }
                    | Phase::WriteFailed { awaiting_reply: true, .. } => Finish::Ended,
                    // the provider answered the goodbye
                    Phase::Draining { then, .. } => *then,
                    Phase::Finished => return,
                    _ => Finish::SessionFinished,
                };
                self.finish(WebSocketEffect::Finish(then), effects);
            }
        }
    }

    /// The peer closed the socket, with its close frame's code and reason. `expected`: the stream
    /// asked the provider to end its session. `nothing_pending`: nothing the provider was sent is
    /// unanswered, so a clean close after the input's end ends the stream, and one before it
    /// finishes the session (the next input goes on a new connection); with words still owed an
    /// unexpected close fails the connection.
    pub(super) fn closed(
        &mut self,
        code: Option<u16>,
        reason: Option<String>,
        expected: bool,
        nothing_pending: bool,
        effects: &mut Vec<WebSocketEffect>,
    ) {
        let then = match &self.phase {
            Phase::AwaitingReply { .. } | Phase::WriteFailed { awaiting_reply: true, .. } => {
                nothing_pending.then_some(Finish::Ended)
            }
            Phase::Finished => return,
            // the provider left the session the goodbye ended: a clean close finishes it
            Phase::Draining { then, .. } => Some(*then),
            // a replacement waiting for the item being written: the item goes on the next one
            Phase::Open | Phase::Held { .. } | Phase::WriteFailed { .. }
                if self.replace_after_write =>
            {
                Some(Finish::Replaced)
            }
            Phase::Open | Phase::Held { .. } | Phase::WriteFailed { .. } => {
                (expected || nothing_pending).then_some(Finish::SessionFinished)
            }
        };
        let effect = match close_outcome(then.is_some(), code, reason) {
            Ok(()) => WebSocketEffect::Finish(then.unwrap_or(Finish::Ended)),
            Err(error) => WebSocketEffect::Fail(error),
        };
        self.finish(effect, effects);
    }

    /// Whether the wait for the provider's reply runs out at `now`: its outcome then depends on
    /// whether anything is left unanswered (`tick`'s `nothing_pending`).
    pub(super) fn reply_wait_runs_out(&self, now: Ts) -> bool {
        matches!(self.phase, Phase::AwaitingReply { until } if now >= until)
    }

    /// The time is `now`: a keepalive or the input-idle frames may be due, or a wait may have run
    /// out. `nothing_pending`: nothing the provider was sent is unanswered, so a reply wait that
    /// ran out ends cleanly.
    pub(super) fn tick(
        &mut self,
        now: Ts,
        nothing_pending: bool,
        effects: &mut Vec<WebSocketEffect>,
    ) {
        match &self.phase {
            Phase::Held { until } if now >= *until => {
                // the provider never said it is ready
                let error = ApiError::timed_out(Timeout::Connect);
                self.finish(WebSocketEffect::Fail(error), effects);
            }
            Phase::Open => {
                if let Some((interval, frame)) = &self.options.keepalive
                    && now >= self.last_sent.saturating_add(*interval)
                {
                    effects.push(WebSocketEffect::Send(frame.clone()));
                    self.last_sent = now;
                }
                if let Some(idle) = self.options.input_idle
                    && !self.input_idle_sent
                    && now >= self.last_input.saturating_add(idle)
                {
                    effects.push(WebSocketEffect::InputIdle);
                    self.input_idle_sent = true;
                }
                if self.response_idle_at().is_some_and(|at| now >= at) {
                    effects.push(WebSocketEffect::ResponseIdle);
                    self.response_idle_sent = true;
                }
                if self.replace_at.is_some_and(|at| now >= at) {
                    self.replacement_due = true;
                }
            }
            Phase::AwaitingReply { until } if now >= *until => {
                let effect = if nothing_pending {
                    WebSocketEffect::Finish(Finish::Ended)
                } else {
                    WebSocketEffect::Fail(ApiError::timed_out(Timeout::Read))
                };
                self.finish(effect, effects);
            }
            Phase::Draining { until, then } if now >= *until => {
                // the provider never answered the goodbye: finished all the same
                let then = *then;
                self.finish(WebSocketEffect::Finish(then), effects);
            }
            Phase::WriteFailed { until, error, .. } if now >= *until => {
                // the peer never said why: the write's own failure
                let error = error.clone();
                self.finish(WebSocketEffect::Fail(error), effects);
            }
            Phase::Held { .. }
            | Phase::AwaitingReply { .. }
            | Phase::Draining { .. }
            | Phase::WriteFailed { .. }
            | Phase::Finished => {}
        }
    }

    /// When [`tick`](Self::tick) has something to do.
    pub(super) fn wake_at(&self) -> Option<Ts> {
        match &self.phase {
            Phase::Open => {
                let keepalive = self
                    .options
                    .keepalive
                    .as_ref()
                    .map(|(interval, _)| self.last_sent.saturating_add(*interval));
                let idle = self
                    .options
                    .input_idle
                    .filter(|_| !self.input_idle_sent)
                    .map(|idle| self.last_input.saturating_add(idle));
                let replacement = self.replace_at.filter(|_| !self.replacement_due);
                keepalive
                    .into_iter()
                    .chain(idle)
                    .chain(self.response_idle_at())
                    .chain(replacement)
                    .min()
            }
            Phase::Held { until }
            | Phase::AwaitingReply { until }
            | Phase::Draining { until, .. }
            | Phase::WriteFailed { until, .. } => Some(*until),
            Phase::Finished => None,
        }
    }

    fn finish(&mut self, effect: WebSocketEffect, effects: &mut Vec<WebSocketEffect>) {
        self.phase = Phase::Finished;
        effects.push(effect);
    }
}

/// How the peer's close ends a connection. `expected`: the close was owed (the input ended, or
/// the stream asked the provider to end its session), so a close on purpose or with no reason
/// ends it cleanly; a close naming a failure (auth, refused input, a server error) fails it either
/// way.
fn close_outcome(
    expected: bool,
    code: Option<u16>,
    reason: Option<String>,
) -> Result<(), ApiError> {
    let kind = ErrorKind::for_close(code);
    if expected && matches!(kind, None | Some(ErrorKind::Connection)) {
        return Ok(());
    }
    let message = match code {
        Some(code) => format!("the provider closed the connection (close code {code})"),
        None => "the provider closed the connection".to_owned(),
    };
    let mut error =
        ApiError::connection(Some(message)).with_kind(kind.unwrap_or(ErrorKind::Connection));
    error.said = ProviderSaid::boxed(reason, None);
    Err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_silence_waits_for_data_and_restarts_after_input_or_response() {
        let at = |ms: u64| Ts::from_nanos(ms * 1_000_000);
        let mut connection = Connection::open(
            ConnectionOptions {
                keepalive: None,
                max_session_duration: None,
                reply_limit: Duration::from_secs(10),
                ping: None,
                readiness: Readiness::Open,
                input_idle: None,
                response_idle: Some(Duration::from_millis(600)),
                goodbye: Vec::new(),
            },
            at(0),
        );
        let mut effects = Vec::new();
        connection.tick(at(1000), false, &mut effects);
        assert!(effects.is_empty(), "slow first audio does not complete the response");
        connection.received_response(at(1000));
        connection.received_response(at(1400));
        connection.tick(at(1800), false, &mut effects);
        assert!(effects.is_empty(), "a later response restarts silence");
        connection.writing_input(at(1900));
        connection.input_written(at(1900), &mut effects);
        connection.tick(at(2000), false, &mut effects);
        assert!(effects.is_empty(), "new text must have time to produce audio");
        connection.tick(at(2500), false, &mut effects);
        assert_eq!(effects, [WebSocketEffect::ResponseIdle]);
        effects.clear();
        connection.tick(at(3000), false, &mut effects);
        assert!(effects.is_empty(), "once per silence");
        connection.received_response(at(3100));
        connection.tick(at(3700), false, &mut effects);
        assert_eq!(effects, [WebSocketEffect::ResponseIdle]);
    }
}
