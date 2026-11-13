use std::time::Duration;

use lk_agents_abi::Ts;
use lk_agents_abi::errors::{ApiError, ErrorKind, ProviderSaid, Timeout};
use lk_agents_abi::utils::socket::WsFrame;

/// What one connection runs under, read when it opens.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ConnectionOptions {
    /// Sent once nothing was sent for this long.
    pub(super) keepalive: Option<(Duration, WsFrame)>,
    /// Sent when the input ended and before a rotation.
    pub(super) goodbye: Option<WsFrame>,
    /// Rotated at the first input boundary after this long.
    pub(super) max_session_duration: Option<Duration>,
    /// How long a drain waits for the provider to finish or close.
    pub(super) drain_limit: Duration,
}

/// What the connection's loops do next, in order.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum SocketEffect {
    /// Send this frame (a keepalive or the goodbye).
    Send(WsFrame),
    /// The connection is done: close it, bounded.
    Finish(Finish),
    /// The connection failed: close it and fail the attempt.
    Fail(ApiError),
}

/// Why a connection finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Finish {
    /// The input ended and the provider finished, or the provider ended the stream.
    Ended,
    /// Rotated: connect again within the same attempt, input held.
    Rotate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Taking input.
    Open,
    /// The goodbye was sent (or none is owed): waiting for the provider to finish or close.
    Draining {
        until: Ts,
        then: Finish,
    },
    Finished,
}

/// The state the loops of one connection share: when it sends its keepalive, rotates, drains
/// and ends.
#[derive(Debug)]
pub(super) struct Connection {
    options: ConnectionOptions,
    phase: Phase,
    last_sent: Ts,
    /// Input went out since the last boundary: a due rotation waits for the next one.
    mid_segment: bool,
    rotate_at: Option<Ts>,
    rotation_due: bool,
}

impl Connection {
    pub(super) fn open(options: ConnectionOptions, now: Ts) -> Self {
        let rotate_at = options.max_session_duration.map(|limit| now.saturating_add(limit));
        Connection {
            options,
            phase: Phase::Open,
            last_sent: now,
            mid_segment: false,
            rotate_at,
            rotation_due: false,
        }
    }

    /// Whether the executor takes the next input item.
    pub(super) fn taking_input(&self) -> bool {
        self.phase == Phase::Open
    }

    /// Whether the connection finished or failed: nothing more is done on it.
    pub(super) fn is_finished(&self) -> bool {
        self.phase == Phase::Finished
    }

    /// An input item's frames went out; `boundary`: it ended a segment.
    pub(super) fn sent(&mut self, now: Ts, boundary: bool, effects: &mut Vec<SocketEffect>) {
        if self.phase != Phase::Open {
            return;
        }
        self.last_sent = now;
        self.mid_segment = !boundary;
        if self.rotation_due && !self.mid_segment {
            self.drain(Finish::Rotate, now, effects);
        }
    }

    /// The last input item went out: the goodbye follows, then the drain.
    pub(super) fn input_ended(&mut self, now: Ts, effects: &mut Vec<SocketEffect>) {
        if self.phase == Phase::Open {
            self.drain(Finish::Ended, now, effects);
        }
    }

    /// `reconnect(reason)`: rotate now.
    pub(super) fn rotate(&mut self, now: Ts, effects: &mut Vec<SocketEffect>) {
        if self.phase == Phase::Open {
            self.drain(Finish::Rotate, now, effects);
        }
    }

    /// The protocol said the provider finished.
    pub(super) fn done(&mut self, effects: &mut Vec<SocketEffect>) {
        let then = match self.phase {
            Phase::Open => Finish::Ended,
            Phase::Draining { then, .. } => then,
            Phase::Finished => return,
        };
        self.finish(SocketEffect::Finish(then), effects);
    }

    /// The peer closed the socket, with its close frame's code and reason.
    pub(super) fn closed(
        &mut self,
        code: Option<u16>,
        reason: Option<String>,
        effects: &mut Vec<SocketEffect>,
    ) {
        let expected = match self.phase {
            Phase::Open => None,
            Phase::Draining { then, .. } => Some(then),
            Phase::Finished => return,
        };
        let effect = match close_outcome(expected.is_some(), code, reason) {
            Ok(()) => SocketEffect::Finish(expected.unwrap_or(Finish::Ended)),
            Err(error) => SocketEffect::Fail(error),
        };
        self.finish(effect, effects);
    }

    /// The time is `now`: a keepalive or a rotation may be due, or a drain may have run out.
    pub(super) fn tick(&mut self, now: Ts, effects: &mut Vec<SocketEffect>) {
        match self.phase {
            Phase::Open => {
                if let Some((interval, frame)) = &self.options.keepalive
                    && now >= self.last_sent.saturating_add(*interval)
                {
                    effects.push(SocketEffect::Send(frame.clone()));
                    self.last_sent = now;
                }
                if !self.rotation_due && self.rotate_at.is_some_and(|at| now >= at) {
                    self.rotation_due = true;
                    if !self.mid_segment {
                        self.drain(Finish::Rotate, now, effects);
                    }
                }
            }
            Phase::Draining { until, then } if now >= until => {
                let effect = match then {
                    // the old connection owes nothing the next one cannot answer
                    Finish::Rotate => SocketEffect::Finish(Finish::Rotate),
                    Finish::Ended => SocketEffect::Fail(ApiError::timed_out(Timeout::Read)),
                };
                self.finish(effect, effects);
            }
            Phase::Draining { .. } | Phase::Finished => {}
        }
    }

    /// When [`tick`](Self::tick) has something to do.
    pub(super) fn wake_at(&self) -> Option<Ts> {
        match self.phase {
            Phase::Open => {
                let keepalive = self
                    .options
                    .keepalive
                    .as_ref()
                    .map(|(interval, _)| self.last_sent.saturating_add(*interval));
                let rotation = self.rotate_at.filter(|_| !self.rotation_due);
                keepalive.into_iter().chain(rotation).min()
            }
            Phase::Draining { until, .. } => Some(until),
            Phase::Finished => None,
        }
    }

    fn drain(&mut self, then: Finish, now: Ts, effects: &mut Vec<SocketEffect>) {
        match (&self.options.goodbye, then) {
            (Some(goodbye), _) => effects.push(SocketEffect::Send(goodbye.clone())),
            // nothing asks the provider to finish what it holds: close at once
            (None, Finish::Rotate) => return self.finish(SocketEffect::Finish(then), effects),
            // the input's end owes the provider's last answer
            (None, Finish::Ended) => {}
        }
        self.phase = Phase::Draining { until: now.saturating_add(self.options.drain_limit), then };
    }

    fn finish(&mut self, effect: SocketEffect, effects: &mut Vec<SocketEffect>) {
        self.phase = Phase::Finished;
        effects.push(effect);
    }
}

/// How the peer's close ends a connection. `expected`: the close was owed (the goodbye was sent,
/// or the input ended), so a close on purpose or with no reason ends it cleanly; a close naming a
/// failure (auth, refused input, a server error) fails it either way.
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
