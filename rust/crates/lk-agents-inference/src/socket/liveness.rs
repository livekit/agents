use std::time::Duration;

use lk_agents_abi::Ts;

/// Whether a socket's peer still answers: a ping goes out every `interval`, and the connection
/// fails when nothing, not even the pong, came back within half an interval of one. Anything the
/// peer sends counts as an answer.
#[derive(Debug)]
pub(super) struct Liveness {
    interval: Duration,
    next_ping: Ts,
    /// The answer owed to the last ping, by then.
    owed: Option<Ts>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Beat {
    Wait,
    Ping,
    /// A ping went unanswered.
    Lost,
}

impl Liveness {
    pub(super) fn new(interval: Duration, now: Ts) -> Self {
        Liveness { interval, next_ping: now.saturating_add(interval), owed: None }
    }

    /// The peer sent something.
    pub(super) fn heard(&mut self) {
        self.owed = None;
    }

    pub(super) fn wake_at(&self) -> Ts {
        self.owed.map_or(self.next_ping, |owed| owed.min(self.next_ping))
    }

    pub(super) fn tick(&mut self, now: Ts) -> Beat {
        if self.owed.is_some_and(|owed| now >= owed) {
            return Beat::Lost;
        }
        if now < self.next_ping {
            return Beat::Wait;
        }
        self.next_ping = now.saturating_add(self.interval);
        self.owed.get_or_insert(now.saturating_add(self.interval / 2));
        Beat::Ping
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(seconds: u64) -> Ts {
        Ts::from_nanos(seconds * 1_000_000_000)
    }

    // Catches: a half-open socket never noticed, or a peer that answers each ping failed anyway.
    #[test]
    fn an_unanswered_ping_is_lost_and_an_answered_one_is_not() {
        let mut liveness = Liveness::new(Duration::from_secs(30), at(0));
        assert_eq!(liveness.tick(at(29)), Beat::Wait);
        assert_eq!(liveness.tick(at(30)), Beat::Ping);
        liveness.heard();
        assert_eq!(liveness.tick(at(45)), Beat::Wait);
        assert_eq!(liveness.tick(at(60)), Beat::Ping);
        assert_eq!(liveness.wake_at(), at(75));
        assert_eq!(liveness.tick(at(75)), Beat::Lost);
    }
}
