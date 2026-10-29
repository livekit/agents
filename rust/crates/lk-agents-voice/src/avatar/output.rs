//! Avatar output admission and route retirement. Time, writers, RPC and tasks belong to runtime.

use std::collections::BTreeMap;
use std::time::Duration;

use lk_agents_abi::voice::io::{AvatarPlaybackSegment, PlaybackFinished, PlaybackSegment};

use super::{FinishedPayload, RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputError {
    Closed,
    Full,
    Retired,
    DifferentSegment,
}

/// The agent's answer to an avatar's playback report RPC.
#[derive(Debug, PartialEq)]
pub enum Reply {
    Ok,
    /// The report names a segment that already finished or retired.
    Stale,
    UnexpectedCaller,
    /// The output that registered the route is gone.
    Retired,
    BadPayload(String),
}

impl Reply {
    pub fn wire(self) -> String {
        match self {
            Reply::Ok => "ok".into(),
            Reply::Stale => "reject: stale playback segment".into(),
            Reply::UnexpectedCaller => "reject: unexpected participant".into(),
            Reply::Retired => "reject: retired output".into(),
            Reply::BadPayload(error) => format!("reject: {error}"),
        }
    }
}

/// The report routes each segment stream registers: `lk.playback_started` only when the agent
/// waits for the avatar to start.
pub fn report_routes(wait_playback_start: bool) -> &'static [&'static str] {
    if wait_playback_start {
        &[RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED]
    } else {
        &[RPC_PLAYBACK_FINISHED]
    }
}

/// Only the avatar the output streams to reports its playback.
pub fn check_caller(caller: &str, destination: &str) -> Result<(), Reply> {
    if caller == destination { Ok(()) } else { Err(Reply::UnexpectedCaller) }
}

/// `lk.playback_finished`'s payload as the stream's finish.
fn finished_report(payload: &str) -> Result<PlaybackFinished, Reply> {
    let report = serde_json::from_str::<FinishedPayload>(payload)
        .map_err(|error| Reply::BadPayload(error.to_string()))?;
    let playback_position = Duration::try_from_secs_f64(report.playback_position)
        .map_err(|error| Reply::BadPayload(error.to_string()))?;
    Ok(PlaybackFinished {
        playback_position,
        interrupted: report.interrupted,
        synchronized_transcript: report.synchronized_transcript,
    })
}

/// What an avatar's `lk.playback_finished` does to its segment.
#[derive(Debug, PartialEq)]
pub enum Finish {
    /// The segment is done; its position is what was heard over all its streams.
    Done(PlaybackFinished),
    /// The avatar gave up on a stream the agent is still feeding (its idle timeout): the segment
    /// continues on a new stream; the given-up stream's route retires.
    Continue { route: Option<AvatarPlaybackSegment> },
}

/// One output attachment admits at most two unfinished segments. Resource owners remain outside
/// this state; only correlation keys and the route token cross its synchronous boundary.
#[derive(Debug, Default)]
pub struct OutputState {
    active: Option<PlaybackSegment>,
    last: Option<PlaybackSegment>,
    outstanding: BTreeMap<PlaybackSegment, Option<AvatarPlaybackSegment>>,
    /// What the avatar reported played of a segment's earlier streams.
    carried: BTreeMap<PlaybackSegment, Duration>,
    closed: bool,
    highest_segment: u64,
}

impl OutputState {
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn admit(&mut self, key: PlaybackSegment) -> Result<PlaybackSegment, OutputError> {
        if self.closed {
            return Err(OutputError::Closed);
        }
        if let Some(active) = self.active {
            if active != key {
                return Err(OutputError::DifferentSegment);
            }
            if !self.outstanding.contains_key(&active) {
                return Err(OutputError::Retired);
            }
        } else {
            if key.segment <= self.highest_segment {
                return Err(OutputError::Retired);
            }
            if self.outstanding.len() >= 2 {
                return Err(OutputError::Full);
            }
            self.outstanding.insert(key, None);
            self.highest_segment = key.segment;
            self.active = Some(key);
        }
        self.last = Some(key);
        Ok(key)
    }

    pub fn active(&self) -> Option<PlaybackSegment> {
        self.active
    }

    pub fn last(&self) -> Option<PlaybackSegment> {
        self.last
    }

    /// The segment's audio is complete. `Some(played)` when the avatar already finished all of
    /// it: no stream continued it, so no other report will come.
    pub fn flush(&mut self, key: PlaybackSegment) -> Option<Duration> {
        if self.active == Some(key) {
            self.active = None;
        }
        let reopened = self.outstanding.get(&key).is_some_and(Option::is_some);
        self.carried.get(&key).copied().filter(|_| !reopened)
    }

    /// The avatar's `lk.playback_finished` for `key`'s stream, `payload` as it sent it.
    pub fn report_finished(
        &mut self,
        key: PlaybackSegment,
        payload: &str,
    ) -> Result<Finish, Reply> {
        let report = finished_report(payload)?;
        self.finished(key, report).ok_or(Reply::Stale)
    }

    /// The avatar finished `key`'s stream; `None` for a segment already retired. Only an
    /// uninterrupted finish of the segment still capturing continues it.
    pub fn finished(
        &mut self,
        key: PlaybackSegment,
        mut report: PlaybackFinished,
    ) -> Option<Finish> {
        let route = self.outstanding.get_mut(&key)?;
        let carried = self.carried.get(&key).copied().unwrap_or_default();
        if report.interrupted || self.active != Some(key) {
            report.playback_position += carried;
            return Some(Finish::Done(report));
        }
        self.carried.insert(key, carried + report.playback_position);
        Some(Finish::Continue { route: route.take() })
    }

    /// The segment already played on an earlier stream: a later stream does not start it again.
    pub fn continued(&self, key: PlaybackSegment) -> bool {
        self.carried.contains_key(&key)
    }

    pub fn register(
        &mut self,
        key: PlaybackSegment,
        token: AvatarPlaybackSegment,
    ) -> Result<(), OutputError> {
        if self.closed {
            return Err(OutputError::Closed);
        }
        let stored = self.outstanding.get_mut(&key).ok_or(OutputError::Retired)?;
        *stored = Some(token);
        Ok(())
    }

    pub fn contains(&self, key: PlaybackSegment) -> bool {
        self.outstanding.contains_key(&key)
    }

    pub fn keys(&self) -> Vec<PlaybackSegment> {
        self.outstanding.keys().copied().collect()
    }

    pub fn retire(&mut self, key: PlaybackSegment) -> Option<AvatarPlaybackSegment> {
        if self.last == Some(key) {
            self.last = None;
        }
        self.carried.remove(&key);
        self.outstanding.remove(&key).flatten()
    }

    pub fn close(&mut self) -> Option<PlaybackSegment> {
        self.closed = true;
        self.active.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(segment: u64) -> PlaybackSegment {
        PlaybackSegment { attachment: 1, segment }
    }

    #[test]
    fn two_unsettled_segments_bound_admission_and_retirement_releases_one() {
        let mut state = OutputState::default();
        state.admit(key(1)).unwrap();
        state.flush(key(1));
        state.admit(key(2)).unwrap();
        state.flush(key(2));
        assert_eq!(state.admit(key(3)), Err(OutputError::Full));
        state.retire(key(1));
        assert_eq!(state.admit(key(3)), Ok(key(3)));
    }

    #[test]
    fn repeated_capture_preserves_route_and_a_retired_write_cannot_register() {
        let mut state = OutputState::default();
        state.admit(key(1)).unwrap();
        let token = AvatarPlaybackSegment { playback: key(1), stream_id: Some("stream-1".into()) };
        state.register(key(1), token.clone()).unwrap();
        state.admit(key(1)).unwrap();
        assert_eq!(state.retire(key(1)), Some(token.clone()));
        assert_eq!(state.register(key(1), token), Err(OutputError::Retired));
        assert_eq!(state.admit(key(1)), Err(OutputError::Retired));
        assert_eq!(state.last(), None);
    }

    #[test]
    fn retiring_an_older_route_does_not_cut_the_replacement_segment() {
        let mut state = OutputState::default();
        let old = key(1);
        let new = PlaybackSegment { attachment: 2, segment: 2 };
        state.admit(old).unwrap();
        let old_token = AvatarPlaybackSegment { playback: old, stream_id: Some("REUSED".into()) };
        state.register(old, old_token.clone()).unwrap();
        state.flush(old);
        state.admit(new).unwrap();
        let new_token = AvatarPlaybackSegment { playback: new, stream_id: Some("REUSED".into()) };
        state.register(new, new_token.clone()).unwrap();
        assert_eq!(state.retire(old), Some(old_token));
        assert_eq!(state.last(), Some(new));
        assert!(state.contains(new));
        assert_eq!(state.retire(new), Some(new_token));
    }

    #[test]
    fn an_idle_finish_continues_the_capturing_segment_and_its_final_finish_carries_it() {
        let mut state = OutputState::default();
        let token =
            |id: &str| AvatarPlaybackSegment { playback: key(1), stream_id: Some(id.into()) };
        state.admit(key(1)).unwrap();
        state.register(key(1), token("first")).unwrap();
        let ms = Duration::from_millis;
        let played =
            |at| PlaybackFinished { playback_position: ms(at), ..PlaybackFinished::default() };
        assert_eq!(
            state.finished(key(1), played(10)),
            Some(Finish::Continue { route: Some(token("first")) })
        );
        state.admit(key(1)).expect("the agent keeps feeding the segment");
        state.register(key(1), token("second")).unwrap();
        assert_eq!(state.flush(key(1)), None, "the continuation stream reports its own finish");
        assert_eq!(state.finished(key(1), played(20)), Some(Finish::Done(played(30))));
    }

    #[test]
    fn closed_output_refuses_late_admission_and_route_registration() {
        let mut state = OutputState::default();
        state.admit(key(1)).unwrap();
        assert_eq!(state.close(), Some(key(1)));
        assert_eq!(state.admit(key(2)), Err(OutputError::Closed));
        assert_eq!(
            state.register(
                key(1),
                AvatarPlaybackSegment { playback: key(1), stream_id: Some("late".into()) }
            ),
            Err(OutputError::Closed)
        );
        assert!(state.contains(key(1)), "close retains routes until explicit retirement");
    }
}
