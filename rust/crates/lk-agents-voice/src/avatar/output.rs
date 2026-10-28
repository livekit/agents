//! Avatar output admission and route retirement. Time, writers, RPC and tasks belong to runtime.

use std::collections::BTreeMap;

use lk_agents_abi::voice::io::{AvatarPlaybackSegment, PlaybackSegment};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputError {
    Closed,
    Full,
    Retired,
    DifferentSegment,
}

/// One output attachment admits at most two unfinished segments. Resource owners remain outside
/// this state; only correlation keys and the route token cross its synchronous boundary.
#[derive(Debug, Default)]
pub struct OutputState {
    active: Option<PlaybackSegment>,
    last: Option<PlaybackSegment>,
    outstanding: BTreeMap<PlaybackSegment, Option<AvatarPlaybackSegment>>,
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

    pub fn flush(&mut self, key: PlaybackSegment) {
        if self.active == Some(key) {
            self.active = None;
        }
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
