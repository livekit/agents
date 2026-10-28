//! Bounded pre-connect admissions and single-owner PCM transfer.

use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::TrackSid;
use lk_agents_abi::{AudioFrame, Ts};

pub const MAX_TRACKS: usize = 32;
pub const MAX_STREAMS: usize = 4;

pub struct PreConnectBuffer {
    pub received_at: Ts,
    pub frames: Vec<AudioFrame>,
}

enum SlotValue {
    Pending,
    Ready(std::result::Result<PreConnectBuffer, String>),
    Taken,
}

#[derive(Default)]
pub struct PreConnectState {
    tracks: BTreeMap<TrackSid, u64>,
    slots: BTreeMap<u64, SlotValue>,
    transfers: BTreeSet<(TrackSid, String)>,
    receiving: BTreeMap<TrackSid, u64>,
    next_slot: u64,
    closed: bool,
}

impl PreConnectState {
    pub fn slot(&mut self, track: &TrackSid) -> Result<u64> {
        if self.closed {
            return Err(IoError::PreConnectMalformed("pre-connect input is closed".into()));
        }
        if let Some(id) = self.tracks.get(track) {
            return Ok(*id);
        }
        if self.tracks.len() == MAX_TRACKS {
            return Err(IoError::PreConnectMalformed("pre-connect track capacity exceeded".into()));
        }
        Ok(self.new_slot(track.clone()))
    }

    /// `None` means a duplicate stream. Refusal settles an already admitted waiter.
    pub fn receive(&mut self, track: TrackSid, transfer: String) -> Result<Option<u64>> {
        let identity = (track.clone(), transfer);
        if self.closed || self.transfers.contains(&identity) || self.receiving.contains_key(&track)
        {
            return Ok(None);
        }
        let limit = if self.receiving.len() == MAX_STREAMS {
            Some("pre-connect stream capacity exceeded")
        } else if self.transfers.len() == MAX_TRACKS {
            Some("pre-connect transfer capacity exceeded")
        } else {
            None
        };
        if let Some(error) = limit {
            if let Some(id) = self.tracks.get(&track)
                && let Some(slot) = self.slots.get_mut(id)
                && matches!(slot, SlotValue::Pending)
            {
                *slot = SlotValue::Ready(Err(error.into()));
            }
            return Err(IoError::PreConnectMalformed(error.into()));
        }
        let slot = self.slot(&track)?;
        let id = if self.slots.get(&slot).is_some_and(|slot| !matches!(slot, SlotValue::Pending)) {
            self.new_slot(track.clone())
        } else {
            slot
        };
        self.transfers.insert(identity);
        self.receiving.insert(track, id);
        Ok(Some(id))
    }

    pub fn complete(
        &mut self,
        track: &TrackSid,
        id: u64,
        outcome: std::result::Result<PreConnectBuffer, String>,
    ) -> bool {
        if self.receiving.get(track) != Some(&id) {
            return false;
        }
        self.receiving.remove(track);
        let Some(slot) = self.slots.get_mut(&id) else {
            return false;
        };
        if !matches!(slot, SlotValue::Pending) {
            return false;
        }
        *slot = SlotValue::Ready(outcome);
        true
    }

    /// `None` keeps the runtime waiter parked. A successful read moves all PCM out of the state.
    pub fn take(&mut self, id: u64) -> Option<std::result::Result<PreConnectBuffer, String>> {
        let Some(slot) = self.slots.get_mut(&id) else {
            return Some(Err("pre-connect input is closed".into()));
        };
        match slot {
            SlotValue::Pending => None,
            SlotValue::Taken => Some(Err("pre-connect buffer already transferred".into())),
            SlotValue::Ready(_) => {
                let SlotValue::Ready(outcome) = std::mem::replace(slot, SlotValue::Taken) else {
                    unreachable!()
                };
                Some(outcome)
            }
        }
    }

    pub fn expire(&mut self, id: u64) {
        if let Some(slot) = self.slots.get_mut(&id)
            && matches!(slot, SlotValue::Pending)
        {
            *slot = SlotValue::Ready(Err("pre-connect receive deadline expired".into()));
        }
    }

    pub fn close(&mut self) {
        self.closed = true;
        for slot in self.slots.values_mut() {
            if !matches!(slot, SlotValue::Taken) {
                *slot = SlotValue::Ready(Err("pre-connect input is closed".into()));
            }
        }
        self.receiving.clear();
        self.transfers.clear();
    }

    fn new_slot(&mut self, track: TrackSid) -> u64 {
        self.next_slot += 1;
        self.tracks.insert(track, self.next_slot);
        self.slots.insert(self.next_slot, SlotValue::Pending);
        self.next_slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer() -> PreConnectBuffer {
        PreConnectBuffer {
            received_at: Ts::ZERO,
            frames: vec![lk_agents_abi::voice::rtc::silence_frame(480, 24_000, 1)],
        }
    }

    #[test]
    fn preconnect_single_transfer() {
        let mut state = PreConnectState::default();
        let track = TrackSid::from("microphone");
        let waiter = state.slot(&track).unwrap();
        assert!(state.take(waiter).is_none());
        let receive = state.receive(track.clone(), "head".into()).unwrap().unwrap();
        assert_eq!(waiter, receive);
        assert!(state.complete(&track, receive, Ok(buffer())));
        assert_eq!(state.take(waiter).unwrap().unwrap().frames.len(), 1);
        assert!(
            state.take(waiter).unwrap().is_err(),
            "a second waiter must not receive the same PCM"
        );
        assert!(
            state.receive(track, "head".into()).unwrap().is_none(),
            "a transfer cannot run twice"
        );
    }

    #[test]
    fn a_new_transfer_preserves_a_previous_waiters_admitted_head() {
        let mut state = PreConnectState::default();
        let track = TrackSid::from("microphone");
        let first = state.receive(track.clone(), "first".into()).unwrap().unwrap();
        assert!(state.complete(&track, first, Ok(buffer())));
        let second = state.receive(track.clone(), "second".into()).unwrap().unwrap();
        assert_ne!(first, second);
        assert_eq!(state.take(first).unwrap().unwrap().frames.len(), 1);
        assert!(state.take(second).is_none());
        assert!(state.complete(&track, second, Ok(buffer())));
        assert_eq!(state.take(second).unwrap().unwrap().frames.len(), 1);
    }

    #[test]
    fn expired_receive_cannot_install_pcm() {
        let mut state = PreConnectState::default();
        let track = TrackSid::from("microphone");
        let receive = state.receive(track.clone(), "expired".into()).unwrap().unwrap();
        state.expire(receive);
        assert!(!state.complete(&track, receive, Ok(buffer())));
        assert!(state.take(receive).unwrap().is_err());
        let next = state.receive(track.clone(), "next".into()).unwrap().unwrap();
        assert!(state.complete(&track, next, Ok(buffer())));
        state.close();
        assert!(state.take(next).unwrap().is_err(), "close releases unread PCM");
    }

    #[test]
    fn preconnect_overload_preserves_already_admitted_tracks() {
        let mut state = PreConnectState::default();
        let tracks =
            (0..MAX_TRACKS).map(|n| TrackSid::from(format!("track-{n}"))).collect::<Vec<_>>();
        let slots = tracks.iter().map(|track| state.slot(track).unwrap()).collect::<Vec<_>>();
        assert!(state.slot(&"overflow".into()).is_err());
        let id = state.receive(tracks[0].clone(), "admitted".into()).unwrap().unwrap();
        assert!(state.complete(&tracks[0], id, Ok(buffer())));
        assert_eq!(state.take(slots[0]).unwrap().unwrap().frames.len(), 1);
        for (n, track) in tracks.iter().enumerate().skip(1).take(MAX_STREAMS) {
            assert_eq!(
                state.receive(track.clone(), format!("stream-{n}")).unwrap(),
                Some(slots[n])
            );
        }
        assert!(state.receive(tracks[MAX_STREAMS + 1].clone(), "overflow".into()).is_err());
        state.close();
        for slot in &slots[1..] {
            assert!(state.take(*slot).unwrap().is_err());
        }
    }
}

pub const MAX_DURATION: std::time::Duration = std::time::Duration::from_secs(10);
pub const MAX_BYTES: usize = 2 * 1024 * 1024;

pub fn source_format(
    info: &lk_agents_abi::voice::rtc::StreamInfo,
) -> std::result::Result<(u32, u16), String> {
    let rate = info
        .attributes
        .get("sampleRate")
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|rate| (8000..=48000).contains(rate))
        .ok_or("pre-connect sampleRate must be between 8000 and 48000")?;
    let channels = info
        .attributes
        .get("channels")
        .and_then(|v| v.parse::<u16>().ok())
        .filter(|channels| (1..=2).contains(channels))
        .ok_or("pre-connect channels must be 1 or 2")?;
    Ok((rate, channels))
}
