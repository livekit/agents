//! Bounded pre-connect admissions and single-owner PCM transfer.

use std::collections::{BTreeMap, BTreeSet};

use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::TrackSid;
use lk_agents_abi::{AudioFrame, Ts};

pub const MAX_TRACKS: usize = 32;
pub const MAX_STREAMS: usize = 4;

/// Why a buffered head is not replayed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HeadError {
    #[error("pre-connect input is closed")]
    Closed,
    #[error("pre-connect buffer already transferred")]
    Transferred,
    #[error("pre-connect receive deadline expired")]
    Expired,
    #[error("pre-connect stream cleanup did not finish")]
    Cleanup,
    #[error("pre-connect {0} capacity exceeded")]
    Capacity(&'static str),
    #[error("pre-connect {0}")]
    Malformed(&'static str),
    #[error("pre-connect stream failed: {0}")]
    Stream(String),
    #[error("pre-connect decode failed: {0}")]
    Decode(String),
}

impl From<lk_agents_media::CodecError> for HeadError {
    fn from(error: lk_agents_media::CodecError) -> Self {
        HeadError::Decode(error.to_string())
    }
}

impl From<HeadError> for IoError {
    fn from(error: HeadError) -> Self {
        IoError::PreConnectMalformed(error.to_string())
    }
}

pub struct PreConnectBuffer {
    pub received_at: Ts,
    pub frames: Vec<AudioFrame>,
}

enum SlotValue {
    Pending,
    Ready(std::result::Result<PreConnectBuffer, HeadError>),
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
            return Err(HeadError::Closed.into());
        }
        if let Some(id) = self.tracks.get(track) {
            return Ok(*id);
        }
        if self.tracks.len() == MAX_TRACKS {
            return Err(HeadError::Capacity("track").into());
        }
        Ok(self.new_slot(track.clone()))
    }

    /// A stream names its track in its `trackId` attribute; one that names none is not
    /// admitted (`None`). Otherwise the track and [`Self::receive`]'s outcome.
    pub fn receive_stream(
        &mut self,
        attributes: &BTreeMap<String, String>,
        transfer: String,
    ) -> Option<(TrackSid, Result<Option<u64>>)> {
        let track = TrackSid::from(attributes.get("trackId")?.clone());
        let outcome = self.receive(track.clone(), transfer);
        Some((track, outcome))
    }

    /// `None` means a duplicate stream. Refusal settles an already admitted waiter.
    pub fn receive(&mut self, track: TrackSid, transfer: String) -> Result<Option<u64>> {
        let identity = (track.clone(), transfer);
        if self.closed || self.transfers.contains(&identity) || self.receiving.contains_key(&track)
        {
            return Ok(None);
        }
        let limit = if self.receiving.len() == MAX_STREAMS {
            Some(HeadError::Capacity("stream"))
        } else if self.transfers.len() == MAX_TRACKS {
            Some(HeadError::Capacity("transfer"))
        } else {
            None
        };
        if let Some(error) = limit {
            if let Some(id) = self.tracks.get(&track)
                && let Some(slot) = self.slots.get_mut(id)
                && matches!(slot, SlotValue::Pending)
            {
                *slot = SlotValue::Ready(Err(error.clone()));
            }
            return Err(error.into());
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
        outcome: std::result::Result<PreConnectBuffer, HeadError>,
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
    pub fn take(&mut self, id: u64) -> Option<std::result::Result<PreConnectBuffer, HeadError>> {
        let Some(slot) = self.slots.get_mut(&id) else {
            return Some(Err(HeadError::Closed));
        };
        match slot {
            SlotValue::Pending => None,
            SlotValue::Taken => Some(Err(HeadError::Transferred)),
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
            *slot = SlotValue::Ready(Err(HeadError::Expired));
        }
    }

    pub fn close(&mut self) {
        self.closed = true;
        for slot in self.slots.values_mut() {
            if !matches!(slot, SlotValue::Taken) {
                *slot = SlotValue::Ready(Err(HeadError::Closed));
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
    fn a_head_decodes_to_the_session_format_in_ten_millisecond_frames() {
        let info = lk_agents_abi::voice::rtc::StreamInfo {
            attributes: [("sampleRate".into(), "48000".into()), ("channels".into(), "2".into())]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let mut head = HeadBytes::new(&info).unwrap();
        head.push(&[100, 0, 200, 0].repeat(960)).unwrap();
        let frames = head.decode((24_000, 1)).unwrap();
        assert!(frames.iter().all(|f| (f.sample_rate, f.num_channels) == (24_000, 1)));
        assert_eq!(frames.iter().map(|f| f.samples_per_channel).sum::<u32>(), 480);
        assert!(frames.iter().all(|f| f.samples_per_channel <= 240));
    }

    #[test]
    fn normalization_preserves_the_heads_duration() {
        for (source, target) in [(24_000u32, 24_000u32), (48_000, 24_000), (24_000, 48_000)] {
            let info = lk_agents_abi::voice::rtc::StreamInfo {
                attributes: [
                    ("sampleRate".into(), source.to_string()),
                    ("channels".into(), "2".into()),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            };
            let mut head = HeadBytes::new(&info).unwrap();
            head.push(&lk_agents_media::pcm_bytes(&[100, 200].repeat(source as usize / 10)))
                .unwrap();
            let frames = head.decode((target, 1)).unwrap();
            let samples: u32 = frames.iter().map(|f| f.samples_per_channel).sum();
            assert_eq!(samples, target / 10, "{source} -> {target}");
        }
    }

    #[test]
    fn a_raw_head_longer_than_its_duration_bound_is_refused() {
        let info = lk_agents_abi::voice::rtc::StreamInfo {
            attributes: [("sampleRate".into(), "8000".into()), ("channels".into(), "1".into())]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let mut head = HeadBytes::new(&info).unwrap();
        let ten_seconds = vec![0; 8_000 * 2 * 10];
        head.push(&ten_seconds).unwrap();
        assert!(head.push(&[0, 0]).is_err());
    }

    #[test]
    fn a_container_head_stops_at_its_decoded_duration_bound() {
        let info = lk_agents_abi::voice::rtc::StreamInfo {
            mime_type: "audio/wav".into(),
            attributes: [("sampleRate".into(), "8000".into()), ("channels".into(), "1".into())]
                .into_iter()
                .collect(),
            ..Default::default()
        };
        let eleven_seconds = vec![0i16; 8_000 * 11];
        let data = (eleven_seconds.len() * 2) as u32;
        let mut wav = [b"RIFF".as_slice(), &(36 + data).to_le_bytes(), b"WAVEfmt "].concat();
        wav.extend(16u32.to_le_bytes());
        wav.extend(1u16.to_le_bytes()); // PCM
        wav.extend(1u16.to_le_bytes()); // mono
        wav.extend(8_000u32.to_le_bytes());
        wav.extend(16_000u32.to_le_bytes()); // byte rate
        wav.extend(2u16.to_le_bytes()); // block align
        wav.extend(16u16.to_le_bytes()); // bits
        wav.extend(b"data");
        wav.extend(data.to_le_bytes());
        wav.extend(lk_agents_media::pcm_bytes(&eleven_seconds));
        let mut head = HeadBytes::new(&info).unwrap();
        head.push(&wav).unwrap();
        assert_eq!(head.decode((8_000, 1)), Err(HeadError::Capacity("decoded duration")));
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

/// One buffered head as it arrives, within its encoded byte and PCM duration bounds.
pub struct HeadBytes {
    format: (u32, u16),
    mime: String,
    pcm_limit: Option<usize>,
    data: Vec<u8>,
}

impl HeadBytes {
    pub fn new(
        info: &lk_agents_abi::voice::rtc::StreamInfo,
    ) -> std::result::Result<Self, HeadError> {
        let format = source_format(info)?;
        if info.total_length.is_some_and(|length| length > MAX_BYTES as u64) {
            return Err(HeadError::Capacity("encoded byte"));
        }
        let raw = info.mime_type.is_empty() || info.mime_type == "audio/pcm";
        let pcm_limit = raw.then(|| sample_limit(format) * 2);
        Ok(Self { format, mime: info.mime_type.clone(), pcm_limit, data: Vec::new() })
    }

    pub fn push(&mut self, chunk: &[u8]) -> std::result::Result<(), HeadError> {
        if chunk.len() > MAX_BYTES.saturating_sub(self.data.len()) {
            return Err(HeadError::Capacity("encoded byte"));
        }
        if self.pcm_limit.is_some_and(|limit| chunk.len() > limit.saturating_sub(self.data.len())) {
            return Err(HeadError::Capacity("PCM duration"));
        }
        self.data.extend_from_slice(chunk);
        Ok(())
    }

    /// Synchronous CPU work: decoded, remixed and resampled to the session's `target`, in 10 ms
    /// frames, never more than [`MAX_DURATION`] of it.
    pub fn decode(self, target: (u32, u16)) -> std::result::Result<Vec<AudioFrame>, HeadError> {
        use lk_agents_media::AudioFormat;
        use lk_agents_media::source::{Source, decode, frames};
        let limit = sample_limit(target);
        let source = match self.pcm_limit {
            Some(_) => Source::Pcm {
                data: self.data,
                format: AudioFormat::new(self.format.0, self.format.1),
            },
            None => Source::Container { mime: self.mime, data: self.data },
        };
        let limits = lk_agents_media::decoder::DecodeLimits {
            min_sample_rate: 8_000,
            max_sample_rate: 192_000,
            max_channels: 8,
            max_streams: 8,
            max_expansion_ratio: 24,
            max_samples_per_frame: limit,
        };
        let format = AudioFormat::new(target.0, target.1);
        let mut pcm = Vec::new();
        decode::<HeadError>(
            source,
            Some(target.0),
            target.1,
            limits,
            &|| false,
            &mut |decoded, samples| {
                if decoded != format {
                    return Err(HeadError::Malformed("decoder output is not the session format"));
                }
                if samples.len() > limit - pcm.len() {
                    return Err(HeadError::Capacity("decoded duration"));
                }
                pcm.extend_from_slice(samples);
                Ok(())
            },
        )?;
        let chunk = (target.0 as usize / 100) * usize::from(target.1);
        Ok(frames(format, &pcm, chunk).collect())
    }
}

/// Interleaved samples in [`MAX_DURATION`] of `format`.
fn sample_limit((rate, channels): (u32, u16)) -> usize {
    rate as usize * usize::from(channels) * MAX_DURATION.as_secs() as usize
}

/// What a waiter replays of a head: nothing once it is older than
/// [`super::PRE_CONNECT_MAX_DELTA`], since the live track already carries that speech.
pub fn replayed(buffer: PreConnectBuffer, now: Ts) -> Vec<AudioFrame> {
    if now.saturating_duration_since(buffer.received_at) > super::PRE_CONNECT_MAX_DELTA {
        return Vec::new();
    }
    buffer.frames
}

pub fn source_format(
    info: &lk_agents_abi::voice::rtc::StreamInfo,
) -> std::result::Result<(u32, u16), HeadError> {
    let rate = info
        .attributes
        .get("sampleRate")
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|rate| (8000..=48000).contains(rate))
        .ok_or(HeadError::Malformed("sampleRate must be between 8000 and 48000"))?;
    let channels = info
        .attributes
        .get("channels")
        .and_then(|v| v.parse::<u16>().ok())
        .filter(|channels| (1..=2).contains(channels))
        .ok_or(HeadError::Malformed("channels must be 1 or 2"))?;
    Ok((rate, channels))
}
