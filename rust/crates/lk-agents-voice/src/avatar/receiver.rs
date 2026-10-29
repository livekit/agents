//! Synchronous receive-side framing, clear epochs and segment identity.

use std::collections::VecDeque;
use std::num::NonZeroU32;

use lk_agents_abi::voice::io::AvatarPlaybackSegment;
use lk_agents_abi::voice::rtc::StreamInfo;
use lk_agents_abi::{AudioByteStream, AudioFrame};

use super::{Received, ReceiverEnd, stream_segment};

pub type Receipt = (u64, bool, Option<AvatarPlaybackSegment>, Option<Received>);

/// The receiver's frame length: a zero-length frame never fills.
pub fn frame_size(frame_size_ms: u32) -> Result<NonZeroU32, ReceiverEnd> {
    NonZeroU32::new(frame_size_ms)
        .ok_or_else(|| ReceiverEnd::Failed("frame_size_ms must be positive".into()))
}

#[derive(Default)]
pub struct ReceiverState {
    items: VecDeque<Receipt>,
    epoch: u64,
    end: Option<ReceiverEnd>,
    reading: bool,
    cleared: bool,
    superseded: bool,
}

pub struct ReaderState {
    pub segment: AvatarPlaybackSegment,
    epoch: u64,
    rate: u32,
    channels: u16,
    started: bool,
    bytes: AudioByteStream,
}

impl ReceiverState {
    pub fn ended(&self) -> bool {
        self.end.is_some()
    }
    pub fn end(&mut self, end: ReceiverEnd) {
        self.end.get_or_insert(end);
    }
    pub fn take(&mut self) -> Result<Option<Receipt>, ReceiverEnd> {
        if let Some(item) = self.items.pop_front() {
            return Ok(Some(item));
        }
        match &self.end {
            Some(end) => Err(end.clone()),
            None => Ok(None),
        }
    }
    pub fn superseded(&self) -> bool {
        self.superseded
    }
    pub fn cleared(&self) -> bool {
        self.cleared
    }
    /// A newer admitted stream makes the current reader's missing trailer irrelevant.
    pub fn stream_admitted(&mut self) -> bool {
        if self.reading {
            self.superseded = true;
        }
        self.reading
    }
    pub fn clear(&mut self) {
        self.epoch += 1;
        if self.reading {
            self.cleared = true;
            self.superseded = true;
        }
        self.items.clear();
    }
    pub fn begin(
        &mut self,
        info: &StreamInfo,
        frame_size: NonZeroU32,
    ) -> Result<ReaderState, ReceiverEnd> {
        self.superseded = false;
        self.reading = true;
        let name = info
            .name
            .as_deref()
            .ok_or_else(|| ReceiverEnd::BadStream("avatar stream has no name".into()))?;
        let segment = AvatarPlaybackSegment {
            playback: stream_segment(name)?,
            stream_id: Some(info.id.clone()),
        };
        let (Some(rate), Some(channels)) = (
            info.attributes.get("sample_rate").and_then(|v| v.parse::<u32>().ok()),
            info.attributes.get("num_channels").and_then(|v| v.parse::<u16>().ok()),
        ) else {
            return Err(ReceiverEnd::BadStream(
                "sample_rate or num_channels not found in byte stream".into(),
            ));
        };
        let samples = (u64::from(rate) * u64::from(frame_size.get())).div_ceil(1000) as u32;
        Ok(ReaderState {
            segment,
            epoch: self.epoch,
            rate,
            channels,
            started: false,
            bytes: AudioByteStream::new(rate, channels, Some(samples), false),
        })
    }
    fn push(&mut self, reader: &mut ReaderState, item: Received, starts: bool) {
        if self.end.is_none() {
            self.items.push_back((reader.epoch, starts, Some(reader.segment.clone()), Some(item)));
        }
    }
    /// False discards the remainder of a reader that an explicit clear cut.
    pub fn chunk(&mut self, reader: &mut ReaderState, data: &[u8]) -> bool {
        if self.cleared {
            self.items.clear();
            reader.bytes.clear();
            return false;
        }
        for frame in reader.bytes.push(data) {
            self.frame(reader, frame);
        }
        true
    }
    fn frame(&mut self, reader: &mut ReaderState, frame: lk_agents_abi::utils::byte_stream::Frame) {
        let starts = !std::mem::replace(&mut reader.started, true);
        self.push(
            reader,
            Received::Audio(AudioFrame::new(
                frame.data,
                reader.rate,
                reader.channels,
                frame.samples_per_channel,
            )),
            starts,
        );
    }
    pub fn flush(&mut self, reader: &mut ReaderState) {
        if !self.cleared {
            for frame in reader.bytes.flush() {
                self.frame(reader, frame);
            }
        }
    }
    pub fn finish(&mut self, reader: Option<&mut ReaderState>) {
        self.reading = false;
        self.cleared = false;
        if let Some(reader) = reader {
            self.push(reader, Received::SegmentEnd, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    const FRAME: NonZeroU32 = NonZeroU32::new(100).unwrap();
    fn info() -> StreamInfo {
        StreamInfo {
            id: "STREAM".into(),
            name: Some("AUDIO_3_4".into()),
            attributes: BTreeMap::from([
                ("sample_rate".into(), "10000".into()),
                ("num_channels".into(), "1".into()),
            ]),
            ..Default::default()
        }
    }
    #[test]
    fn framing_and_identity_belong_to_the_admitted_reader() {
        let mut state = ReceiverState::default();
        let mut reader = state.begin(&info(), FRAME).unwrap();
        assert!(state.chunk(&mut reader, &vec![0; 5000]));
        state.flush(&mut reader);
        state.finish(Some(&mut reader));
        let items: Vec<_> = std::iter::from_fn(|| state.take().unwrap()).collect();
        assert_eq!(items.len(), 4);
        assert!(items[0].1);
        assert!(!items[1].1);
        assert!(
            items.iter().all(|item| item.2.as_ref().unwrap().playback == reader.segment.playback)
        );
        assert!(
            matches!(items[2].3, Some(Received::Audio(ref frame)) if frame.samples_per_channel == 500)
        );
        assert!(matches!(items[3].3, Some(Received::SegmentEnd)));
    }
    #[test]
    fn a_clear_discards_queued_audio_but_keeps_the_old_end_epoch() {
        let mut state = ReceiverState::default();
        let mut old = state.begin(&info(), FRAME).unwrap();
        state.chunk(&mut old, &vec![0; 4000]);
        state.clear();
        assert!(state.superseded());
        assert!(!state.chunk(&mut old, &[0; 2]));
        state.flush(&mut old);
        state.finish(Some(&mut old));
        let end = state.take().unwrap().unwrap();
        assert_eq!(end.0, 0);
        assert!(matches!(end.3, Some(Received::SegmentEnd)));
        let mut next = state.begin(&info(), FRAME).unwrap();
        state.chunk(&mut next, &vec![0; 2000]);
        assert_eq!(state.take().unwrap().unwrap().0, 1);
    }
    #[test]
    fn a_superseding_header_releases_only_the_active_reader() {
        let mut state = ReceiverState::default();
        assert!(!state.stream_admitted());
        let mut reader = state.begin(&info(), FRAME).unwrap();
        assert!(state.stream_admitted());
        assert!(state.superseded());
        state.finish(Some(&mut reader));
        state.begin(&info(), FRAME).unwrap();
        assert!(!state.superseded());
    }
}
