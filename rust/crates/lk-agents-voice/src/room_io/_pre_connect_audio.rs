use crate::room_io::{PRE_CONNECT_AUDIO_BUFFER_STREAM, PRE_CONNECT_MAX_DELTA};
use lk_agents_abi::AbortHandle;
use lk_agents_abi::media::decoder::{ByteDecoderOptions, open_bytes};
use lk_agents_abi::voice::io::IoError;
use lk_agents_abi::voice::io::Result;
use lk_agents_abi::{AudioFrame, Clock, Spawner, SpawnerExt, Ts};
use lk_agents_rtc::{ByteStream, RtcRoom, TrackSid};
use std::collections::{BTreeMap, BTreeSet};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

#[derive(Debug)]
pub struct PreConnectBuffer {
    /// When the buffer finished arriving. Staleness is measured from here.
    pub received_at: Ts,
    pub frames: Vec<AudioFrame>,
}

#[derive(Default)]
enum SlotValue {
    #[default]
    Pending,
    Ready(std::result::Result<PreConnectBuffer, String>),
    Taken,
}

#[derive(Default)]
struct Slot {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    value: Mutex<SlotValue>,
    wake: Notify,
}

impl Slot {
    fn set(&self, v: std::result::Result<PreConnectBuffer, String>) {
        let mut value = self.value.lock().unwrap();
        if !matches!(*value, SlotValue::Pending) {
            return;
        }
        *value = SlotValue::Ready(v);
        drop(value);
        self.wake.notify_waiters();
    }

    async fn get(&self) -> std::result::Result<PreConnectBuffer, String> {
        loop {
            let parked = self.wake.notified();
            tokio::pin!(parked);
            parked.as_mut().enable();
            {
                let mut value = self.value.lock().unwrap();
                match &*value {
                    SlotValue::Ready(_) => {
                        let SlotValue::Ready(result) =
                            std::mem::replace(&mut *value, SlotValue::Taken)
                        else {
                            unreachable!()
                        };
                        return result;
                    }
                    SlotValue::Taken => return Err("pre-connect buffer already transferred".into()),
                    SlotValue::Pending => {}
                }
            }
            parked.await;
        }
    }
}

// A pre-connect head is at most ten seconds; admit at most four incoming heads and 32
// track/transfer identities per session. Encoded input and retained PCM have independent caps.
const MAX_DURATION: Duration = Duration::from_secs(10);
const MAX_BYTES: usize = 2 * 1024 * 1024;
const MAX_STREAMS: usize = 4;
pub(super) const MAX_TRACKS: usize = 32;

#[derive(Default)]
struct Buffers {
    slots: BTreeMap<TrackSid, Arc<Slot>>,
    transfers: BTreeSet<(TrackSid, String)>,
    receiving: BTreeSet<TrackSid>,
    closed: bool,
}

#[derive(Default)]
struct Tasks {
    closed: bool,
    listener: Option<AbortHandle>,
    readers: Vec<AbortHandle>,
}

pub struct PreConnectAudioHandler {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    timeout: Duration,
    format: (u32, u16),
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    buffers: Mutex<Buffers>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    tasks: Mutex<Tasks>,
}

impl PreConnectAudioHandler {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
        timeout: Duration,
        format: (u32, u16),
    ) -> Arc<Self> {
        Arc::new(Self {
            room,
            spawner,
            clock,
            timeout,
            format,
            buffers: Mutex::new(Buffers::default()),
            tasks: Mutex::new(Tasks::default()),
        })
    }

    pub fn register(self: &Arc<Self>) -> bool {
        if self.tasks.lock().unwrap().closed {
            return false;
        }
        let Ok(mut streams) = self.room.byte_streams(PRE_CONNECT_AUDIO_BUFFER_STREAM) else {
            return false;
        };
        let this = self.clone();
        let handle = self.spawner.spawn_unboxed(async move {
            while let Some(stream) = streams.recv().await {
                let mut tasks = this.tasks.lock().unwrap();
                if tasks.closed {
                    break;
                }
                tasks.readers.retain(|task| !task.is_finished());
                if tasks.readers.len() == MAX_STREAMS {
                    if let Some(track) = stream.info.attributes.get("trackId")
                        && let Ok(slot) = this.slot(&TrackSid::from(track.clone()))
                    {
                        slot.set(Err("pre-connect stream capacity exceeded".into()));
                    }
                    continue;
                }
                tasks.readers.push(this.spawner.spawn_unboxed(this.clone().read_stream(stream)));
            }
        });
        let mut tasks = self.tasks.lock().unwrap();
        if tasks.closed {
            drop(tasks);
            handle.abort();
            self.room.unregister_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM);
            return false;
        }
        tasks.listener = Some(handle);
        true
    }

    fn slot(&self, track: &TrackSid) -> Result<Arc<Slot>> {
        let mut buffers = self.buffers.lock().unwrap();
        if buffers.closed {
            return Err(IoError::PreConnectMalformed("pre-connect input is closed".into()));
        }
        if !buffers.slots.contains_key(track) && buffers.slots.len() >= MAX_TRACKS {
            return Err(IoError::PreConnectMalformed("pre-connect track capacity exceeded".into()));
        }
        Ok(buffers.slots.entry(track.clone()).or_default().clone())
    }

    /// Move this head to one input. Competing waiters observe consumption, never cloned PCM.
    pub async fn wait_for_data(&self, track: &TrackSid) -> Result<Vec<AudioFrame>> {
        let slot = self.slot(track)?;
        let outcome = lk_agents_abi::timeout(&*self.clock, self.timeout, slot.get()).await.ok();
        match outcome {
            None => {
                slot.set(Err("pre-connect receive deadline expired".into()));
                Err(IoError::PreConnectTimeout(track.clone()))
            }
            Some(Err(error)) => Err(IoError::PreConnectMalformed(error)),
            Some(Ok(buffer)) => {
                if self.clock.now().saturating_duration_since(buffer.received_at)
                    > PRE_CONNECT_MAX_DELTA
                {
                    return Ok(Vec::new());
                }
                Ok(buffer.frames)
            }
        }
    }

    async fn read_stream(self: Arc<Self>, stream: ByteStream) {
        let Some(track) = stream.info.attributes.get("trackId") else { return };
        let track = TrackSid::from(track.clone());
        let slot = {
            let mut buffers = self.buffers.lock().unwrap();
            let identity = (track.clone(), stream.info.id.clone());
            if buffers.closed
                || buffers.transfers.contains(&identity)
                || buffers.receiving.contains(&track)
            {
                return;
            }
            if buffers.transfers.len() == MAX_TRACKS {
                if let Some(slot) = buffers.slots.get(&track) {
                    slot.set(Err("pre-connect transfer capacity exceeded".into()));
                }
                return;
            }
            if !buffers.slots.contains_key(&track) && buffers.slots.len() == MAX_TRACKS {
                return;
            }
            let slot = buffers.slots.entry(track.clone()).or_default();
            if !matches!(*slot.value.lock().unwrap(), SlotValue::Pending) {
                *slot = Arc::default();
            }
            let slot = slot.clone();
            buffers.transfers.insert(identity);
            buffers.receiving.insert(track.clone());
            slot
        };
        let received =
            lk_agents_abi::timeout(&*self.clock, self.timeout, receive(stream, self.format)).await;
        let outcome = match received {
            Ok(Ok(frames)) => Ok(PreConnectBuffer { received_at: self.clock.now(), frames }),
            Ok(Err(error)) => Err(error),
            Err(_) => Err("pre-connect receive deadline expired".into()),
        };
        let mut buffers = self.buffers.lock().unwrap();
        slot.set(outcome);
        buffers.receiving.remove(&track);
    }

    pub async fn aclose(&self) {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.closed = true;
        for (_, slot) in std::mem::take(&mut buffers.slots) {
            slot.set(Err("pre-connect input is closed".into()));
            // Release unread PCM, too. A waiter holding this slot sees a terminal outcome.
            let mut value = slot.value.lock().unwrap();
            if matches!(*value, SlotValue::Ready(Ok(_))) {
                *value = SlotValue::Ready(Err("pre-connect input is closed".into()));
            }
        }
        buffers.transfers.clear();
        buffers.receiving.clear();
        drop(buffers);
        let (listener, readers) = {
            let mut tasks = self.tasks.lock().unwrap();
            tasks.closed = true;
            (tasks.listener.take(), std::mem::take(&mut tasks.readers))
        };
        // A failed registration never acquired the topic. Removing it would disconnect the
        // handler that was already there; a registration racing close retires its own result.
        if let Some(listener) = listener {
            self.room.unregister_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM);
            listener.abort();
        }
        for task in readers {
            task.abort();
        }
    }
}

async fn receive(
    mut stream: ByteStream,
    format: (u32, u16),
) -> std::result::Result<Vec<AudioFrame>, String> {
    let attrs = &stream.info.attributes;
    let sample_rate = attrs
        .get("sampleRate")
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|rate| (8000..=48000).contains(rate))
        .ok_or("pre-connect sampleRate must be between 8000 and 48000")?;
    let num_channels = attrs
        .get("channels")
        .and_then(|v| v.parse::<u16>().ok())
        .filter(|channels| (1..=2).contains(channels))
        .ok_or("pre-connect channels must be 1 or 2")?;
    if !(8000..=192000).contains(&format.0) || !(1..=8).contains(&format.1) {
        return Err("pre-connect target must be 8-192 kHz with 1-8 channels".into());
    }
    if stream.info.total_length.is_some_and(|size| size > MAX_BYTES as u64) {
        return Err("pre-connect encoded byte capacity exceeded".into());
    }
    let raw = stream.info.mime_type.is_empty() || stream.info.mime_type == "audio/pcm";
    let pcm_limit =
        (u64::from(sample_rate) * u64::from(num_channels) * 2 * MAX_DURATION.as_secs()) as usize;
    let mut data = Vec::new();
    while let Some(chunk) = stream.chunks.recv().await {
        if chunk.len() > MAX_BYTES.saturating_sub(data.len()) {
            return Err("pre-connect encoded byte capacity exceeded".into());
        }
        if raw && chunk.len() > pcm_limit.saturating_sub(data.len()) {
            return Err("pre-connect PCM duration/byte capacity exceeded".into());
        }
        data.extend_from_slice(&chunk);
        tokio::task::yield_now().await;
    }
    let mut decoder = open_bytes(ByteDecoderOptions {
        mime: stream.info.mime_type,
        data,
        pcm_format: raw.then_some((sample_rate, num_channels)),
        sample_rate: format.0,
        num_channels: format.1,
        max_input_bytes: MAX_BYTES,
        max_duration: MAX_DURATION,
    })?;
    let outcome = async {
        let mut frames = Vec::new();
        let mut retained = 0usize;
        let pcm_limit =
            format.0 as usize * usize::from(format.1) * 2 * MAX_DURATION.as_secs() as usize;
        while let Some(frame) = decoder.next().await? {
            if (frame.sample_rate, frame.num_channels) != format
                || frame.data.len() > pcm_limit.saturating_sub(retained)
                || frame.data.len() > MAX_BYTES.saturating_sub(retained)
            {
                return Err(
                    "pre-connect decoder exceeded the admitted PCM format or capacity".into()
                );
            }
            retained += frame.data.len();
            frames.push(frame);
            tokio::task::yield_now().await;
        }
        Ok(frames)
    }
    .await;
    let cleanup = decoder.close(lk_agents_abi::HOST_TEARDOWN_TIMEOUT).await;
    match (outcome, cleanup) {
        (Ok(frames), Ok(())) => Ok(frames),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(format!("pre-connect decoder cleanup: {error}")),
        (Err(error), Err(cleanup)) => {
            Err(format!("{error}; pre-connect decoder cleanup: {cleanup}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::media::decoder::{FileDecoder, FileDecoderProvider};
    use lk_agents_abi::utils::spawner::BoxFuture;

    struct DecodedFrames(std::vec::IntoIter<AudioFrame>);

    impl FileDecoder for DecodedFrames {
        fn next(&mut self) -> BoxFuture<'_, std::result::Result<Option<AudioFrame>, String>> {
            Box::pin(std::future::ready(Ok(self.0.next())))
        }

        fn close(&mut self, _: Duration) -> BoxFuture<'_, std::result::Result<(), String>> {
            self.0 = Vec::new().into_iter();
            Box::pin(std::future::ready(Ok(())))
        }
    }

    // Core tests use native media calculators; runtime tests exercise real admission and cleanup.
    fn decode_fixture(
        options: ByteDecoderOptions,
    ) -> std::result::Result<Box<dyn FileDecoder>, String> {
        use lk_agents_media::{AudioFormat, FrameResampler, Quality, downmix_mono, pcm_bytes};
        if !matches!(options.mime.as_str(), "" | "audio/pcm") {
            return Err("the core fixture only receives supplied PCM".into());
        }
        let (source_rate, source_channels) = options.pcm_format.ok_or("missing PCM format")?;
        let mut source =
            lk_agents_abi::AudioByteStream::new(source_rate, source_channels, None, false);
        let mut frames = source.push(&options.data);
        frames.extend(source.flush());
        let mut resampler = if source_rate == options.sample_rate {
            None
        } else {
            Some(
                FrameResampler::fixed(
                    AudioFormat::new(source_rate, options.num_channels),
                    options.sample_rate,
                    Quality::default(),
                )
                .map_err(|error| error.to_string())?,
            )
        };
        let mut normalized = Vec::new();
        for frame in frames {
            let samples = frame
                .data
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect::<Vec<_>>();
            let samples = if source_channels == options.num_channels {
                samples
            } else if options.num_channels == 1 {
                downmix_mono(&samples, source_channels)
            } else if source_channels == 1 {
                samples
                    .into_iter()
                    .flat_map(|sample| {
                        std::iter::repeat_n(sample, usize::from(options.num_channels))
                    })
                    .collect()
            } else {
                return Err("unsupported fixture channel conversion".into());
            };
            let frame = AudioFrame::new(
                pcm_bytes(&samples),
                source_rate,
                options.num_channels,
                frame.samples_per_channel,
            );
            if let Some(resampler) = &mut resampler {
                normalized.extend(resampler.push(&frame).map_err(|error| error.to_string())?);
            } else {
                normalized.push(frame);
            }
        }
        if let Some(resampler) = &mut resampler {
            normalized.extend(resampler.flush().map_err(|error| error.to_string())?);
        }
        Ok(Box::new(DecodedFrames(normalized.into_iter())))
    }

    lk_agents_abi::inventory::submit! {
        FileDecoderProvider {
            open: |_| Err("the room core does not open files".into()),
            open_bytes: decode_fixture,
        }
    }

    #[tokio::test]
    async fn failed_registration_does_not_remove_another_stream_owner() {
        let (rtc, clock) = lk_agents_rtc::FakeRtc::with_virtual_clock();
        let room = rtc.room("registration-owner");
        let mut original = room.byte_streams(PRE_CONNECT_AUDIO_BUFFER_STREAM).unwrap();
        let handler = PreConnectAudioHandler::new(
            room.clone(),
            Arc::new(lk_agents_abi::TokioSpawner::current()),
            clock,
            Duration::from_secs(3),
            (24_000, 1),
        );
        assert!(!handler.register());
        handler.aclose().await;
        let info = lk_agents_rtc::StreamInfo {
            id: "original-owner".into(),
            topic: PRE_CONNECT_AUDIO_BUFFER_STREAM.into(),
            ..Default::default()
        };
        room.send_byte_stream("user", info, [vec![1, 2]]).unwrap();
        let mut stream = original.recv().await.unwrap();
        assert_eq!(stream.chunks.recv().await.unwrap(), vec![1, 2]);
    }

    #[tokio::test]
    async fn closed_preconnect_input_cannot_register_again() {
        let (rtc, clock) = lk_agents_rtc::FakeRtc::with_virtual_clock();
        let room = rtc.room("closed-registration");
        let handler = PreConnectAudioHandler::new(
            room.clone(),
            Arc::new(lk_agents_abi::TokioSpawner::current()),
            clock,
            Duration::from_secs(3),
            (24_000, 1),
        );
        handler.aclose().await;
        assert!(!handler.register());
        assert!(room.byte_streams(PRE_CONNECT_AUDIO_BUFFER_STREAM).is_ok());
    }

    fn pcm_info(track: &str) -> lk_agents_rtc::StreamInfo {
        lk_agents_rtc::StreamInfo {
            id: track.into(),
            topic: PRE_CONNECT_AUDIO_BUFFER_STREAM.into(),
            attributes: [
                ("trackId".into(), track.into()),
                ("sampleRate".into(), "24000".into()),
                ("channels".into(), "1".into()),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn preconnect_overload_preserves_already_admitted_tracks() {
        let (rtc, clock) = lk_agents_rtc::FakeRtc::with_virtual_clock();
        let room = rtc.room("track-capacity");
        let handler = PreConnectAudioHandler::new(
            room.clone(),
            Arc::new(lk_agents_abi::TokioSpawner::current()),
            clock,
            Duration::from_secs(3),
            (24_000, 1),
        );
        assert!(handler.register());
        let tracks = (0..MAX_TRACKS).map(|n| TrackSid::from(format!("TR_{n}"))).collect::<Vec<_>>();
        let mut waiters =
            tracks.iter().map(|track| Box::pin(handler.wait_for_data(track))).collect::<Vec<_>>();
        for waiter in &mut waiters {
            crate::room_io::tests::pending(waiter.as_mut()).await;
        }
        assert!(matches!(
            handler.wait_for_data(&"TR_overload".into()).await,
            Err(IoError::PreConnectMalformed(error)) if error.contains("track capacity")
        ));
        room.send_byte_stream("user", pcm_info("TR_0"), [vec![1; 960]]).unwrap();
        let frames = waiters.remove(0).await.unwrap();
        assert_eq!(frames.iter().map(|frame| frame.data.len()).sum::<usize>(), 960);
        handler.aclose().await;
        for waiter in waiters {
            assert!(waiter.await.is_err());
        }
    }

    #[tokio::test]
    async fn preconnect_receive_deadline_discards_partial_pcm() {
        let (rtc, clock) = lk_agents_rtc::FakeRtc::with_virtual_clock();
        let room = rtc.room("receive-deadline");
        let handler = PreConnectAudioHandler::new(
            room.clone(),
            Arc::new(lk_agents_abi::TokioSpawner::current()),
            clock.clone(),
            Duration::from_secs(3),
            (24_000, 1),
        );
        assert!(handler.register());
        let chunks = room.open_byte_stream("user", pcm_info("TR_expired")).unwrap();
        chunks.send(vec![1; 960]).unwrap();
        let track = TrackSid::from("TR_expired");
        let mut result = Box::pin(handler.wait_for_data(&track));
        crate::room_io::tests::pending(result.as_mut()).await;
        tokio::task::yield_now().await;
        clock.advance(Duration::from_secs(3));
        assert!(result.await.is_err());
        tokio::task::yield_now().await;
        assert!(chunks.send(vec![2; 960]).is_err(), "expired input must be closed");
        handler.aclose().await;
    }

    #[tokio::test]
    async fn preconnect_single_transfer() {
        let slot = Slot::default();
        let mut first = Box::pin(slot.get());
        let mut second = Box::pin(slot.get());
        crate::room_io::tests::pending(first.as_mut()).await;
        crate::room_io::tests::pending(second.as_mut()).await;
        slot.set(Ok(PreConnectBuffer {
            received_at: Ts::ZERO,
            frames: vec![lk_agents_rtc::silence_frame(480, 24000, 1)],
        }));
        let (first, second) = (first.await, second.await);
        let transferred = [first, second]
            .into_iter()
            .filter_map(|r| r.ok())
            .map(|b| b.frames.len())
            .sum::<usize>();
        assert_eq!(transferred, 1, "a pre-connect head belongs to one consumer");
    }
    #[tokio::test]
    async fn preconnect_duration_is_bounded_before_transfer() {
        let (rtc, clock) = lk_agents_rtc::FakeRtc::with_virtual_clock();
        let room = rtc.room("duration");
        let handler = PreConnectAudioHandler::new(
            room.clone(),
            Arc::new(lk_agents_abi::TokioSpawner::current()),
            clock,
            Duration::from_secs(3),
            (24_000, 1),
        );
        let info = lk_agents_rtc::StreamInfo {
            id: "buffer".into(),
            topic: PRE_CONNECT_AUDIO_BUFFER_STREAM.into(),
            attributes: [
                ("trackId".into(), "TR_buffer".into()),
                ("sampleRate".into(), "24000".into()),
                ("channels".into(), "1".into()),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        assert!(handler.register());
        room.send_byte_stream("user", info, (0..501).map(|_| vec![1; 960])).unwrap();
        assert!(
            handler.wait_for_data(&"TR_buffer".into()).await.is_err(),
            "more than ten seconds of pre-connect PCM must be refused"
        );
        handler.aclose().await;
    }
}
