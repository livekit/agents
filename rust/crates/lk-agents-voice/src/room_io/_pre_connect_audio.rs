use crate::room_io::{PRE_CONNECT_AUDIO_BUFFER_STREAM, PRE_CONNECT_MAX_DELTA};
use lk_agents_abi::AbortHandle;
use lk_agents_abi::voice::io::IoError;
use lk_agents_abi::voice::io::Result;
use lk_agents_abi::{AudioByteStream, AudioFrame, Clock, Spawner, SpawnerExt, Ts};
use lk_agents_media::{Decoder, pcm_bytes};
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
    listener: Option<AbortHandle>,
    readers: Vec<AbortHandle>,
}

pub struct PreConnectAudioHandler {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    timeout: Duration,
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
    ) -> Arc<Self> {
        Arc::new(Self {
            room,
            spawner,
            clock,
            timeout,
            buffers: Mutex::new(Buffers::default()),
            tasks: Mutex::new(Tasks::default()),
        })
    }

    pub fn register(self: &Arc<Self>) -> bool {
        let Ok(mut streams) = self.room.byte_streams(PRE_CONNECT_AUDIO_BUFFER_STREAM) else {
            return false;
        };
        let this = self.clone();
        let handle = self.spawner.spawn_unboxed(async move {
            while let Some(stream) = streams.recv().await {
                let mut tasks = this.tasks.lock().unwrap();
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
        self.tasks.lock().unwrap().listener = Some(handle);
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
        let received = lk_agents_abi::timeout(&*self.clock, self.timeout, receive(stream)).await;
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
        self.room.unregister_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM);
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
        let mut tasks = self.tasks.lock().unwrap();
        if let Some(listener) = tasks.listener.take() {
            listener.abort();
        }
        for task in tasks.readers.drain(..) {
            task.abort();
        }
    }
}

async fn receive(mut stream: ByteStream) -> std::result::Result<Vec<AudioFrame>, String> {
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
    if stream.info.total_length.is_some_and(|size| size > MAX_BYTES as u64) {
        return Err("pre-connect encoded byte capacity exceeded".into());
    }
    let is_opus =
        stream.info.mime_type == "audio/opus" || stream.info.mime_type.contains("codecs=opus");
    let mut decoder = if is_opus {
        Some(
            Decoder::new(&stream.info.mime_type, Some(sample_rate), num_channels)
                .map_err(|error| format!("pre-connect opus: {error}"))?,
        )
    } else {
        None
    };
    let pcm_limit =
        (u64::from(sample_rate) * u64::from(num_channels) * 2 * MAX_DURATION.as_secs()) as usize;
    let mut bstream = AudioByteStream::new(sample_rate, num_channels, None, false);
    let mut frames = Vec::new();
    let mut encoded_bytes = 0usize;
    let mut pcm_bytes_received = 0usize;
    let mut push_pcm = |bytes: &[u8]| -> std::result::Result<(), String> {
        if bytes.len() > pcm_limit.saturating_sub(pcm_bytes_received)
            || bytes.len() > MAX_BYTES.saturating_sub(pcm_bytes_received)
        {
            return Err("pre-connect PCM duration/byte capacity exceeded".into());
        }
        pcm_bytes_received += bytes.len();
        frames.extend(bstream.push(bytes).into_iter().map(|frame| {
            AudioFrame::new(frame.data, sample_rate, num_channels, frame.samples_per_channel)
        }));
        Ok(())
    };
    while let Some(chunk) = stream.chunks.recv().await {
        if chunk.len() > MAX_BYTES.saturating_sub(encoded_bytes) {
            return Err("pre-connect encoded byte capacity exceeded".into());
        }
        encoded_bytes += chunk.len();
        match &mut decoder {
            Some(decoder) => {
                let pcm =
                    decoder.push(&chunk).map_err(|error| format!("pre-connect decode: {error}"))?;
                push_pcm(&pcm_bytes(&pcm))?;
            }
            None => push_pcm(&chunk)?,
        }
        tokio::task::yield_now().await;
    }
    if let Some(decoder) = &mut decoder {
        let pcm = decoder.flush().map_err(|error| format!("pre-connect flush: {error}"))?;
        push_pcm(&pcm_bytes(&pcm))?;
    }
    if !pcm_bytes_received.is_multiple_of(usize::from(num_channels) * 2) {
        return Err("pre-connect PCM ended within a sample".into());
    }
    frames.extend(bstream.flush().into_iter().map(|frame| {
        AudioFrame::new(frame.data, sample_rate, num_channels, frame.samples_per_channel)
    }));
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

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
