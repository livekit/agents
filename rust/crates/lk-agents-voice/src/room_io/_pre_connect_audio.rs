use crate::io::{IoError, Result};
use crate::room_io::types::{PRE_CONNECT_AUDIO_BUFFER_STREAM, PRE_CONNECT_MAX_DELTA};
use lk_agents_abi::{AudioByteStream, AudioFrame, Clock, Spawner, SpawnerExt, Ts};
use lk_agents_media::Decoder;
use lk_agents_rtc::{ByteStream, RtcRoom, TrackSid};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;
use lk_agents_abi::AbortHandle;

#[derive(Debug, Clone)]
pub struct PreConnectBuffer {
    /// When the buffer finished arriving. Staleness is measured from here.
    pub received_at: Ts,
    pub frames: Vec<AudioFrame>,
}

#[derive(Default)]
struct Slot {
    value: Mutex<Option<std::result::Result<PreConnectBuffer, String>>>,
    wake: Notify,
}

impl Slot {
    fn set(&self, v: std::result::Result<PreConnectBuffer, String>) {
        *self.value.lock().unwrap() = Some(v);
        self.wake.notify_waiters();
    }

    async fn get(&self) -> std::result::Result<PreConnectBuffer, String> {
        loop {
            let parked = self.wake.notified();
            tokio::pin!(parked);
            parked.as_mut().enable();
            if let Some(v) = self.value.lock().unwrap().clone() {
                return v;
            }
            parked.await;
        }
    }
}

pub struct PreConnectAudioHandler {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    timeout: Duration,
    buffers: Mutex<HashMap<TrackSid, Arc<Slot>>>,
    tasks: Mutex<Vec<AbortHandle>>,
}

impl PreConnectAudioHandler {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
        timeout: Duration,
    ) -> Arc<Self> {
        Arc::new(PreConnectAudioHandler {
            room,
            spawner,
            clock,
            timeout,
            buffers: Mutex::new(HashMap::new()),
            tasks: Mutex::new(Vec::new()),
        })
    }

    /// Subscribe to the pre-connect topic. A duplicate registration warns and continues; returns
    /// whether it took.
    pub fn register(self: &Arc<Self>) -> bool {
        let Ok(mut streams) =
            self.room.byte_streams(PRE_CONNECT_AUDIO_BUFFER_STREAM)
        else {
            return false;
        };
        let this = self.clone();
        let handle = self.spawner.spawn_unboxed(async move {
            while let Some(stream) = streams.recv().await {
                let this = this.clone();
                // One task and timeout per stream: several tracks can buffer at once.
                let handle = this.spawner.spawn_unboxed(this.clone().read_stream(stream));
                this.tasks.lock().unwrap().push(handle);
            }
        });
        self.tasks.lock().unwrap().push(handle);
        true
    }

    fn slot(&self, track: &TrackSid) -> Arc<Slot> {
        self.buffers.lock().unwrap().entry(track.clone()).or_default().clone()
    }

    /// The buffered head of this track's audio, consumed once. A timeout is not fatal.
    pub async fn wait_for_data(&self, track: &TrackSid) -> Result<Vec<AudioFrame>> {
        let slot = self.slot(track);
        let outcome = {
            let ready = slot.value.lock().unwrap().clone();
            match ready {
                Some(v) => Some(v),
                // Bounded on the session clock, so a test can drive the timeout.
                None => lk_agents_abi::timeout(&*self.clock, self.timeout, slot.get()).await.ok(),
            }
        };
        self.buffers.lock().unwrap().remove(track);

        match outcome {
            None => Err(IoError::PreConnectTimeout(track.clone())),
            Some(Err(e)) => Err(IoError::PreConnectMalformed(e)),
            Some(Ok(buf)) => {
                let age = self.clock.now().saturating_duration_since(buf.received_at);
                if age > PRE_CONNECT_MAX_DELTA {
                    // Stale: the user abandoned this utterance.
                    return Ok(Vec::new());
                }
                Ok(buf.frames)
            }
        }
    }

    async fn read_stream(self: Arc<Self>, mut stream: ByteStream) {
        let attrs = &stream.info.attributes;
        let Some(track) = attrs.get("trackId") else {
            // No track to attach it to; the reference warns and drops it.
            return;
        };
        let track = TrackSid::from(track.clone());

        // A slot that already holds a buffer is replaced: the client is sending a *new* head.
        {
            let mut buffers = self.buffers.lock().unwrap();
            if buffers.get(&track).is_some_and(|s| s.value.lock().unwrap().is_some()) {
                buffers.remove(&track);
            }
        }
        let slot = self.slot(&track);

        let (Some(rate), Some(channels)) = (attrs.get("sampleRate"), attrs.get("channels")) else {
            slot.set(Err("sampleRate or channels not found in pre-connect byte stream".into()));
            return;
        };
        let (Ok(sample_rate), Ok(num_channels)) = (rate.parse::<u32>(), channels.parse::<u16>())
        else {
            slot.set(Err(format!("unparseable sampleRate/channels: {rate:?}/{channels:?}")));
            return;
        };

        // `audio/opus` or `audio/webm;codecs=opus` needs a decoder (stateful, one per stream);
        // anything else is raw PCM.
        let is_opus =
            stream.info.mime_type == "audio/opus" || stream.info.mime_type.contains("codecs=opus");
        let mut decoder = if is_opus {
            match Decoder::new(&stream.info.mime_type, sample_rate, num_channels) {
                Ok(d) => Some(d),
                // The decoder's own error names the container and the missing demuxer.
                Err(e) => {
                    slot.set(Err(format!("pre-connect opus: {e}")));
                    return;
                }
            }
        } else {
            None
        };

        let mut bstream = AudioByteStream::new(sample_rate, num_channels, None, false);
        let mut frames = Vec::new();
        let mut push_pcm = |bytes: &[u8], frames: &mut Vec<AudioFrame>| {
            for f in bstream.push(bytes) {
                frames.push(AudioFrame::new(
                    f.data,
                    sample_rate,
                    num_channels,
                    f.samples_per_channel,
                ));
            }
        };
        while let Some(chunk) = stream.chunks.recv().await {
            match &mut decoder {
                // The decoder yields `i16` samples; the byte stream frames bytes.
                Some(d) => match d.push(&chunk) {
                    Ok(pcm) => push_pcm(&pcm_bytes(&pcm), &mut frames),
                    Err(e) => {
                        slot.set(Err(format!("pre-connect opus decode failed: {e}")));
                        return;
                    }
                },
                None => push_pcm(&chunk, &mut frames),
            }
        }
        // Flush the decoder first, or the final syllable is truncated.
        if let Some(d) = &mut decoder {
            match d.flush() {
                Ok(pcm) => push_pcm(&pcm_bytes(&pcm), &mut frames),
                Err(e) => {
                    slot.set(Err(format!("pre-connect opus flush failed: {e}")));
                    return;
                }
            }
        }
        for f in bstream.flush().frames {
            frames.push(AudioFrame::new(f.data, sample_rate, num_channels, f.samples_per_channel));
        }

        slot.set(Ok(PreConnectBuffer { received_at: self.clock.now(), frames }));
    }

    pub async fn aclose(&self) {
        self.room.unregister_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM);
        for h in self.tasks.lock().unwrap().drain(..) {
            h.abort();
        }
    }
}

/// Interleaved `i16` PCM as the little-endian bytes `AudioByteStream` frames.
fn pcm_bytes(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}
