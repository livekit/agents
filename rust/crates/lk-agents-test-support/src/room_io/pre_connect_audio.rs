//! Simulated buffered-head transport using production transfer admission and PCM normalization.

use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::{ByteStream, RtcRoom, TrackSid};
use lk_agents_abi::{AudioFrame, Clock, Closed, Spawner, TaskSet};
use lk_agents_voice::room_io::pre_connect_state::{
    MAX_BYTES, MAX_DURATION, PreConnectBuffer, PreConnectState, source_format,
};
use lk_agents_voice::room_io::{PRE_CONNECT_AUDIO_BUFFER_STREAM, PRE_CONNECT_MAX_DELTA};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

pub struct PreConnectAudioHandler {
    state: Mutex<PreConnectState>,
    clock: Arc<dyn Clock>,
    format: (u32, u16),
    timeout: Duration,
    changed: Notify,
    tasks: Arc<TaskSet>,
    closed: Mutex<bool>,
    stopped: Notify,
    cleanup: Mutex<Option<Closed>>,
    outcome: Mutex<Closed>,
}
impl PreConnectAudioHandler {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
        timeout: Duration,
        format: (u32, u16),
    ) -> Result<Arc<Self>> {
        let mut streams = room.byte_streams(PRE_CONNECT_AUDIO_BUFFER_STREAM)?;
        let handler = Arc::new(Self {
            state: Mutex::default(),
            clock,
            format,
            timeout,
            changed: Notify::new(),
            tasks: Arc::default(),
            closed: Mutex::new(false),
            stopped: Notify::new(),
            cleanup: Mutex::new(None),
            outcome: Mutex::new(Closed::Clean),
        });
        let owner = handler.clone();
        let executor = spawner.clone();
        handler.tasks.spawn(spawner.as_ref(), "simulated buffered-head ingress", async move {
            loop {
                let stopped = owner.stopped.notified();
                tokio::pin!(stopped);
                stopped.as_mut().enable();
                if *owner.closed.lock().unwrap() {
                    break;
                }
                let stream = tokio::select! {stream=streams.next()=>stream,()=&mut stopped=>break};
                let Ok(Some(stream)) = stream else { break };
                let Some(track) =
                    stream.info.attributes.get("trackId").map(|id| TrackSid(id.clone()))
                else {
                    owner.record_cleanup(
                        stream
                            .close(
                                owner
                                    .clock
                                    .now()
                                    .saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT),
                            )
                            .await,
                    );
                    continue;
                };
                let slot =
                    owner.state.lock().unwrap().receive(track.clone(), stream.info.id.clone());
                if let Ok(Some(id)) = slot {
                    let task_owner = owner.clone();
                    owner.tasks.spawn(executor.as_ref(), "simulated buffered head", async move {
                        task_owner.read_stream(track, id, stream).await;
                    });
                } else {
                    owner.record_cleanup(
                        stream
                            .close(
                                owner
                                    .clock
                                    .now()
                                    .saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT),
                            )
                            .await,
                    );
                }
            }
            owner.record_cleanup(
                streams
                    .close(owner.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
                    .await,
            );
        });
        Ok(handler)
    }
    async fn read_stream(&self, track: TrackSid, id: u64, mut stream: ByteStream) {
        let stopped = self.stopped.notified();
        tokio::pin!(stopped);
        stopped.as_mut().enable();
        let outcome = if *self.closed.lock().unwrap() {
            Err("pre-connect input is closed".into())
        } else {
            tokio::select! {
                biased;
                () = &mut stopped => Err("pre-connect input is closed".into()),
                received = lk_agents_abi::timeout(self.clock.as_ref(), self.timeout, self.receive(&mut stream)) =>
                    received.unwrap_or_else(|_| Err("pre-connect receive deadline expired".into())),
            }
        };
        let cleanup = stream
            .close(self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
            .await;
        self.record_cleanup(cleanup);
        let outcome = if cleanup == Closed::Clean {
            outcome
        } else {
            Err("pre-connect source cleanup failed".into())
        };
        self.state.lock().unwrap().complete(
            &track,
            id,
            outcome.map(|frames| PreConnectBuffer { received_at: self.clock.now(), frames }),
        );
        self.changed.notify_waiters();
    }

    async fn receive(
        &self,
        stream: &mut ByteStream,
    ) -> std::result::Result<Vec<AudioFrame>, String> {
        let (rate, channels) = source_format(&stream.info)?;
        if stream.info.total_length.is_some_and(|length| length > MAX_BYTES as u64) {
            return Err("pre-connect encoded byte capacity exceeded".into());
        }
        let raw = stream.info.mime_type.is_empty() || stream.info.mime_type == "audio/pcm";
        let mut bytes = Vec::new();
        let limit = (rate as usize) * (channels as usize) * 2 * (MAX_DURATION.as_secs() as usize);
        let read: std::result::Result<(), String> = async {
            while let Some(chunk) = stream.chunks.next().await.map_err(|e| e.to_string())? {
                if chunk.len() > MAX_BYTES.saturating_sub(bytes.len())
                    || (raw && chunk.len() > limit.saturating_sub(bytes.len()))
                {
                    return Err("pre-connect encoded byte/duration capacity exceeded".into());
                }
                bytes.extend(chunk);
            }
            Ok(())
        }
        .await;
        read?;
        if raw && bytes.len() % (usize::from(channels) * 2) != 0 {
            return Err("pre-connect PCM contains an incomplete sample".into());
        }
        let mut pcm = if raw {
            bytes.chunks_exact(2).map(|p| i16::from_le_bytes([p[0], p[1]])).collect::<Vec<_>>()
        } else {
            let mut decoder = lk_agents_media::Decoder::wav(
                Some(rate),
                lk_agents_media::decoder::DecodeLimits {
                    min_sample_rate: 8_000,
                    max_sample_rate: 192_000,
                    max_channels: 8,
                    max_streams: 8,
                    max_expansion_ratio: 24,
                    // This fake transforms the admitted finite head in one calculator step.
                    max_samples_per_frame: limit / 2,
                },
            )
            .map_err(|e| e.to_string())?;
            let mut pcm = decoder.push(&bytes).map_err(|e| e.to_string())?;
            pcm.extend(decoder.finish().map_err(|e| e.to_string())?);
            pcm
        };
        if pcm.len() > limit / 2 {
            return Err("pre-connect decoded duration capacity exceeded".into());
        }
        if channels != self.format.1 {
            pcm = if self.format.1 == 1 {
                lk_agents_media::downmix_mono(&pcm, channels)
            } else if channels == 1 {
                pcm.into_iter()
                    .flat_map(|v| std::iter::repeat_n(v, self.format.1 as usize))
                    .collect()
            } else {
                return Err("unsupported channel conversion".into());
            };
        }
        let mut converter = lk_agents_media::new_resampler(
            rate,
            self.format.0,
            u32::from(self.format.1),
            lk_agents_media::Quality::default(),
        )
        .map_err(|e| e.to_string())?;
        let mut normalized = converter.push(&pcm).map_err(|e| e.to_string())?.to_vec();
        normalized.extend(converter.flush().map_err(|e| e.to_string())?);
        let sample_limit =
            self.format.0 as usize * usize::from(self.format.1) * MAX_DURATION.as_secs() as usize;
        if normalized.len() > sample_limit {
            return Err("pre-connect normalized duration capacity exceeded".into());
        }
        let width = (self.format.0 as usize / 100) * self.format.1 as usize;
        Ok(normalized
            .chunks(width.max(1))
            .map(|pcm| {
                AudioFrame::new(
                    lk_agents_media::pcm_bytes(pcm),
                    self.format.0,
                    self.format.1,
                    (pcm.len() / self.format.1 as usize) as u32,
                )
            })
            .collect())
    }
    pub async fn wait_for_data(&self, track: &TrackSid) -> Result<Vec<AudioFrame>> {
        let id = self.state.lock().unwrap().slot(track)?;
        let waiting = async {
            loop {
                let changed = self.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if let Some(outcome) = self.state.lock().unwrap().take(id) {
                    return outcome;
                }
                changed.await;
            }
        };
        let outcome = lk_agents_abi::timeout(self.clock.as_ref(), self.timeout, waiting).await;
        match outcome {
            Ok(Ok(buffer))
                if self.clock.now().saturating_duration_since(buffer.received_at)
                    <= PRE_CONNECT_MAX_DELTA =>
            {
                Ok(buffer.frames)
            }
            Ok(Ok(_)) => Ok(Vec::new()),
            Ok(Err(error)) => Err(IoError::PreConnectMalformed(error)),
            Err(_) => {
                self.state.lock().unwrap().expire(id);
                Err(IoError::PreConnectTimeout(track.clone()))
            }
        }
    }
    fn record_cleanup(&self, outcome: Closed) {
        let mut recorded = self.outcome.lock().unwrap();
        *recorded = recorded.and(outcome);
    }

    pub async fn aclose(&self) -> Closed {
        if let Some(outcome) = *self.cleanup.lock().unwrap() {
            return outcome;
        }
        *self.closed.lock().unwrap() = true;
        self.state.lock().unwrap().close();
        self.changed.notify_waiters();
        self.stopped.notify_waiters();
        let outcome = if self
            .tasks
            .wait_empty(
                self.clock.as_ref(),
                Some(self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)),
            )
            .await
        {
            Closed::Clean
        } else {
            Closed::TimedOut
        };
        *self.cleanup.lock().unwrap().get_or_insert(outcome.and(*self.outcome.lock().unwrap()))
    }
}
