//! Runtime subscriptions, bounded receiving and decoder cleanup for pre-connect audio.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::codecs::open_bytes;
use lk_agents_abi::media::decoder::ByteDecoderOptions;
use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::{ByteStream, RtcRoom, StreamInfo, TrackSid};
use lk_agents_abi::{AudioFrame, Clock, Closed, Spawner, StreamSource, TaskSet};
use lk_agents_voice::room_io::pre_connect_state::{PreConnectBuffer, PreConnectState};
use tokio::sync::Notify;

use lk_agents_voice::room_io::{PRE_CONNECT_AUDIO_BUFFER_STREAM, PRE_CONNECT_MAX_DELTA};

use lk_agents_voice::room_io::pre_connect_state::{MAX_BYTES, MAX_DURATION, source_format};

#[derive(Default)]
struct State {
    core: PreConnectState,
    registering: bool,
    registered: bool,
    closed: bool,
    cleanup: Option<Closed>,
}

pub struct PreConnectAudioHandler {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    timeout: Duration,
    format: (u32, u16),
    state: Mutex<State>,
    changed: Notify,
    stopped: Notify,
    tasks: Arc<TaskSet>,
}

impl PreConnectAudioHandler {
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
            state: Mutex::new(State::default()),
            changed: Notify::new(),
            stopped: Notify::new(),
            tasks: Arc::default(),
        })
    }

    pub fn register(self: &Arc<Self>) -> bool {
        {
            let mut state = self.state.lock().unwrap();
            if state.closed || state.registering || state.registered {
                return false;
            }
            state.registering = true;
        }
        let streams = self.room.byte_streams(PRE_CONNECT_AUDIO_BUFFER_STREAM);
        let streams = match streams {
            Ok(streams) => streams,
            Err(_) => {
                self.state.lock().unwrap().registering = false;
                self.changed.notify_waiters();
                return false;
            }
        };
        self.state.lock().unwrap().registered = true;
        let owner = self.clone();
        self.tasks.spawn(self.spawner.as_ref(), "pre-connect stream listener", async move {
            owner.listen(streams).await;
        });
        self.state.lock().unwrap().registering = false;
        self.changed.notify_waiters();
        true
    }

    async fn listen(self: Arc<Self>, mut streams: Box<dyn StreamSource<ByteStream>>) {
        loop {
            let stopped = self.stopped.notified();
            tokio::pin!(stopped);
            stopped.as_mut().enable();
            if self.state.lock().unwrap().closed {
                break;
            }
            let stream = tokio::select! {
                biased;
                () = &mut stopped => break,
                next = streams.next() => next,
            };
            let stream = match stream {
                Ok(Some(stream)) => stream,
                Ok(None) => break,
                Err(error) => {
                    self.state.lock().unwrap().core.close();
                    self.changed.notify_waiters();
                    log::warn!(target: "livekit.agents", "pre-connect stream listener failed: {error}");
                    break;
                }
            };
            let admission = stream.info.attributes.get("trackId").map(|track| {
                let track = TrackSid::from(track.clone());
                let outcome =
                    self.state.lock().unwrap().core.receive(track.clone(), stream.info.id.clone());
                (track, outcome)
            });
            self.changed.notify_waiters();
            if let Some((track, Ok(Some(id)))) = admission {
                let owner = self.clone();
                self.tasks.spawn(self.spawner.as_ref(), "pre-connect audio reader", async move {
                    owner.read_stream(track, id, stream).await;
                });
            } else {
                let deadline =
                    self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
                let cleanup = stream.chunks.close(deadline).await;
                self.record_cleanup(cleanup);
            }
        }
        let deadline = self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let cleanup = streams.close(deadline).await;
        self.record_cleanup(cleanup);
        let unregister = {
            let mut state = self.state.lock().unwrap();
            std::mem::take(&mut state.registered)
        };
        if unregister
            && self.room.unregister_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM).is_err()
        {
            self.record_cleanup(Closed::Failed);
        }
    }

    pub async fn wait_for_data(&self, track: &TrackSid) -> Result<Vec<AudioFrame>> {
        let id = self.state.lock().unwrap().core.slot(track)?;
        let wait = async {
            loop {
                let changed = self.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if let Some(outcome) = self.state.lock().unwrap().core.take(id) {
                    return outcome;
                }
                changed.await;
            }
        };
        let outcome = match lk_agents_abi::timeout(self.clock.as_ref(), self.timeout, wait).await {
            Ok(outcome) => outcome.map_err(IoError::PreConnectMalformed)?,
            Err(_) => {
                self.state.lock().unwrap().core.expire(id);
                self.changed.notify_waiters();
                return Err(IoError::PreConnectTimeout(track.clone()));
            }
        };
        if self.clock.now().saturating_duration_since(outcome.received_at) > PRE_CONNECT_MAX_DELTA {
            return Ok(Vec::new());
        }
        Ok(outcome.frames)
    }

    async fn read_stream(self: Arc<Self>, track: TrackSid, id: u64, mut stream: ByteStream) {
        let stop = self.stopped.notified();
        tokio::pin!(stop);
        stop.as_mut().enable();
        let receive = async {
            if self.state.lock().unwrap().closed {
                return Err("pre-connect input is closed".into());
            }
            receive_bytes(&mut stream).await
        };
        let received = tokio::select! {
            biased;
            () = &mut stop => Err("pre-connect input is closed".into()),
            received = lk_agents_abi::timeout(self.clock.as_ref(), self.timeout, receive) => {
                received.unwrap_or_else(|_| Err("pre-connect receive deadline expired".into()))
            },
        };
        let deadline = self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let cleanup = stream.chunks.close(deadline).await;
        self.record_cleanup(cleanup);
        let outcome = match received {
            Ok(data) if cleanup == Closed::Clean => self.decode(data, stream.info).await,
            Ok(_) => Err("pre-connect stream cleanup did not finish".into()),
            Err(error) => Err(error),
        };
        let received_at = self.clock.now();
        self.state.lock().unwrap().core.complete(
            &track,
            id,
            outcome.map(|frames| PreConnectBuffer { received_at, frames }),
        );
        self.changed.notify_waiters();
    }

    async fn decode(
        &self,
        data: Vec<u8>,
        info: StreamInfo,
    ) -> std::result::Result<Vec<AudioFrame>, String> {
        let source_format = source_format(&info)?;
        if self.state.lock().unwrap().closed {
            return Err("pre-connect input is closed".into());
        }
        let raw = info.mime_type.is_empty() || info.mime_type == "audio/pcm";
        let mut decoder = open_bytes(ByteDecoderOptions {
            mime: info.mime_type,
            data,
            pcm_format: raw.then_some(source_format),
            sample_rate: self.format.0,
            num_channels: self.format.1,
            max_input_bytes: MAX_BYTES,
            max_duration: MAX_DURATION,
        })?;
        let decode = async {
            let mut frames = Vec::new();
            let mut retained = 0usize;
            let pcm_limit = self.format.0 as usize
                * usize::from(self.format.1)
                * 2
                * MAX_DURATION.as_secs() as usize;
            loop {
                let stopped = self.stopped.notified();
                tokio::pin!(stopped);
                stopped.as_mut().enable();
                if self.state.lock().unwrap().closed {
                    return Err("pre-connect input is closed".into());
                }
                let frame = tokio::select! {
                    biased;
                    () = &mut stopped => return Err("pre-connect input is closed".into()),
                    frame = decoder.next() => frame?,
                };
                let Some(frame) = frame else {
                    return Ok(frames);
                };
                if (frame.sample_rate, frame.num_channels) != self.format
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
        }
        .await;
        let cleanup = decoder.close(lk_agents_abi::HOST_TEARDOWN_TIMEOUT).await;
        match (decode, cleanup) {
            (Ok(frames), Ok(())) => Ok(frames),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(format!("pre-connect decoder cleanup: {error}")),
            (Err(error), Err(cleanup)) => {
                Err(format!("{error}; pre-connect decoder cleanup: {cleanup}"))
            }
        }
    }

    fn record_cleanup(&self, outcome: Closed) {
        let mut state = self.state.lock().unwrap();
        state.cleanup = Some(state.cleanup.unwrap_or(Closed::Clean).and(outcome));
    }

    pub async fn aclose(&self) -> Closed {
        {
            let mut state = self.state.lock().unwrap();
            state.closed = true;
            state.core.close();
        }
        self.stopped.notify_waiters();
        self.changed.notify_waiters();
        let deadline = self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if !self.state.lock().unwrap().registering {
                break;
            }
            if lk_agents_abi::timeout(
                self.clock.as_ref(),
                deadline.saturating_duration_since(self.clock.now()),
                &mut changed,
            )
            .await
            .is_err()
            {
                return Closed::TimedOut;
            }
        }
        let settled = self.tasks.wait_empty(self.clock.as_ref(), Some(deadline)).await;
        let state = self.state.lock().unwrap();
        if settled && !state.registering {
            state.cleanup.unwrap_or(Closed::Clean)
        } else {
            Closed::TimedOut
        }
    }
}

async fn receive_bytes(stream: &mut ByteStream) -> std::result::Result<Vec<u8>, String> {
    let (sample_rate, num_channels) = source_format(&stream.info)?;
    if stream.info.total_length.is_some_and(|length| length > MAX_BYTES as u64) {
        return Err("pre-connect encoded byte capacity exceeded".into());
    }
    let raw = stream.info.mime_type.is_empty() || stream.info.mime_type == "audio/pcm";
    let pcm_limit =
        (u64::from(sample_rate) * u64::from(num_channels) * 2 * MAX_DURATION.as_secs()) as usize;
    let mut data = Vec::new();
    while let Some(chunk) = stream.chunks.next().await.map_err(|error| error.to_string())? {
        if chunk.len() > MAX_BYTES.saturating_sub(data.len()) {
            return Err("pre-connect encoded byte capacity exceeded".into());
        }
        if raw && chunk.len() > pcm_limit.saturating_sub(data.len()) {
            return Err("pre-connect PCM duration/byte capacity exceeded".into());
        }
        data.extend_from_slice(&chunk);
        tokio::task::yield_now().await;
    }
    Ok(data)
}

#[cfg(test)]
#[path = "pre_connect_audio_tests.rs"]
mod tests;
