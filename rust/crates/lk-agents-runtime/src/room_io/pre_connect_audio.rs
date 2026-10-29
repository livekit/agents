//! Runtime subscriptions, bounded receiving and decoder cleanup for pre-connect audio.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::{ByteStream, RtcRoom, TrackSid};
use lk_agents_abi::{AudioFrame, Clock, Closed, Spawner, StreamSource};
use lk_agents_media::pool::{JobClass, MediaPool};
use lk_agents_net::task::TaskSet;
use lk_agents_voice::room_io::pre_connect_state::{PreConnectBuffer, PreConnectState};
use tokio::sync::Notify;

use lk_agents_voice::room_io::PRE_CONNECT_AUDIO_BUFFER_STREAM;
use lk_agents_voice::room_io::pre_connect_state::{HeadBytes, HeadError, replayed};

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
    /// Where a head decodes; `None` decodes on the executor's own task, as a deterministic
    /// scheduler needs.
    pool: Option<MediaPool>,
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
        pool: Option<MediaPool>,
    ) -> Arc<Self> {
        Arc::new(Self {
            room,
            spawner,
            clock,
            timeout,
            format,
            pool,
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
        let streams =
            crate::rtc::events::byte_streams(self.room.as_ref(), PRE_CONNECT_AUDIO_BUFFER_STREAM);
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
            let admission = self
                .state
                .lock()
                .unwrap()
                .core
                .receive_stream(&stream.info.attributes, stream.info.id.clone());
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
            Ok(outcome) => outcome?,
            Err(_) => {
                self.state.lock().unwrap().core.expire(id);
                self.changed.notify_waiters();
                return Err(IoError::PreConnectTimeout(track.clone()));
            }
        };
        Ok(replayed(outcome, self.clock.now()))
    }

    async fn read_stream(self: Arc<Self>, track: TrackSid, id: u64, mut stream: ByteStream) {
        let stop = self.stopped.notified();
        tokio::pin!(stop);
        stop.as_mut().enable();
        let receive = async {
            if self.state.lock().unwrap().closed {
                return Err(HeadError::Closed);
            }
            receive_bytes(&mut stream).await
        };
        let received = tokio::select! {
            biased;
            () = &mut stop => Err(HeadError::Closed),
            received = lk_agents_abi::timeout(self.clock.as_ref(), self.timeout, receive) => {
                received.unwrap_or(Err(HeadError::Expired))
            },
        };
        let deadline = self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let cleanup = stream.chunks.close(deadline).await;
        self.record_cleanup(cleanup);
        let outcome = match received {
            Ok(head) if cleanup == Closed::Clean => self.decode(head).await,
            Ok(_) => Err(HeadError::Cleanup),
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

    /// On the media pool when there is one, abandoned at close.
    async fn decode(&self, head: HeadBytes) -> std::result::Result<Vec<AudioFrame>, HeadError> {
        let stopped = self.stopped.notified();
        tokio::pin!(stopped);
        stopped.as_mut().enable();
        if self.state.lock().unwrap().closed {
            return Err(HeadError::Closed);
        }
        let format = self.format;
        let Some(pool) = &self.pool else { return head.decode(format) };
        let job = pool
            .try_spawn(JobClass::Decoder, move || head.decode(format))
            .map_err(|error| HeadError::Decode(error.to_string()))?;
        tokio::select! {
            biased;
            () = &mut stopped => Err(HeadError::Closed),
            decoded = job => decoded.map_err(|_| HeadError::Decode("the decoder panicked".into()))?,
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

async fn receive_bytes(stream: &mut ByteStream) -> std::result::Result<HeadBytes, HeadError> {
    let mut head = HeadBytes::new(&stream.info)?;
    let next = |error: lk_agents_abi::errors::ProviderError| HeadError::Stream(error.to_string());
    while let Some(chunk) = stream.chunks.next().await.map_err(next)? {
        head.push(&chunk)?;
        crate::process::yield_now().await;
    }
    Ok(head)
}

#[cfg(test)]
#[path = "pre_connect_audio_tests.rs"]
mod tests;
