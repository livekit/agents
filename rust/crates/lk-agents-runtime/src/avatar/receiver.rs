//! Runtime avatar receiver execution over the pure receive state.

use lk_agents_abi::voice::io::AvatarPlaybackSegment;
use lk_agents_abi::voice::rtc::{ParticipantIdentity, RpcHandler, RtcRoom};
use lk_agents_abi::{AbortHandle, Clock, Spawner, SpawnerExt};
use lk_agents_voice::avatar::receiver::ReceiverState;
use lk_agents_voice::avatar::{
    AUDIO_STREAM_TOPIC, ClearBufferListener, FinishedPayload, RPC_CLEAR_BUFFER,
    RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED, Received, ReceiverEnd, STREAM_IDLE_TIMEOUT,
    playback_method,
};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::time::Duration;

const LOG: &str = "livekit.agents";
const RPC_UNSUPPORTED_METHOD: u32 = 1400;
enum RpcEvent {
    Finished { segment: AvatarPlaybackSegment, playback_position: f64, interrupted: bool },
    Started(AvatarPlaybackSegment),
}

struct RxInner {
    room: Arc<dyn RtcRoom>,
    sender_identity: Option<String>,
    frame_size_ms: u32,
    rpc_max_retries: u32,
    clock: Arc<dyn Clock>,
    on_clear: Option<Arc<dyn Fn() + Send + Sync>>,
    remote: Mutex<Option<String>>,
    state: Mutex<ReceiverState>,
    out_ready: tokio::sync::Notify,
    streams: Mutex<std::collections::VecDeque<lk_agents_abi::voice::rtc::ByteStream>>,
    streams_ready: tokio::sync::Notify,
    superseded_wake: tokio::sync::Notify,
    rpc: tokio::sync::mpsc::UnboundedSender<RpcEvent>,
    rpc_rx: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<RpcEvent>>>,
}

impl RxInner {
    fn end(&self, end: ReceiverEnd) {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).end(end);
        self.out_ready.notify_waiters();
        self.streams_ready.notify_waiters();
    }
    fn ended(&self) -> bool {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).ended()
    }
    fn on_clear_buffer(&self) -> String {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.superseded_wake.notify_waiters();
        if let Some(cb) = &self.on_clear {
            cb();
        }
        "ok".into()
    }
    async fn recv(self: Arc<Self>) -> std::result::Result<(), ReceiverEnd> {
        loop {
            let wake = self.streams_ready.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            let next = self.streams.lock().unwrap_or_else(|p| p.into_inner()).pop_front();
            let Some(mut stream) = next else {
                if self.ended() {
                    return Ok(());
                }
                wake.await;
                continue;
            };
            let mut reader = self
                .state
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .begin(&stream.info, self.frame_size_ms);
            let outcome = match &mut reader {
                Err(error) => Err(error.clone()),
                Ok(reader) => async {
                    loop {
                        let superseded = self.superseded_wake.notified();
                        tokio::pin!(superseded);
                        superseded.as_mut().enable();
                        let gave_up = self.state.lock().unwrap_or_else(|p| p.into_inner()).superseded();
                        let idle_until = self.clock.now().saturating_add(STREAM_IDLE_TIMEOUT);
                        let chunk = tokio::select! {
                            biased;
                            c = stream.chunks.next() => Some(c),
                            () = async {}, if gave_up => None,
                            () = &mut superseded => None,
                            () = self.clock.wait_until(idle_until) => None,
                        };
                        match chunk {
                            Some(Ok(Some(data))) => {
                                let keep = self.state.lock().unwrap_or_else(|p| p.into_inner()).chunk(reader, &data);
                                self.out_ready.notify_waiters();
                                if !keep { break; }
                            }
                            Some(Ok(None)) => break,
                            Some(Err(error)) => return Err(ReceiverEnd::BadStream(error.to_string())),
                            None => {
                                let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
                                if state.cleared() {
                                    log::debug!(target: LOG, stream_id = stream.info.id.as_str(); "audio stream ended early after clear_buffer");
                                } else {
                                    let reason = if state.superseded() { "superseded by a newer stream".to_owned() }
                                        else { format!("idle for {}s", STREAM_IDLE_TIMEOUT.as_secs()) };
                                    log::warn!(target: LOG, stream_id = stream.info.id.as_str(); "audio stream ended before its trailer arrived ({reason})");
                                }
                                break;
                            }
                        }
                    }
                    self.state.lock().unwrap_or_else(|p| p.into_inner()).flush(reader);
                    self.out_ready.notify_waiters();
                    Ok(())
                }.await,
            };
            let deadline = self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
            let closed = stream.close(deadline).await;
            let outcome = match (outcome, closed) {
                (outcome, lk_agents_abi::Closed::Clean) => outcome,
                (outcome, cleanup) => Err(ReceiverEnd::BadStream(format!(
                    "{}stream cleanup failed: {cleanup:?}",
                    outcome.err().map(|e| format!("{e}; ")).unwrap_or_default()
                ))),
            };
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.finish(if outcome.is_ok() { reader.as_mut().ok() } else { None });
            drop(state);
            outcome?;
            self.out_ready.notify_waiters();
        }
    }

    /// `_send_task`: the playback reports, retried as the reference does.
    async fn send(self: Arc<Self>) -> std::result::Result<(), ReceiverEnd> {
        let Some(mut rx) = self.rpc_rx.lock().unwrap_or_else(|p| p.into_inner()).take() else {
            return Ok(());
        };
        let remote =
            self.remote.lock().unwrap_or_else(|p| p.into_inner()).clone().unwrap_or_default();
        while let Some(ev) = rx.recv().await {
            let (method, payload) = match ev {
                RpcEvent::Finished { segment, playback_position, interrupted } => (
                    playback_method(RPC_PLAYBACK_FINISHED, &segment)?,
                    serde_json::to_string(&FinishedPayload {
                        playback_position,
                        interrupted,
                        synchronized_transcript: None,
                    })
                    .expect("a playback report serializes"),
                ),
                RpcEvent::Started(segment) => {
                    (playback_method(RPC_PLAYBACK_STARTED, &segment)?, String::new())
                }
            };
            let mut retry = 0u32;
            while retry < self.rpc_max_retries {
                log::debug!(target: LOG, payload = payload.as_str(); "notifying {method}");
                let reply = self
                    .room
                    .local()
                    .perform_rpc(
                        ParticipantIdentity(remote.clone()),
                        method.clone(),
                        payload.clone(),
                    )
                    .await
                    .map_err(|e| ReceiverEnd::Failed(e.to_string()))?;
                let (code, message) = reply;
                if message.starts_with("reject:") {
                    return Err(ReceiverEnd::Failed(message));
                }
                if code == 0 {
                    break;
                }
                if code == RPC_UNSUPPORTED_METHOD && method.starts_with(RPC_PLAYBACK_STARTED) {
                    log::error!(target: LOG, "remote participant didn't register lk.playback_started RPC");
                    break;
                }
                if retry == self.rpc_max_retries - 1 {
                    log::error!(target: LOG, "failed to call {method} after {} retries", retry + 1);
                    return Err(ReceiverEnd::Failed(format!("RpcError {code}: {message}")));
                }
                retry += 1;
                log::warn!(target: LOG, "failed to call {method}, retrying...");
                let at = self.clock.now().saturating_add(Duration::from_millis(100));
                self.clock.wait_until(at).await;
            }
        }
        Ok(())
    }
}

/// `DataStreamAudioReceiver`: the avatar worker's side of `DataStreamAudioOutput`. Reads the
/// agent's segments off `lk.audio_stream`, answers `lk.clear_buffer`, and reports playback.
pub struct DataStreamAudioReceiver {
    inner: Arc<RxInner>,
    spawner: Arc<dyn Spawner>,
    tasks: Mutex<Vec<AbortHandle>>,
}

impl Drop for DataStreamAudioReceiver {
    fn drop(&mut self) {
        for t in self.tasks.lock().unwrap_or_else(|p| p.into_inner()).drain(..) {
            t.abort();
        }
    }
}

impl DataStreamAudioReceiver {
    /// `room` is the avatar's `rtc.Room`; `on_clear_buffer` hears the agent clear.
    pub fn for_host(
        room: Arc<dyn RtcRoom>,
        sender_identity: Option<String>,
        frame_size_ms: u32,
        rpc_max_retries: u32,
        on_clear_buffer: Option<Arc<dyn ClearBufferListener>>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        let on_clear = on_clear_buffer
            .map(|l| Arc::new(move || l.clear_buffer()) as Arc<dyn Fn() + Send + Sync>);
        let frame_size_ms = frame_size_ms.max(1);
        DataStreamAudioReceiver::new(
            room,
            sender_identity,
            frame_size_ms,
            rpc_max_retries,
            on_clear,
            clock,
            spawner,
        )
    }

    /// `sender_identity` `None` follows the first agent participant; `on_clear` is the
    /// `clear_buffer` event the avatar runner listens to.
    pub fn new(
        room: Arc<dyn RtcRoom>,
        sender_identity: Option<String>,
        frame_size_ms: u32,
        rpc_max_retries: u32,
        on_clear: Option<Arc<dyn Fn() + Send + Sync>>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        let (rpc, rpc_rx) = tokio::sync::mpsc::unbounded_channel();
        DataStreamAudioReceiver {
            inner: Arc::new(RxInner {
                room,
                sender_identity,
                frame_size_ms,
                rpc_max_retries,
                clock,
                on_clear,
                remote: Mutex::new(None),
                state: Mutex::default(),
                out_ready: tokio::sync::Notify::new(),
                streams: Mutex::default(),
                streams_ready: tokio::sync::Notify::new(),
                superseded_wake: tokio::sync::Notify::new(),
                rpc,
                rpc_rx: Mutex::new(Some(rpc_rx)),
            }),
            spawner,
            tasks: Mutex::new(Vec::new()),
        }
    }

    /// `start()`: wait for the agent, then serve its streams and the clear RPC.
    pub async fn start(&self) -> std::result::Result<(), ReceiverEnd> {
        let inner = &self.inner;
        let mut events = inner.room.events();
        let remote = loop {
            let found = match &inner.sender_identity {
                Some(id) => inner
                    .room
                    .participant(&ParticipantIdentity(id.clone()))
                    .map_err(|error| ReceiverEnd::Failed(error.to_string()))?
                    .map(|p| p.identity().0.clone()),
                None => inner
                    .room
                    .remote_participants()
                    .map_err(|error| ReceiverEnd::Failed(error.to_string()))?
                    .into_iter()
                    .find(|p| p.kind() == lk_agents_abi::voice::rtc::ParticipantKind::Agent)
                    .map(|p| p.identity().0.clone()),
            };
            if let Some(found) = found {
                break found;
            }
            if events
                .next()
                .await
                .map_err(|error| ReceiverEnd::Failed(error.to_string()))?
                .is_none()
            {
                return Err(ReceiverEnd::Failed("the room closed".into()));
            }
        };
        *inner.remote.lock().unwrap_or_else(|p| p.into_inner()) = Some(remote.clone());

        // `_register_clear_buffer_rpc`: routed by caller, as several receivers may share a room
        let room = inner.room.name().map_err(|error| ReceiverEnd::Failed(error.to_string()))?;
        RX_ROUTES
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(room.clone())
            .or_default()
            .insert(remote.clone(), Arc::downgrade(inner));
        let handler = RpcHandler::new(move |caller, _payload| {
            let found = RX_ROUTES
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .get(&room)
                .and_then(|m| m.get(caller))
                .and_then(Weak::upgrade);
            match found {
                Some(rx) => rx.on_clear_buffer(),
                None => {
                    log::warn!(target: LOG, caller_identity = caller;
                        "clear buffer event received from unexpected participant");
                    "reject".to_owned()
                }
            }
        });
        inner
            .room
            .local()
            .register_rpc_method(RPC_CLEAR_BUFFER.to_owned(), handler)
            .map_err(|e| ReceiverEnd::Failed(e.to_string()))?;

        let mut streams = inner
            .room
            .byte_streams(AUDIO_STREAM_TOPIC)
            .map_err(|e| ReceiverEnd::Failed(e.to_string()))?;
        let pump = {
            let inner = Arc::clone(inner);
            async move {
                loop {
                    let stream = match streams.next().await {
                        Ok(Some(stream)) => stream,
                        Ok(None) => return,
                        Err(error) => {
                            inner.end(ReceiverEnd::BadStream(error.to_string()));
                            return;
                        }
                    };
                    if stream.sender.0 != remote {
                        let deadline =
                            inner.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
                        if stream.close(deadline).await != lk_agents_abi::Closed::Clean {
                            inner.end(ReceiverEnd::Failed("foreign stream cleanup failed".into()));
                            return;
                        }
                        continue;
                    }
                    inner.streams.lock().unwrap_or_else(|p| p.into_inner()).push_back(stream);
                    if inner.state.lock().unwrap_or_else(|p| p.into_inner()).stream_admitted() {
                        inner.superseded_wake.notify_waiters();
                    }
                    inner.streams_ready.notify_waiters();
                }
            }
        };
        let main = {
            let inner = Arc::clone(inner);
            async move {
                let recv = Arc::clone(&inner).recv();
                let send = Arc::clone(&inner).send();
                let result = tokio::try_join!(recv, send);
                inner.end(match result {
                    Ok(_) => ReceiverEnd::Closed,
                    Err(e) => e,
                });
            }
        };
        let mut tasks = self.tasks.lock().unwrap_or_else(|p| p.into_inner());
        tasks.push(self.spawner.spawn_unboxed(pump));
        tasks.push(self.spawner.spawn_unboxed(main));
        Ok(())
    }

    pub fn notify_playback_finished(
        &self,
        segment: AvatarPlaybackSegment,
        playback_position: f64,
        interrupted: bool,
    ) -> std::result::Result<(), ReceiverEnd> {
        playback_method(RPC_PLAYBACK_FINISHED, &segment)?;
        self.inner
            .rpc
            .send(RpcEvent::Finished { segment, playback_position, interrupted })
            .map_err(|_| ReceiverEnd::Closed)
    }

    pub fn notify_playback_started(
        &self,
        segment: AvatarPlaybackSegment,
    ) -> std::result::Result<(), ReceiverEnd> {
        playback_method(RPC_PLAYBACK_STARTED, &segment)?;
        self.inner.rpc.send(RpcEvent::Started(segment)).map_err(|_| ReceiverEnd::Closed)
    }

    /// The next frame or segment end; the end once closed or failed.
    pub async fn recv(
        &self,
    ) -> std::result::Result<
        (u64, bool, Option<AvatarPlaybackSegment>, Option<Received>),
        ReceiverEnd,
    > {
        loop {
            let wake = self.inner.out_ready.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            if let Some(item) = self.inner.state.lock().unwrap_or_else(|p| p.into_inner()).take()? {
                return Ok(item);
            }
            wake.await;
        }
    }

    pub fn aclose(&self) {
        self.inner.end(ReceiverEnd::Closed);
        for t in self.tasks.lock().unwrap_or_else(|p| p.into_inner()).drain(..) {
            t.abort();
        }
    }
}

type RxRoutes = HashMap<String, HashMap<String, Weak<RxInner>>>;
static RX_ROUTES: LazyLock<Mutex<RxRoutes>> = LazyLock::new(Mutex::default);

/// The join wait uses the supplied room and clock; no executor lives in the avatar core.
pub async fn wait_avatar_join_room(
    room: Arc<dyn RtcRoom>,
    identity: String,
    clock: Arc<dyn Clock>,
) -> std::result::Result<(f64, f64), lk_agents_abi::voice::rtc::RtcError> {
    let mut events = room.events();
    while room.connection_state()? == lk_agents_abi::voice::rtc::ConnectionState::Disconnected {
        if events.next().await?.is_none() {
            return Err(lk_agents_abi::voice::rtc::RtcError::NotConnected);
        }
    }
    let started = lk_agents_abi::unix_secs(clock.wall());
    room.wait_for_participant(Some(identity.clone()), None, true).await?;
    let video = Some(vec![lk_agents_abi::voice::rtc::TRACK_KIND_VIDEO]);
    room.wait_for_track_publication(Some(identity), video, true, false).await?;
    Ok((started, lk_agents_abi::unix_secs(clock.wall())))
}

#[cfg(test)]
#[path = "receiver_tests.rs"]
mod tests;
