//! The avatar worker's audio receiver (`DataStreamAudioReceiver`): executes the pure receive
//! state's streams and reports, and implements its ABI class and host receiver directly.

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::utils::video::AudioReceiverFrame;
use lk_agents_abi::voice::avatar::{self, AvatarAudioReceiver, ClearBufferListener};
use lk_agents_abi::voice::io::AvatarPlaybackSegment;
use lk_agents_abi::voice::rtc::{ParticipantIdentity, RpcHandler, RtcRoom};
use lk_agents_abi::{AbortHandle, BoxFuture, Clock, Spawner, SpawnerExt};
use lk_agents_net::env::ProcessEnv;
use lk_agents_voice::avatar::receiver::{PlaybackReport, ReceiverState, ReportStep, frame_size};
use lk_agents_voice::avatar::{
    AUDIO_STREAM_TOPIC, LOG, RPC_CLEAR_BUFFER, Received, ReceiverEnd, STREAM_IDLE_TIMEOUT,
};
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::{Arc, LazyLock, Mutex, Weak};

struct RxInner {
    room: Arc<dyn RtcRoom>,
    sender_identity: Option<String>,
    frame_size: NonZeroU32,
    rpc_max_retries: u32,
    clock: Arc<dyn Clock>,
    on_clear: Option<Arc<dyn Fn() + Send + Sync>>,
    state: Mutex<ReceiverState>,
    out_ready: tokio::sync::Notify,
    streams: Mutex<std::collections::VecDeque<lk_agents_abi::voice::rtc::ByteStream>>,
    streams_ready: tokio::sync::Notify,
    superseded_wake: tokio::sync::Notify,
    rpc: tokio::sync::mpsc::UnboundedSender<PlaybackReport>,
    rpc_rx: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<PlaybackReport>>>,
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
                .begin(&stream.info, self.frame_size);
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

    /// `_send_task`: the playback reports, in order, each retried by its attempt policy.
    async fn send(self: Arc<Self>) -> std::result::Result<(), ReceiverEnd> {
        let Some(mut rx) = self.rpc_rx.lock().unwrap_or_else(|p| p.into_inner()).take() else {
            return Ok(());
        };
        let remote = self
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .sender()
            .unwrap_or_default()
            .to_owned();
        while let Some(report) = rx.recv().await {
            let mut attempts = report.attempts(self.rpc_max_retries);
            while attempts.pending() {
                log::debug!(target: LOG, payload = report.payload.as_str(); "notifying {}", report.method);
                let (code, message) = self
                    .room
                    .local()
                    .perform_rpc(
                        ParticipantIdentity(remote.clone()),
                        report.method.clone(),
                        report.payload.clone(),
                    )
                    .await
                    .map_err(|e| ReceiverEnd::Failed(e.to_string()))?;
                match attempts.reply(&report.method, code, &message) {
                    ReportStep::Sent => break,
                    ReportStep::Failed(end) => return Err(end),
                    ReportStep::RetryAfter(delay) => {
                        self.clock.wait_until(self.clock.now().saturating_add(delay)).await;
                    }
                }
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
        self.aclose();
    }
}

impl DataStreamAudioReceiver {
    /// `sender_identity` `None` follows the first agent participant; `on_clear` is the
    /// `clear_buffer` event the avatar runner listens to.
    pub fn new(
        room: Arc<dyn RtcRoom>,
        sender_identity: Option<String>,
        frame_size: NonZeroU32,
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
                frame_size,
                rpc_max_retries,
                clock,
                on_clear,
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
            let participants = inner
                .room
                .remote_participants()
                .map_err(|error| ReceiverEnd::Failed(error.to_string()))?;
            let chosen = inner.state.lock().unwrap_or_else(|p| p.into_inner()).choose_sender(
                inner.sender_identity.as_deref(),
                participants.iter().map(|p| (p.identity().0.as_str(), p.kind())),
            );
            if let Some(chosen) = chosen {
                break chosen;
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
                    let admitted = inner
                        .state
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .admits(&stream.sender.0);
                    if !admitted {
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
        let report = PlaybackReport::finished(&segment, playback_position, interrupted)?;
        self.inner.rpc.send(report).map_err(|_| ReceiverEnd::Closed)
    }

    pub fn notify_playback_started(
        &self,
        segment: AvatarPlaybackSegment,
    ) -> std::result::Result<(), ReceiverEnd> {
        let report = PlaybackReport::started(&segment)?;
        self.inner.rpc.send(report).map_err(|_| ReceiverEnd::Closed)
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

type AudioReceipt = (u64, bool, Option<AvatarPlaybackSegment>, Option<AudioReceiverFrame>);

fn failure(error: ReceiverEnd) -> ProviderError {
    ApiError::new(error.to_string()).into()
}

impl avatar::DataStreamAudioReceiver for DataStreamAudioReceiver {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        room: Arc<dyn RtcRoom>,
        sender_identity: Option<String>,
        frame_size_ms: u32,
        rpc_max_retries: u32,
        on_clear_buffer: Option<Arc<dyn ClearBufferListener>>,
    ) -> Result<Self, ProviderError> {
        let frame_size = frame_size(frame_size_ms).map_err(failure)?;
        let spawner = (env.spawner)().map_err(ApiError::new)?;
        let on_clear = on_clear_buffer.map(|listener| {
            Arc::new(move || listener.clear_buffer()) as Arc<dyn Fn() + Send + Sync>
        });
        Ok(DataStreamAudioReceiver::new(
            crate::rtc::room::Room::wrap(room),
            sender_identity,
            frame_size,
            rpc_max_retries,
            on_clear,
            env.clock.clone(),
            spawner,
        ))
    }

    async fn start(&self) -> Result<(), ProviderError> {
        DataStreamAudioReceiver::start(self).await.map_err(failure)
    }

    async fn next_audio(&self) -> Result<AudioReceipt, ProviderError> {
        match self.recv().await {
            Ok(receipt) => Ok(receipt),
            Err(ReceiverEnd::Closed) => Ok((0, false, None, None)),
            Err(error) => Err(failure(error)),
        }
    }

    fn notify_playback_started(&self, segment: AvatarPlaybackSegment) -> Result<(), ProviderError> {
        DataStreamAudioReceiver::notify_playback_started(self, segment).map_err(failure)
    }

    fn notify_playback_finished(
        &self,
        segment: AvatarPlaybackSegment,
        playback_position: f64,
        interrupted: bool,
    ) -> Result<(), ProviderError> {
        DataStreamAudioReceiver::notify_playback_finished(
            self,
            segment,
            playback_position,
            interrupted,
        )
        .map_err(failure)
    }

    fn aclose(&self) -> Result<(), ProviderError> {
        DataStreamAudioReceiver::aclose(self);
        Ok(())
    }
}

impl AvatarAudioReceiver for DataStreamAudioReceiver {
    fn open_audio(&self) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(avatar::DataStreamAudioReceiver::start(self))
    }
    fn next_audio(&self) -> BoxFuture<'_, Result<AudioReceipt, ProviderError>> {
        Box::pin(avatar::DataStreamAudioReceiver::next_audio(self))
    }
    fn notify_playback_started(
        &self,
        segment: AvatarPlaybackSegment,
    ) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(
            async move { avatar::DataStreamAudioReceiver::notify_playback_started(self, segment) },
        )
    }
    fn notify_playback_finished(
        &self,
        segment: AvatarPlaybackSegment,
        playback_position: f64,
        interrupted: bool,
    ) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move {
            avatar::DataStreamAudioReceiver::notify_playback_finished(
                self,
                segment,
                playback_position,
                interrupted,
            )
        })
    }
    fn aclose(&self) -> BoxFuture<'_, Result<(), ProviderError>> {
        Box::pin(async move { avatar::DataStreamAudioReceiver::aclose(self) })
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
