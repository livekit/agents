//! `voice/avatar/_datastream_io.py`'s `DataStreamAudioOutput`: the agent's audio, streamed to a
//! remote avatar worker as one byte stream per segment, with the worker reporting playback back
//! over RPC. The wire (topic, RPC methods, attributes) is the reference's, byte for byte.

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::HashMap;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::voice::io::{AudioOutput, IoError, PlaybackFinished, PlaybackTracker, Result};
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AbortHandle, AudioFrame, Clock, Spawner, SpawnerExt, Ts, unix_secs};
use lk_agents_rtc::{
    ByteStreamWriter, ParticipantIdentity, RoomEvent, RpcHandler, RtcRoom, StreamBytesOptions,
};
use tokio::sync::watch;

pub const RPC_CLEAR_BUFFER: &str = "lk.clear_buffer";
pub const RPC_PLAYBACK_FINISHED: &str = "lk.playback_finished";
pub const RPC_PLAYBACK_STARTED: &str = "lk.playback_started";
pub const AUDIO_STREAM_TOPIC: &str = "lk.audio_stream";

const LOG: &str = "livekit.agents";

/// `utils.shortuuid("AUDIO_")`.
fn stream_name() -> String {
    lk_agents_abi::utils::strings::shortuuid("AUDIO_")
}

// ----------------------------------------------------------------------------- the RPC routes

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
/// The reference's class-level handler tables: one RPC method per room serves every output on it,
/// routed by the caller (the avatar each output streams to).
type Routes = HashMap<(String, &'static str), HashMap<String, Weak<Inner>>>;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
static ROUTES: LazyLock<Mutex<Routes>> = LazyLock::new(Mutex::default);

fn route(room: &str, method: &'static str, caller: &str) -> Option<Arc<Inner>> {
    let routes = ROUTES.lock().unwrap_or_else(|p| p.into_inner());
    routes.get(&(room.to_owned(), method))?.get(caller)?.upgrade()
}

fn register(inner: &Arc<Inner>, method: &'static str) {
    let room = inner.room.name();
    ROUTES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .entry((room.clone(), method))
        .or_default()
        .insert(inner.destination.clone(), Arc::downgrade(inner));
    let handler = RpcHandler::new(move |caller, payload| match route(&room, method, caller) {
        Some(inner) => match method {
            RPC_PLAYBACK_FINISHED => inner.on_playback_finished(payload),
            _ => inner.on_playback_started(),
        },
        None => {
            log::warn!(target: LOG, caller_identity = caller, method = method;
                "playback event received from unexpected participant");
            "reject".to_owned()
        }
    });
    if let Err(e) = inner.room.local().register_rpc_method(method.to_owned(), handler) {
        log::warn!(target: LOG, method = method, error = e.to_string().as_str(); "failed to register rpc method");
    }
}

// ----------------------------------------------------------------------------- the sink

/// `lk.playback_finished`'s payload, sent by the avatar worker's receiver.
#[derive(serde::Serialize, serde::Deserialize)]
struct FinishedPayload {
    #[serde(default)]
    playback_position: f64,
    #[serde(default)]
    interrupted: bool,
    #[serde(default)]
    synchronized_transcript: Option<String>,
}

struct Inner {
    room: Arc<dyn RtcRoom>,
    destination: String,
    wait_remote_track: Option<i32>,
    wait_playback_start: bool,
    clear_buffer_timeout: Option<Duration>,
    sample_rate: Option<u32>,
    tracker: Arc<PlaybackTracker>,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    /// `None` while starting, then whether the avatar became ready.
    started: watch::Sender<Option<std::result::Result<(), String>>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// The open segment. Its stream is opened, written and closed by tasks of their own, so a
    /// capture the playout drops mid-write leaves the stream to the segment's close.
    segment: Mutex<Option<Arc<Segment>>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pushed: Mutex<Duration>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// The give-up timer after a clear, and when it fires.
    clear_timer: Mutex<Option<(AbortHandle, Ts)>>,
}

/// One segment's byte stream: opened by its first frame, in order behind the async lock.
#[derive(Default)]
struct Segment {
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    writer: tokio::sync::Mutex<Option<Box<dyn ByteStreamWriter>>>,
}

impl Inner {
    fn is_started(&self) -> bool {
        matches!(*self.started.borrow(), Some(Ok(())))
    }

    /// `_start_task`: the room connected, the RPC methods registered, the avatar in the room (and
    /// its track published, when asked).
    async fn start(self: Arc<Self>) {
        let mut events = self.room.events();
        while !self.room.connection_state().is_connected() {
            if events.recv().await.is_none() {
                self.started.send_replace(Some(Err("the room closed before it connected".into())));
                return;
            }
        }
        register(&self, RPC_PLAYBACK_FINISHED);
        if self.wait_playback_start {
            register(&self, RPC_PLAYBACK_STARTED);
        }
        let identity = ParticipantIdentity(self.destination.clone());
        log::debug!(target: LOG, identity = self.destination.as_str(); "waiting for the remote participant");
        while self.room.participant(&identity).is_none() {
            if events.recv().await.is_none() {
                self.started.send_replace(Some(Err("the room closed".into())));
                return;
            }
        }
        if let Some(kind) = self.wait_remote_track {
            log::debug!(target: LOG, identity = self.destination.as_str(), kind = kind; "waiting for the remote track");
            loop {
                let published = self
                    .room
                    .participant(&identity)
                    .is_some_and(|p| p.track_publications().iter().any(|t| t.kind() == kind));
                if published {
                    break;
                }
                match events.recv().await {
                    Some(
                        RoomEvent::TrackPublished { .. } | RoomEvent::ParticipantConnected { .. },
                    ) => {}
                    Some(_) => {}
                    None => {
                        self.started.send_replace(Some(Err("the room closed".into())));
                        return;
                    }
                }
            }
        }
        log::debug!(target: LOG, identity = self.destination.as_str(); "remote participant ready");
        self.started.send_replace(Some(Ok(())));
    }

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// One frame into `segment`, opening its stream on the first.
    async fn write(&self, segment: &Segment, frame: &AudioFrame) -> Result<()> {
        let mut writer = segment.writer.lock().await;
        if writer.is_none() {
            let opts = StreamBytesOptions {
                name: stream_name(),
                topic: AUDIO_STREAM_TOPIC.to_owned(),
                destination_identities: vec![ParticipantIdentity(self.destination.clone())],
                attributes: HashMap::from([
                    ("sample_rate".to_owned(), frame.sample_rate.to_string()),
                    ("num_channels".to_owned(), frame.num_channels.to_string()),
                ]),
            };
            *writer = Some(self.room.local().stream_bytes(opts).await?);
            *self.pushed.lock().unwrap_or_else(|p| p.into_inner()) = Duration::ZERO;
            if !self.wait_playback_start {
                // approximated: the avatar does not report lk.playback_started
                self.tracker.report_started(unix_secs(self.clock.wall()));
            }
        }
        if let Some(writer) = writer.as_ref() {
            writer.write(&frame.data).await?;
        }
        *self.pushed.lock().unwrap_or_else(|p| p.into_inner()) +=
            Duration::from_secs_f64(frame.duration());
        Ok(())
    }

    /// `_handle_playback_finished`.
    fn on_playback_finished(&self, payload: &str) -> String {
        log::info!(target: LOG, caller_identity = self.destination.as_str(); "playback finished event received");
        if let Some((timer, _)) = self.clear_timer.lock().unwrap_or_else(|p| p.into_inner()).take()
        {
            timer.abort();
        }
        let finished = serde_json::from_str::<FinishedPayload>(payload)
            .map_err(|e| e.to_string())
            .and_then(|ev| {
                let position = Duration::try_from_secs_f64(ev.playback_position)
                    .map_err(|_| format!("playback_position={}", ev.playback_position))?;
                Ok(PlaybackFinished {
                    playback_position: position,
                    interrupted: ev.interrupted,
                    synchronized_transcript: ev.synchronized_transcript,
                })
            });
        match finished {
            Ok(ev) => {
                self.tracker.playback_finished(ev);
            }
            Err(e) => {
                log::error!(target: LOG, caller_identity = self.destination.as_str(), error = e.as_str();
                    "invalid playback finished event");
                return "reject".to_owned();
            }
        }
        "ok".to_owned()
    }

    /// `_handle_playback_started`.
    fn on_playback_started(&self) -> String {
        self.tracker.report_started(unix_secs(self.clock.wall()));
        "ok".to_owned()
    }

    /// `_clear_buffer_task`: ask the avatar to clear, and mark the segment done if it never says
    /// it finished.
    async fn clear(self: Arc<Self>, pushed: Duration) {
        let rpc = self
            .room
            .local()
            .perform_rpc(
                ParticipantIdentity(self.destination.clone()),
                RPC_CLEAR_BUFFER.to_owned(),
                String::new(),
            )
            .await;
        let failed = match rpc {
            Ok((0, _)) => None,
            Ok((code, message)) => Some(format!("{code}: {message}")),
            Err(e) => Some(e.to_string()),
        };
        // a failed clear marks the playout done at once
        let timeout = match failed {
            Some(e) => {
                log::error!(target: LOG, error = e.as_str(); "failed to perform clear buffer rpc");
                Some(Duration::ZERO)
            }
            None => self.clear_buffer_timeout,
        };
        if let Some((timer, _)) = self.clear_timer.lock().unwrap_or_else(|p| p.into_inner()).take()
        {
            timer.abort();
        }
        let Some(timeout) = timeout else { return };
        let me = Arc::clone(&self);
        let at = self.clock.now().saturating_add(timeout);
        let timer = self.spawner.spawn_unboxed(async move {
            me.clock.wait_until(at).await;
            me.clear_timer.lock().unwrap_or_else(|p| p.into_inner()).take();
            log::warn!(target: LOG,
                "didn't receive playback finished event after clear buffer, marking playout as done arbitrarily");
            me.tracker.playback_finished(PlaybackFinished {
                playback_position: pushed,
                interrupted: true,
                synchronized_transcript: None,
            });
            me.tracker.reset();
        });
        *self.clear_timer.lock().unwrap_or_else(|p| p.into_inner()) = Some((timer, at));
    }
}

/// `DataStreamAudioOutput`: streams the agent's audio to a remote avatar worker over LiveKit
/// DataStream, which reports its playback back.
pub struct DataStreamAudioOutput {
    inner: Arc<Inner>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    start: Mutex<Option<AbortHandle>>,
}

impl Drop for DataStreamAudioOutput {
    fn drop(&mut self) {
        if let Some(h) = self.start.lock().unwrap_or_else(|p| p.into_inner()).take() {
            h.abort();
        }
    }
}

impl DataStreamAudioOutput {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub fn new(
        room: Arc<dyn RtcRoom>,
        destination_identity: String,
        sample_rate: Option<u32>,
        wait_remote_track: Option<i32>,
        clear_buffer_timeout: Option<Duration>,
        wait_playback_start: bool,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        let inner = Arc::new(Inner {
            room,
            destination: destination_identity,
            wait_remote_track,
            wait_playback_start,
            clear_buffer_timeout,
            sample_rate,
            tracker: Arc::new(PlaybackTracker::new()),
            clock,
            spawner: Arc::clone(&spawner),
            started: watch::channel(None).0,
            segment: Mutex::new(None),
            pushed: Mutex::new(Duration::ZERO),
            clear_timer: Mutex::new(None),
        });
        let start = spawner.spawn_unboxed(Arc::clone(&inner).start());
        DataStreamAudioOutput { inner, start: Mutex::new(Some(start)) }
    }
}

#[lk_agents_abi::host(object(name = "DataStreamAudioOutput", cell = "arc", as_dyn(AudioOutput = as_output)))]
impl DataStreamAudioOutput {
    /// `destination_identity` is the avatar's participant; `wait_remote_track` a `rtc.TrackKind`
    /// it must publish before audio flows; `clear_buffer_timeout` how long an interruption waits
    /// for the avatar's `lk.playback_finished` (`None`: forever); `wait_playback_start` whether the
    /// avatar reports `lk.playback_started` (else the first frame of a segment stands for it).
    #[host(new(
        kw_only = "destination_identity",
        defaults(clear_buffer_timeout = Some(2.0), wait_playback_start = false)
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn for_host(
        room: Arc<dyn RtcRoom>,
        destination_identity: String,
        sample_rate: Option<u32>,
        wait_remote_track: Option<i32>,
        clear_buffer_timeout: Option<f64>,
        wait_playback_start: bool,
        #[host(ambient)] clock: Arc<dyn Clock>,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Self {
        let timeout = clear_buffer_timeout.map(|s| Duration::from_secs_f64(s.max(0.0)));
        DataStreamAudioOutput::new(
            room,
            destination_identity,
            sample_rate,
            wait_remote_track,
            timeout,
            wait_playback_start,
            clock,
            spawner,
        )
    }

    /// When the give-up timer after a clear fires, while one is armed.
    pub fn wake_at(&self) -> Option<Ts> {
        self.inner.clear_timer.lock().unwrap_or_else(|p| p.into_inner()).as_ref().map(|(_, at)| *at)
    }

    /// The sink as the session's audio output (a native sink is not proxied).
    pub fn as_output(self: &Arc<Self>) -> Arc<dyn AudioOutput> {
        Arc::clone(self) as Arc<dyn AudioOutput>
    }

    #[host(getter(name = "label"))]
    pub fn label_for_host(&self) -> String {
        "DataStreamIO".to_owned()
    }

    #[host(getter(name = "sample_rate"))]
    pub fn sample_rate_for_host(&self) -> Option<u32> {
        self.inner.sample_rate
    }
}

#[async_trait]
impl AudioOutput for DataStreamAudioOutput {
    fn label(&self) -> &str {
        "DataStreamIO"
    }

    fn tracker(&self) -> &Arc<PlaybackTracker> {
        &self.inner.tracker
    }

    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: false }
    }

    fn sample_rate(&self) -> Option<u32> {
        self.inner.sample_rate
    }

    /// `capture_frame`: wait for the avatar, open the segment's stream on its first frame.
    async fn capture(&self, frame: &AudioFrame) -> Result<()> {
        let mut started = self.inner.started.subscribe();
        let ready = started.wait_for(Option::is_some).await.map(|s| s.clone()).ok().flatten();
        match ready {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(IoError::Rtc(e)),
            None => return Err(IoError::Rtc("the avatar output closed".into())),
        }
        let segment = Arc::clone(
            self.inner
                .segment
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .get_or_insert_with(Default::default),
        );
        // Spawned, so dropping this capture (an interruption) never drops the host's writer.
        let (tx, rx) = tokio::sync::oneshot::channel();
        let inner = Arc::clone(&self.inner);
        let frame = frame.clone();
        self.inner.spawner.spawn_unboxed(async move {
            let _ = tx.send(inner.write(&segment, &frame).await);
        });
        rx.await.unwrap_or_else(|_| Err(IoError::Rtc("the avatar output closed".into())))
    }

    /// `flush`: close the segment's stream, its trailer ending the segment on the avatar.
    fn flush_segment(&self) {
        if !self.inner.is_started() {
            return;
        }
        if let Some(segment) = self.inner.segment.lock().unwrap_or_else(|p| p.into_inner()).take() {
            // Behind any write still in flight, whose stream it then closes.
            self.inner.spawner.spawn_unboxed(async move {
                if let Some(writer) = segment.writer.lock().await.take() {
                    let _ = writer.close().await;
                }
            });
        }
    }

    /// Once per pushed segment: the interruption's cut and the playout's both clear, and a second
    /// `lk.clear_buffer` with nothing pushed since gets no `lk.playback_finished`, so its timer
    /// would mark a later segment done.
    fn clear_buffer(&self) {
        if !self.inner.is_started() {
            return;
        }
        let pushed =
            std::mem::take(&mut *self.inner.pushed.lock().unwrap_or_else(|p| p.into_inner()));
        if pushed.is_zero() {
            return;
        }
        self.inner.spawner.spawn_unboxed(Arc::clone(&self.inner).clear(pushed));
    }

    fn pause(&self) {
        log::warn!(target: LOG,
            "pause is not supported by DataStreamAudioOutput, disable `VoiceSession.resume_false_interruption` if you are using an avatar plugin.");
    }

    fn resume(&self) {}
}

// ----------------------------------------------------------------------------- the avatar's side

/// What the avatar worker reads: audio ([`AvatarFrame::Audio`]), and where each segment ends
/// ([`AvatarFrame::SegmentEnd`]).
pub type Received = lk_agents_abi::utils::video::AvatarFrame;

/// Why reading stopped: closed by us (the end of iteration), or failed (the reference re-raises it
/// from iteration).
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReceiverEnd {
    #[host(kind = "StopAsyncIteration")]
    #[error("the receiver is closed")]
    Closed,
    /// A stream without `sample_rate`/`num_channels`: the reference's `ValueError`.
    #[host(kind = "ValueError")]
    #[error("{0}")]
    BadStream(String),
    #[error("{0}")]
    Failed(String),
}

pub use lk_agents_abi::voice::avatar::ClearBufferListener;

/// `STREAM_IDLE_TIMEOUT`: with no chunk or trailer for this long, the last segment's trailer is
/// assumed lost.
pub const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(10);

/// `rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD`.
const RPC_UNSUPPORTED_METHOD: u32 = 1400;

enum RpcEvent {
    Finished { playback_position: f64, interrupted: bool },
    Started,
}

#[derive(Default)]
struct RxOut {
    items: std::collections::VecDeque<Received>,
    end: Option<ReceiverEnd>,
}

struct RxInner {
    room: Arc<dyn RtcRoom>,
    sender_identity: Option<String>,
    frame_size_ms: u32,
    rpc_max_retries: u32,
    clock: Arc<dyn Clock>,
    on_clear: Option<Arc<dyn Fn() + Send + Sync>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    remote: Mutex<Option<String>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    out: Mutex<RxOut>,
    out_ready: tokio::sync::Notify,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    streams: Mutex<std::collections::VecDeque<lk_agents_rtc::ByteStream>>,
    streams_ready: tokio::sync::Notify,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// A segment is being read.
    reading: std::sync::atomic::AtomicBool,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// `_current_reader_cleared`.
    cleared: std::sync::atomic::AtomicBool,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// `_current_reader_superseded`: a newer stream or a clear means its trailer no longer matters.
    superseded: std::sync::atomic::AtomicBool,
    superseded_wake: tokio::sync::Notify,
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    rpc: tokio::sync::mpsc::UnboundedSender<RpcEvent>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    rpc_rx: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<RpcEvent>>>,
}

impl RxInner {
    fn push(&self, item: Received) {
        let mut out = self.out.lock().unwrap_or_else(|p| p.into_inner());
        if out.end.is_none() {
            out.items.push_back(item);
        }
        drop(out);
        self.out_ready.notify_waiters();
    }

    fn drain(&self) {
        self.out.lock().unwrap_or_else(|p| p.into_inner()).items.clear();
    }

    fn end(&self, end: ReceiverEnd) {
        let mut out = self.out.lock().unwrap_or_else(|p| p.into_inner());
        out.end.get_or_insert(end);
        drop(out);
        self.out_ready.notify_waiters();
        self.streams_ready.notify_waiters();
    }

    fn ended(&self) -> bool {
        self.out.lock().unwrap_or_else(|p| p.into_inner()).end.is_some()
    }

    fn supersede(&self) {
        self.superseded.store(true, std::sync::atomic::Ordering::Release);
        self.superseded_wake.notify_waiters();
    }

    /// `_handle_clear_buffer`.
    fn on_clear_buffer(&self) -> String {
        if self.reading.load(std::sync::atomic::Ordering::Acquire) {
            self.cleared.store(true, std::sync::atomic::Ordering::Release);
            // post-clear data is discarded, so don't wait for a trailer that may be lost
            self.supersede();
        }
        self.drain();
        if let Some(cb) = &self.on_clear {
            cb();
        }
        "ok".to_owned()
    }

    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
    /// `_recv_task`: each stream is one segment, framed at `frame_size_ms`.
    async fn recv(self: Arc<Self>) -> std::result::Result<(), ReceiverEnd> {
        use std::sync::atomic::Ordering;
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
            self.superseded.store(false, Ordering::Release);
            self.reading.store(true, Ordering::Release);
            let attrs = &stream.info.attributes;
            let (Some(rate), Some(channels)) = (
                attrs.get("sample_rate").and_then(|v| v.parse::<u32>().ok()),
                attrs.get("num_channels").and_then(|v| v.parse::<u16>().ok()),
            ) else {
                return Err(ReceiverEnd::BadStream(
                    "sample_rate or num_channels not found in byte stream".into(),
                ));
            };
            let samples = (u64::from(rate) * u64::from(self.frame_size_ms)).div_ceil(1000) as u32;
            let mut bstream =
                lk_agents_abi::AudioByteStream::new(rate, channels, Some(samples), false);
            let frame = |f: lk_agents_abi::utils::byte_stream::Frame| {
                AudioFrame::new(f.data, rate, channels, f.samples_per_channel)
            };
            loop {
                let superseded = self.superseded_wake.notified();
                tokio::pin!(superseded);
                superseded.as_mut().enable();
                let gave_up = self.superseded.load(Ordering::Acquire);
                let idle_until = self.clock.now().saturating_add(STREAM_IDLE_TIMEOUT);
                // a queued chunk or trailer always wins the race
                let chunk = tokio::select! {
                    biased;
                    c = stream.chunks.recv() => Some(c),
                    () = async {}, if gave_up => None,
                    () = &mut superseded => None,
                    () = self.clock.wait_until(idle_until) => None,
                };
                match chunk {
                    Some(Some(data)) => {
                        if self.cleared.load(Ordering::Acquire) {
                            // the rest of the cleared segment is discarded
                            self.drain();
                            bstream.clear();
                            break;
                        }
                        for f in bstream.push(&data) {
                            self.push(Received::Audio(frame(f)));
                        }
                    }
                    Some(None) => break,
                    None => {
                        if self.cleared.load(Ordering::Acquire) {
                            log::debug!(target: LOG, stream_id = stream.info.id.as_str();
                                "audio stream ended early after clear_buffer");
                        } else {
                            let reason = if self.superseded.load(Ordering::Acquire) {
                                "superseded by a newer stream".to_owned()
                            } else {
                                format!("idle for {}s", STREAM_IDLE_TIMEOUT.as_secs())
                            };
                            log::warn!(target: LOG, stream_id = stream.info.id.as_str();
                                "audio stream ended before its trailer arrived ({reason})");
                        }
                        break;
                    }
                }
            }
            if !self.cleared.load(Ordering::Acquire) {
                for f in bstream.flush() {
                    self.push(Received::Audio(frame(f)));
                }
            }
            self.reading.store(false, Ordering::Release);
            self.cleared.store(false, Ordering::Release);
            self.push(Received::SegmentEnd);
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
                RpcEvent::Finished { playback_position, interrupted } => (
                    RPC_PLAYBACK_FINISHED,
                    serde_json::to_string(&FinishedPayload {
                        playback_position,
                        interrupted,
                        synchronized_transcript: None,
                    })
                    .expect("a playback report serializes"),
                ),
                RpcEvent::Started => (RPC_PLAYBACK_STARTED, String::new()),
            };
            let mut retry = 0u32;
            while retry < self.rpc_max_retries {
                log::debug!(target: LOG, payload = payload.as_str(); "notifying {method}");
                let reply = self
                    .room
                    .local()
                    .perform_rpc(
                        ParticipantIdentity(remote.clone()),
                        method.to_owned(),
                        payload.clone(),
                    )
                    .await
                    .map_err(|e| ReceiverEnd::Failed(e.to_string()))?;
                let (code, message) = reply;
                if code == 0 {
                    break;
                }
                if code == RPC_UNSUPPORTED_METHOD && method == RPC_PLAYBACK_STARTED {
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
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    tasks: Mutex<Vec<AbortHandle>>,
}

impl Drop for DataStreamAudioReceiver {
    fn drop(&mut self) {
        for t in self.tasks.lock().unwrap_or_else(|p| p.into_inner()).drain(..) {
            t.abort();
        }
    }
}

#[lk_agents_abi::host(object(name = "DataStreamAudioReceiver", cell = "arc"))]
impl DataStreamAudioReceiver {
    /// `room` is the avatar's `rtc.Room`; `on_clear_buffer` hears the agent clear.
    #[host(new(kw_only = "sender_identity", defaults(frame_size_ms = 100, rpc_max_retries = 3)))]
    pub fn for_host(
        room: Arc<dyn RtcRoom>,
        sender_identity: Option<String>,
        frame_size_ms: u32,
        rpc_max_retries: u32,
        on_clear_buffer: Option<Arc<dyn ClearBufferListener>>,
        #[host(ambient)] clock: Arc<dyn Clock>,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
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

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
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
                out: Mutex::default(),
                out_ready: tokio::sync::Notify::new(),
                streams: Mutex::default(),
                streams_ready: tokio::sync::Notify::new(),
                reading: Default::default(),
                cleared: Default::default(),
                superseded: Default::default(),
                superseded_wake: tokio::sync::Notify::new(),
                rpc,
                rpc_rx: Mutex::new(Some(rpc_rx)),
            }),
            spawner,
            tasks: Mutex::new(Vec::new()),
        }
    }

    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
    /// `start()`: wait for the agent, then serve its streams and the clear RPC.
    #[host(method)]
    pub async fn start(&self) -> std::result::Result<(), ReceiverEnd> {
        let inner = &self.inner;
        let mut events = inner.room.events();
        let remote = loop {
            let found = match &inner.sender_identity {
                Some(id) => inner
                    .room
                    .participant(&ParticipantIdentity(id.clone()))
                    .map(|p| p.identity().0.clone()),
                None => inner
                    .room
                    .remote_participants()
                    .into_iter()
                    .find(|p| p.kind() == lk_agents_rtc::ParticipantKind::Agent)
                    .map(|p| p.identity().0.clone()),
            };
            if let Some(found) = found {
                break found;
            }
            if events.recv().await.is_none() {
                return Err(ReceiverEnd::Failed("the room closed".into()));
            }
        };
        *inner.remote.lock().unwrap_or_else(|p| p.into_inner()) = Some(remote.clone());

        // `_register_clear_buffer_rpc`: routed by caller, as several receivers may share a room
        let room = inner.room.name();
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
                while let Some(stream) = streams.recv().await {
                    if stream.sender.0 != remote {
                        continue;
                    }
                    inner.streams.lock().unwrap_or_else(|p| p.into_inner()).push_back(stream);
                    if inner.reading.load(std::sync::atomic::Ordering::Acquire) {
                        // a new segment started: the open reader's trailer is not coming
                        inner.supersede();
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

    #[host(method)]
    pub fn notify_playback_finished(&self, playback_position: f64, interrupted: bool) {
        let _ = self.inner.rpc.send(RpcEvent::Finished { playback_position, interrupted });
    }

    #[host(method)]
    pub fn notify_playback_started(&self) {
        let _ = self.inner.rpc.send(RpcEvent::Started);
    }

    /// The next frame or segment end; the end once closed or failed.
    #[host(method)]
    pub async fn recv(&self) -> std::result::Result<Received, ReceiverEnd> {
        loop {
            let wake = self.inner.out_ready.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            {
                let mut out = self.inner.out.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(item) = out.items.pop_front() {
                    return Ok(item);
                }
                if let Some(end) = &out.end {
                    return Err(end.clone());
                }
            }
            wake.await;
        }
    }

    #[host(method)]
    pub fn aclose(&self) {
        self.inner.end(ReceiverEnd::Closed);
        for t in self.tasks.lock().unwrap_or_else(|p| p.into_inner()).drain(..) {
            t.abort();
        }
    }
}

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
type RxRoutes = HashMap<String, HashMap<String, Weak<RxInner>>>;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
static RX_ROUTES: LazyLock<Mutex<RxRoutes>> = LazyLock::new(Mutex::default);

// ----------------------------------------------------------------------------- the avatar's join

/// `AvatarSession`'s join wait: from the room's connection (awaited when it is not connected
/// yet), the avatar participant `identity` and its video track. Returns when the wait started and
/// when the avatar joined, in `time.time()` seconds, for its `AvatarMetrics`.
#[lk_agents_abi::host(export)]
pub async fn wait_avatar_join(
    room: Arc<dyn RtcRoom>,
    identity: String,
    #[host(ambient)] clock: Arc<dyn Clock>,
) -> std::result::Result<(f64, f64), lk_agents_rtc::RtcError> {
    let mut events = room.events();
    while room.connection_state() == lk_agents_rtc::ConnectionState::Disconnected {
        if events.recv().await.is_none() {
            return Err(lk_agents_rtc::RtcError::NotConnected);
        }
    }
    let started = unix_secs(clock.wall());
    lk_agents_rtc::wait_for_participant(Arc::clone(&room), Some(identity.clone()), None, true)
        .await?;
    let video = Some(vec![lk_agents_rtc::TRACK_KIND_VIDEO]);
    lk_agents_rtc::wait_for_track_publication(room, Some(identity), video, true, false).await?;
    Ok((started, unix_secs(clock.wall())))
}

#[cfg(test)]
#[path = "avatar_tests.rs"]
mod tests;
