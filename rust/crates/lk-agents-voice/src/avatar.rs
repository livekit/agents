//! `voice/avatar/_datastream_io.py`'s `DataStreamAudioOutput`: the agent's audio, streamed to a
//! remote avatar worker as one byte stream per segment, with the worker reporting playback back
//! over RPC. The wire (topic, RPC methods, attributes) is the reference's, byte for byte.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::voice::io::{AudioOutput, IoError, PlaybackTracker, Result};
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AbortHandle, AudioFrame, Clock, Spawner, SpawnerExt, unix_secs};
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

/// The reference's class-level handler tables: one RPC method per room serves every output on it,
/// routed by the caller (the avatar each output streams to).
type Routes = HashMap<(String, &'static str), HashMap<String, Weak<Inner>>>;
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
    writer: Mutex<Option<Box<dyn ByteStreamWriter>>>,
    pushed: Mutex<f64>,
    clear_timer: Mutex<Option<AbortHandle>>,
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
                    Some(RoomEvent::TrackPublished { .. } | RoomEvent::ParticipantConnected { .. }) => {}
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

    /// `_handle_playback_finished`.
    fn on_playback_finished(&self, payload: &str) -> String {
        log::info!(target: LOG, caller_identity = self.destination.as_str(); "playback finished event received");
        if let Some(timer) = self.clear_timer.lock().unwrap_or_else(|p| p.into_inner()).take() {
            timer.abort();
        }
        let ev: serde_json::Value = serde_json::from_str(payload).unwrap_or_default();
        let position = ev.get("playback_position").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
        let interrupted = ev.get("interrupted").and_then(serde_json::Value::as_bool).unwrap_or(false);
        let transcript = ev.get("synchronized_transcript").and_then(serde_json::Value::as_str).map(str::to_owned);
        self.tracker.report_finished(position, interrupted, transcript);
        "ok".to_owned()
    }

    /// `_handle_playback_started`.
    fn on_playback_started(&self) -> String {
        self.tracker.report_started(unix_secs(self.clock.wall()));
        "ok".to_owned()
    }

    /// `_clear_buffer_task`: ask the avatar to clear, and mark the segment done if it never says
    /// it finished.
    async fn clear(self: Arc<Self>, pushed: f64) {
        let rpc = self
            .room
            .local()
            .perform_rpc(ParticipantIdentity(self.destination.clone()), RPC_CLEAR_BUFFER.to_owned(), String::new())
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
        if let Some(timer) = self.clear_timer.lock().unwrap_or_else(|p| p.into_inner()).take() {
            timer.abort();
        }
        let Some(timeout) = timeout else { return };
        let me = Arc::clone(&self);
        let timer = self.spawner.spawn_unboxed(async move {
            let at = me.clock.now().saturating_add(timeout);
            me.clock.wait_until(at).await;
            log::warn!(target: LOG,
                "didn't receive playback finished event after clear buffer, marking playout as done arbitrarily");
            me.tracker.report_finished(pushed, true, None);
            me.tracker.reset();
        });
        *self.clear_timer.lock().unwrap_or_else(|p| p.into_inner()) = Some(timer);
    }
}

/// `DataStreamAudioOutput`: streams the agent's audio to a remote avatar worker over LiveKit
/// DataStream, which reports its playback back.
pub struct DataStreamAudioOutput {
    inner: Arc<Inner>,
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
            writer: Mutex::new(None),
            pushed: Mutex::new(0.0),
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
        let writer = self.inner.writer.lock().unwrap_or_else(|p| p.into_inner()).take();
        let writer = match writer {
            Some(w) => w,
            None => {
                let opts = StreamBytesOptions {
                    name: stream_name(),
                    topic: AUDIO_STREAM_TOPIC.to_owned(),
                    destination_identities: vec![ParticipantIdentity(self.inner.destination.clone())],
                    attributes: HashMap::from([
                        ("sample_rate".to_owned(), frame.sample_rate.to_string()),
                        ("num_channels".to_owned(), frame.num_channels.to_string()),
                    ]),
                };
                let w = self.inner.room.local().stream_bytes(opts).await?;
                *self.inner.pushed.lock().unwrap_or_else(|p| p.into_inner()) = 0.0;
                if !self.inner.wait_playback_start {
                    // approximated: the avatar does not report lk.playback_started
                    self.inner.tracker.report_started(unix_secs(self.inner.clock.wall()));
                }
                w
            }
        };
        let written = writer.write(&frame.data).await;
        *self.inner.writer.lock().unwrap_or_else(|p| p.into_inner()) = Some(writer);
        written?;
        *self.inner.pushed.lock().unwrap_or_else(|p| p.into_inner()) += frame.duration();
        Ok(())
    }

    /// `flush`: close the segment's stream, its trailer ending the segment on the avatar.
    fn flush_segment(&self) {
        if !self.inner.is_started() {
            return;
        }
        if let Some(writer) = self.inner.writer.lock().unwrap_or_else(|p| p.into_inner()).take() {
            self.inner.spawner.spawn_unboxed(async move {
                let _ = writer.close().await;
            });
        }
    }

    fn clear_buffer(&self) {
        if !self.inner.is_started() {
            return;
        }
        let pushed = *self.inner.pushed.lock().unwrap_or_else(|p| p.into_inner());
        self.inner.spawner.spawn_unboxed(Arc::clone(&self.inner).clear(pushed));
    }

    fn pause(&self) {
        log::warn!(target: LOG,
            "pause is not supported by DataStreamAudioOutput, disable `VoiceSession.resume_false_interruption` if you are using an avatar plugin.");
    }

    fn resume(&self) {}
}

// ----------------------------------------------------------------------------- the avatar's side

/// What the avatar worker reads: audio, and where each segment ends.
#[derive(Debug, Clone, PartialEq)]
pub enum Received {
    Frame(AudioFrame),
    SegmentEnd,
}

/// Why reading stopped: closed by us, or failed (the reference re-raises it from iteration).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiverEnd {
    Closed,
    /// A stream without `sample_rate`/`num_channels`: the reference's `ValueError`.
    BadStream(String),
    Failed(String),
}

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
    remote: Mutex<Option<String>>,
    out: Mutex<RxOut>,
    out_ready: tokio::sync::Notify,
    streams: Mutex<std::collections::VecDeque<lk_agents_rtc::ByteStream>>,
    streams_ready: tokio::sync::Notify,
    /// A segment is being read.
    reading: std::sync::atomic::AtomicBool,
    /// `_current_reader_cleared`.
    cleared: std::sync::atomic::AtomicBool,
    /// `_current_reader_superseded`: a newer stream or a clear means its trailer no longer matters.
    superseded: std::sync::atomic::AtomicBool,
    superseded_wake: tokio::sync::Notify,
    rpc: tokio::sync::mpsc::UnboundedSender<RpcEvent>,
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
                return Err(ReceiverEnd::BadStream("sample_rate or num_channels not found in byte stream".into()));
            };
            let samples = (u64::from(rate) * u64::from(self.frame_size_ms)).div_ceil(1000) as u32;
            let mut bstream = lk_agents_abi::AudioByteStream::new(rate, channels, Some(samples), false);
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
                            self.push(Received::Frame(frame(f)));
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
                    self.push(Received::Frame(frame(f)));
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
        let remote = self.remote.lock().unwrap_or_else(|p| p.into_inner()).clone().unwrap_or_default();
        while let Some(ev) = rx.recv().await {
            let (method, payload) = match ev {
                RpcEvent::Finished { playback_position, interrupted } => (
                    RPC_PLAYBACK_FINISHED,
                    serde_json::json!({"playback_position": playback_position, "interrupted": interrupted, "synchronized_transcript": null}).to_string(),
                ),
                RpcEvent::Started => (RPC_PLAYBACK_STARTED, String::new()),
            };
            let mut retry = 0u32;
            while retry < self.rpc_max_retries {
                log::debug!(target: LOG, payload = payload.as_str(); "notifying {method}");
                let reply = self
                    .room
                    .local()
                    .perform_rpc(ParticipantIdentity(remote.clone()), method.to_owned(), payload.clone())
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

    /// `start()`: wait for the agent, then serve its streams and the clear RPC.
    pub async fn start(&self) -> std::result::Result<(), ReceiverEnd> {
        let inner = &self.inner;
        let mut events = inner.room.events();
        let remote = loop {
            let found = match &inner.sender_identity {
                Some(id) => inner.room.participant(&ParticipantIdentity(id.clone())).map(|p| p.identity().0.clone()),
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

        let mut streams = inner.room.byte_streams(AUDIO_STREAM_TOPIC).map_err(|e| ReceiverEnd::Failed(e.to_string()))?;
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

    pub fn notify_playback_finished(&self, playback_position: f64, interrupted: bool) {
        let _ = self.inner.rpc.send(RpcEvent::Finished { playback_position, interrupted });
    }

    pub fn notify_playback_started(&self) {
        let _ = self.inner.rpc.send(RpcEvent::Started);
    }

    /// The next frame or segment end; the end once closed or failed.
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

    pub fn aclose(&self) {
        self.inner.end(ReceiverEnd::Closed);
        for t in self.tasks.lock().unwrap_or_else(|p| p.into_inner()).drain(..) {
            t.abort();
        }
    }
}

type RxRoutes = HashMap<String, HashMap<String, Weak<RxInner>>>;
static RX_ROUTES: LazyLock<Mutex<RxRoutes>> = LazyLock::new(Mutex::default);

// ----------------------------------------------------------------------------- the runner's bookkeeping

/// `AvatarRunner`'s playback bookkeeping: what played of the segment, and whether one is playing,
/// so the runner reports `lk.playback_started`/`lk.playback_finished` at the reference's moments.
#[derive(Debug, Default)]
pub struct AvatarPlayback {
    position: f64,
    playing: bool,
}

#[lk_agents_abi::host(object(cell = "mutex"))]
impl AvatarPlayback {
    #[host(new)]
    pub fn new() -> Self {
        AvatarPlayback::default()
    }

    /// Audio reached the video generator.
    #[host(method)]
    pub fn audio_received(&mut self) {
        self.playing = true;
    }

    /// An audio frame went out to the room: `True` when it starts the segment's playback.
    #[host(method)]
    pub fn audio_played(&mut self, duration: f64) -> bool {
        let started = self.position == 0.0 && duration > 0.0;
        self.position += duration;
        started
    }

    /// The generator ended the segment: its played length to report, if audio was playing.
    #[host(method)]
    pub fn segment_ended(&mut self) -> Option<f64> {
        if !std::mem::take(&mut self.playing) {
            return None;
        }
        Some(std::mem::take(&mut self.position))
    }

    /// `clear_buffer` arrived: whether audio was playing (the interrupted report is owed).
    #[host(method)]
    pub fn clear(&mut self) -> bool {
        std::mem::take(&mut self.playing)
    }

    /// After the generator cleared: the interrupted segment's played length.
    #[host(method)]
    pub fn cleared(&mut self) -> f64 {
        std::mem::take(&mut self.position)
    }
}

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

// ----------------------------------------------------------------------------- the receiver's binding

#[cfg(feature = "python")]
mod python {
    use std::sync::Arc;

    use lk_agents_abi::python::{HostProxy, check_pid};
    use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration, PyValueError};
    use pyo3::prelude::*;

    use super::{DataStreamAudioReceiver, Received, ReceiverEnd};

    fn end_to_py(end: ReceiverEnd) -> PyErr {
        match end {
            ReceiverEnd::Closed => PyStopAsyncIteration::new_err(()),
            ReceiverEnd::BadStream(e) => PyValueError::new_err(e),
            ReceiverEnd::Failed(e) => PyRuntimeError::new_err(e),
        }
    }

    /// `_core.DataStreamAudioReceiver`: the engine of `DataStreamAudioReceiver`.
    #[pyclass(frozen, name = "DataStreamAudioReceiver", module = "livekit.agents._core")]
    pub struct PyDataStreamAudioReceiver {
        inner: Arc<DataStreamAudioReceiver>,
    }

    lk_agents_abi::register_host!(class PyDataStreamAudioReceiver);

    #[pymethods]
    impl PyDataStreamAudioReceiver {
        /// `room` is the avatar's `rtc.Room`; `on_clear_buffer()` is called when the agent clears.
        #[new]
        #[pyo3(signature = (room, *, sender_identity = None, frame_size_ms = 100, rpc_max_retries = 3, on_clear_buffer = None))]
        fn new(
            room: &Bound<'_, PyAny>,
            sender_identity: Option<String>,
            frame_size_ms: u32,
            rpc_max_retries: u32,
            on_clear_buffer: Option<Py<PyAny>>,
        ) -> PyResult<Self> {
            check_pid()?;
            let room = <dyn lk_agents_rtc::RtcRoom as HostProxy>::from_host(room, None)?;
            let on_clear = on_clear_buffer.map(|cb| {
                Arc::new(move || {
                    Python::attach(|py| {
                        if let Err(e) = cb.call0(py) {
                            e.print(py);
                        }
                    })
                }) as Arc<dyn Fn() + Send + Sync>
            });
            let clock = <Arc<dyn lk_agents_abi::Clock> as lk_agents_abi::python::Ambient>::ambient();
            let spawner = <Arc<dyn lk_agents_abi::Spawner> as lk_agents_abi::python::Ambient>::ambient();
            Ok(PyDataStreamAudioReceiver {
                inner: Arc::new(DataStreamAudioReceiver::new(
                    room,
                    sender_identity,
                    frame_size_ms.max(1),
                    rpc_max_retries,
                    on_clear,
                    clock,
                    spawner,
                )),
            })
        }

        fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            lk_agents_abi::python::future_into_py(py, async move { inner.start().await.map_err(end_to_py) })
        }

        /// The next `rtc.AudioFrame`, or `None` at a segment's end; raises `StopAsyncIteration`
        /// once closed, or what made it fail.
        fn recv<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = Arc::clone(&self.inner);
            lk_agents_abi::python::future_into_py(py, async move {
                match inner.recv().await {
                    Ok(Received::Frame(f)) => Ok(Some(f)),
                    Ok(Received::SegmentEnd) => Ok(None),
                    Err(end) => Err(end_to_py(end)),
                }
            })
        }

        fn notify_playback_finished(&self, playback_position: f64, interrupted: bool) {
            self.inner.notify_playback_finished(playback_position, interrupted);
        }

        fn notify_playback_started(&self) {
            self.inner.notify_playback_started();
        }

        fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            self.inner.aclose();
            lk_agents_abi::python::future_into_py(py, async { Ok(()) })
        }
    }
}

#[cfg(test)]
#[path = "avatar_tests.rs"]
mod tests;
