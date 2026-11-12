use crate::rtc::events::RpcHandler;
use crate::telemetry;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::avatar::{AudioReceiverObservation, AvatarAudio};
use lk_agents_abi::voice::avatar_queue::{QueueClosed, QueueEffects};
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputExt, AvatarPlaybackSegment, IoError, OutputAlreadyChained,
    PlaybackEvent, PlaybackFinished, PlaybackFinishedEvent, PlaybackListener, PlaybackObserver,
    PlaybackSegment, PlaybackTracker, Result,
};
use lk_agents_abi::voice::rtc::{
    self, AudioOutputCapabilities, ByteStreamWriter, ParticipantIdentity, RoomEvent, RtcError,
    RtcRoom, StreamBytesOptions,
};
use lk_agents_abi::{AudioFrame, BoxFuture, Clock, Closed, Spawner, Ts, WallAnchor};
use lk_agents_net::signal;
use lk_agents_net::task::{ScopedTask, TaskSet};
use lk_agents_voice::avatar::machine::output::{
    AfterClear, Finish, OutputState, Reject, after_clear, check_caller, report_routes,
    unacknowledged_clear,
};
use lk_agents_voice::avatar::machine::queue::Queue;
use lk_agents_voice::avatar::{
    AUDIO_STREAM_TOPIC, RPC_CLEAR_BUFFER, RPC_PLAYBACK_FINISHED, playback_method, stream_name,
};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

/// Waits for the room's next event; its events ending is `Closed(closed)`.
async fn next_room_event(
    events: &mut rtc::Receiver<RoomEvent>,
    closed: &'static str,
) -> rtc::Result<()> {
    match events.next().await? {
        Some(_) => Ok(()),
        None => Err(RtcError::Closed(closed)),
    }
}

fn output_error(error: lk_agents_voice::avatar::machine::output::OutputError) -> IoError {
    use lk_agents_voice::avatar::machine::output::OutputError;
    IoError::Rtc(
        match error {
            OutputError::Closed => "the avatar output closed",
            OutputError::Full => "avatar output segment admission is full",
            OutputError::Retired => "the avatar segment retired",
            OutputError::DifferentSegment => "another avatar segment is still capturing",
        }
        .into(),
    )
}

struct Inner {
    room: Arc<dyn RtcRoom>,
    destination: String,
    wait_remote_track: Option<i32>,
    wait_playback_start: bool,
    clear_buffer_timeout: Option<Duration>,
    sample_rate: Option<u32>,
    tracker: Arc<dyn PlaybackTracker>,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    /// `None` while starting, then whether the avatar became ready. The start task's: its end
    /// (a close cancels it) ends every wait.
    started: signal::Cell<Option<rtc::Result<()>>>,
    /// The avatar lifecycle's verdict while the avatar is overdue: why audio fails instead of
    /// waiting for it. Its producer is the lifecycle's join wait.
    absence: Mutex<Option<signal::Cell<Option<String>>>>,
    /// The host stage built over this output, which its playback reports reach; released at the
    /// close.
    parent: Mutex<Option<Arc<Parent>>>,
    /// The open segment. Its stream is opened, written and closed by tasks of their own, so a
    /// capture the playout drops mid-write leaves the stream to the segment's close.
    segment: Mutex<Segments>,
    tasks: Arc<TaskSet>,
    closing: Arc<TaskSet>,
    close_outcome: Arc<Mutex<Closed>>,
    /// The give-up timer after a clear, and when it fires.
    clear_timer: Mutex<Option<(ScopedTask, Ts, PlaybackSegment)>>,
    /// The plain v1 report methods' routes for the avatar, by method.
    plain_routes: Mutex<Vec<(&'static str, u64)>>,
}

#[derive(Default)]
struct Segments {
    core: OutputState,
    writers: BTreeMap<PlaybackSegment, Arc<Segment>>,
}

/// One segment's byte stream: opened by its first frame, in order behind the async lock.
struct Segment {
    key: PlaybackSegment,
    writer: tokio::sync::Mutex<SegmentWriter>,
}

#[derive(Default)]
enum SegmentWriter {
    #[default]
    Unopened,
    Open(Box<dyn ByteStreamWriter>),
    Closed,
}

/// The host stage built over the output (`next_in_chain=`): each playback report reaches its
/// `on_playback_*`, stamped in wall seconds.
struct Parent {
    stage: Arc<dyn PlaybackListener>,
    anchor: WallAnchor,
}

impl PlaybackObserver for Parent {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        match ev {
            PlaybackEvent::Started(e) => {
                self.stage.on_playback_started(e.segment, self.anchor.seconds(e.created_at));
            }
            PlaybackEvent::Progressed(e) => self.stage.on_playback_progressed(
                e.segment,
                self.anchor.seconds(e.started_at),
                e.offset.as_secs_f64(),
                e.duration.as_secs_f64(),
            ),
            PlaybackEvent::Finished { segment, event } => self.stage.on_playback_finished(
                *segment,
                event.playback_position.as_secs_f64(),
                event.interrupted,
                event.synchronized_transcript.clone(),
            ),
        }
    }
}

impl Inner {
    /// Until the avatar is ready, or at once while the lifecycle reports it overdue.
    async fn ready(&self) -> Result<()> {
        let absence = self.absence.lock().clone();
        let absent = async move {
            match &absence {
                Some(absence) => absence.wait_map(Option::clone).await.ok(),
                None => None,
            }
        };
        tokio::select! {
            biased;
            ready = self.started.wait() => {
                Ok(ready.unwrap_or(Err(RtcError::Closed("the avatar output")))?)
            }
            Some(reason) = absent => Err(IoError::Rtc(reason)),
        }
    }

    fn is_started(&self) -> bool {
        let started = self.started.with(|started| matches!(started, Some(Ok(()))));
        started && !self.segment.lock().core.is_closed()
    }

    /// `_start_task`: the room connected, the RPC methods registered, the avatar in the room (and
    /// its track published, when asked).
    async fn start(self: Arc<Self>, started: signal::Producer<Option<rtc::Result<()>>>) {
        let outcome = self.wait_until_ready().await;
        started.resolve(outcome);
    }

    async fn wait_until_ready(self: &Arc<Self>) -> rtc::Result<()> {
        let mut events = self.room.events();
        while !self.room.connection_state()?.is_connected() {
            next_room_event(&mut events, "room before connection").await?;
        }
        self.register_plain_routes()?;
        if self.segment.lock().core.is_closed() {
            self.retire_routes();
            return Err(RtcError::Closed("avatar output"));
        }
        let identity = ParticipantIdentity(self.destination.clone());
        log::debug!(target: lk_agents_abi::LOG_TARGET, identity = self.destination.as_str(); "waiting for the remote participant");
        while self.room.participant(&identity)?.is_none() {
            next_room_event(&mut events, "room").await?;
        }
        if let Some(kind) = self.wait_remote_track {
            log::debug!(target: lk_agents_abi::LOG_TARGET, identity = self.destination.as_str(), kind = kind; "waiting for the remote track");
            while !self.remote_track_published(&identity, kind)? {
                next_room_event(&mut events, "room").await?;
            }
        }
        log::debug!(target: lk_agents_abi::LOG_TARGET, identity = self.destination.as_str(); "remote participant ready");
        Ok(())
    }

    /// Whether `identity` is in the room with a track of `kind` published.
    fn remote_track_published(
        &self,
        identity: &ParticipantIdentity,
        kind: i32,
    ) -> rtc::Result<bool> {
        let Some(participant) = self.room.participant(identity)? else { return Ok(false) };
        Ok(participant.track_publications()?.iter().any(|track| track.kind() == kind))
    }

    /// One frame into `segment`, opening its stream on the first.
    async fn write(self: &Arc<Self>, segment: &Arc<Segment>, frame: &AudioFrame) -> Result<()> {
        let mut writer = segment.writer.lock().await;
        if matches!(*writer, SegmentWriter::Closed) {
            return Err(IoError::Rtc("the avatar segment closed".into()));
        }
        if matches!(*writer, SegmentWriter::Unopened) {
            let opts = StreamBytesOptions {
                name: stream_name(segment.key),
                topic: AUDIO_STREAM_TOPIC.to_owned(),
                destination_identities: vec![ParticipantIdentity(self.destination.clone())],
                attributes: BTreeMap::from([
                    ("sample_rate".to_owned(), frame.sample_rate.to_string()),
                    ("num_channels".to_owned(), frame.num_channels.to_string()),
                ]),
            };
            let opened = self.room.local().stream_bytes(&telemetry::Cx::NONE, opts).await?;
            let token = AvatarPlaybackSegment {
                playback: segment.key,
                stream_id: Some(opened.id().to_owned()),
            };
            *writer = SegmentWriter::Open(opened);
            let continued = self.register_segment(segment, token)?;
            if !self.wait_playback_start && !continued {
                // Approximated: the avatar does not report lk.playback_started.
                self.tracker.playback_started(segment.key, self.clock.now());
            }
        }
        if let SegmentWriter::Open(writer) = &*writer {
            writer.write(&telemetry::Cx::NONE, &frame.data).await?;
        }
        Ok(())
    }

    /// Routes the stream's playback reports to its segment; whether an earlier stream of the
    /// segment already started it.
    fn register_segment(
        self: &Arc<Self>,
        segment: &Arc<Segment>,
        token: AvatarPlaybackSegment,
    ) -> Result<bool> {
        let continued = {
            let mut state = self.segment.lock();
            state.core.register(segment.key, token.clone()).map_err(output_error)?;
            state.core.continued(segment.key)
        };
        for &base in report_routes(self.wait_playback_start) {
            let method = playback_method(base, &token);
            let target = Arc::downgrade(self);
            let key = segment.key;
            let handler = RpcHandler::new(move |caller, payload| {
                let Some(inner) = target.upgrade() else { return Reject::Retired.wire() };
                inner.on_report(base, key, continued, caller, payload)
            });
            self.room.local().register_rpc_method(method.clone(), handler)?;
            let retained = self.segment.lock().core.contains(segment.key);
            if !retained {
                // Retirement may remove the token before this registration returns.
                self.room.local().unregister_rpc_method(method)?;
                return Err(IoError::Rtc("the avatar segment retired".into()));
            }
        }
        Ok(continued)
    }

    /// v1 avatar workers report on the plain methods, about the oldest segment streaming to them
    /// (v1 `_datastream_io.py` registers them per caller at the room's connection).
    fn register_plain_routes(self: &Arc<Self>) -> rtc::Result<()> {
        let routes = self.room.rpc_router().clone();
        for &base in report_routes(self.wait_playback_start) {
            let target = Arc::downgrade(self);
            let callback = RpcHandler::new(move |caller, payload| {
                let Some(inner) = target.upgrade() else { return Reject::Retired.wire() };
                let oldest = {
                    let state = inner.segment.lock();
                    state.core.oldest_streamed().map(|key| (key, state.core.continued(key)))
                };
                let Some((key, continued)) = oldest else { return Reject::Stale.wire() };
                inner.on_report(base, key, continued, caller, payload)
            });
            let registration = routes.insert(base, self.destination.clone(), callback);
            self.plain_routes.lock().push((base, registration));
            let shared = routes.clone();
            let handler =
                RpcHandler::new(move |caller, payload| shared.call(base, caller, payload));
            self.room.local().register_rpc_method(base.to_owned(), handler)?;
        }
        Ok(())
    }

    /// One avatar report about `key`: `continued` when an earlier stream already started it.
    fn on_report(
        &self,
        base: &str,
        key: PlaybackSegment,
        continued: bool,
        caller: &str,
        payload: &str,
    ) -> String {
        if let Err(reply) = check_caller(caller, &self.destination) {
            return reply.wire();
        }
        let reply = if base == RPC_PLAYBACK_FINISHED {
            self.on_playback_finished(key, payload)
        } else if continued || self.tracker.playback_started(key, self.clock.now()) {
            Ok(())
        } else {
            Err(Reject::Stale)
        };
        reply.map_or_else(Reject::wire, |()| "ok".into())
    }

    fn on_playback_finished(
        &self,
        key: PlaybackSegment,
        payload: &str,
    ) -> std::result::Result<(), Reject> {
        let finish = self.segment.lock().core.report_finished(key, payload);
        let event = match finish {
            Err(reply) => return Err(reply),
            Ok(Finish::Continue { route }) => {
                self.reopen_segment(key, route);
                return Ok(());
            }
            Ok(Finish::Done(event)) => event,
        };
        if !self.tracker.playback_finished(key, event) {
            return Err(Reject::Stale);
        }
        let timer = {
            let mut timer = self.clear_timer.lock();
            if timer.as_ref().is_some_and(|(_, _, segment)| *segment == key) {
                timer.take()
            } else {
                None
            }
        };
        drop(timer);
        self.retire_segment(key);
        Ok(())
    }

    fn retire_segment(&self, key: PlaybackSegment) {
        let (writer, token) = {
            let mut state = self.segment.lock();
            let token = state.core.retire(key);
            (state.writers.remove(&key), token)
        };
        if let Some(writer) = writer {
            self.close_segment(writer);
        }
        self.retire_route(token);
    }

    /// The avatar gave up on the segment's stream while the agent still feeds it: that stream
    /// closes and the next frame opens another.
    fn reopen_segment(&self, key: PlaybackSegment, route: Option<AvatarPlaybackSegment>) {
        let segment = self.segment.lock().writers.get(&key).cloned();
        if let Some(segment) = segment {
            let outcome = self.close_outcome.clone();
            self.closing.spawn(&*self.spawner, "avatar_stream_reopen", async move {
                let mut writer = segment.writer.lock().await;
                if let SegmentWriter::Open(_) = &*writer
                    && let SegmentWriter::Open(given_up) =
                        std::mem::replace(&mut *writer, SegmentWriter::Unopened)
                    && let Err(error) = given_up.close(&telemetry::Cx::NONE).await
                {
                    let mut closed = outcome.lock();
                    *closed = closed.and(Closed::Failed);
                    log::error!(target: lk_agents_abi::LOG_TARGET, error = error.to_string().as_str();
                        "failed to close avatar audio stream");
                }
            });
        }
        self.retire_route(route);
    }

    fn retire_route(&self, token: Option<AvatarPlaybackSegment>) {
        let Some(token) = token else { return };
        for &base in report_routes(self.wait_playback_start) {
            let method = playback_method(base, &token);
            if let Err(error) = self.room.local().unregister_rpc_method(method) {
                log::error!(target: lk_agents_abi::LOG_TARGET, error = error.to_string().as_str(); "failed to retire avatar playback route");
            }
        }
    }

    /// `_clear_buffer_task`: ask the avatar to clear, and mark the segment done if it never says
    /// it finished.
    async fn clear(self: Arc<Self>, key: PlaybackSegment, _pushed: Duration) {
        let rpc = self
            .room
            .local()
            .perform_rpc(
                &telemetry::Cx::NONE,
                ParticipantIdentity(self.destination.clone()),
                RPC_CLEAR_BUFFER.to_owned(),
                String::new(),
            )
            .await
            .map(drop);
        let pending = self.tracker.has_pending_segment(key);
        let give_up_after = match after_clear(pending, rpc, self.clear_buffer_timeout) {
            AfterClear::Retire => {
                self.retire_segment(key);
                return;
            }
            AfterClear::AwaitFinish { give_up_after } => give_up_after,
        };
        let replaced = self.clear_timer.lock().take();
        drop(replaced);
        let Some(timeout) = give_up_after else { return };
        let me = Arc::clone(&self);
        let at = self.clock.now().saturating_add(timeout);
        let timer = self.tasks.try_scoped(&*self.spawner, "avatar_clear_timeout", async move {
            me.clock.wait_until(at).await;
            {
                let mut timer = me.clear_timer.lock();
                if timer.as_ref().is_some_and(|(_, _, segment)| *segment == key)
                    && let Some((own, _, _)) = timer.take()
                {
                    own.release();
                }
            }
            me.tracker.playback_finished(key, unacknowledged_clear());
            me.retire_segment(key);
        });
        if let Some(timer) = timer {
            *self.clear_timer.lock() = Some((timer, at, key));
        }
    }

    fn close_segment(&self, segment: Arc<Segment>) {
        let outcome = self.close_outcome.clone();
        self.closing.spawn(&*self.spawner, "avatar_stream_close", async move {
            let writer =
                std::mem::replace(&mut *segment.writer.lock().await, SegmentWriter::Closed);
            if let SegmentWriter::Open(writer) = writer
                && let Err(error) = writer.close(&telemetry::Cx::NONE).await
            {
                let mut closed = outcome.lock();
                *closed = closed.and(Closed::Failed);
                log::error!(target: lk_agents_abi::LOG_TARGET, error = error.to_string().as_str();
                        "failed to close avatar audio stream");
            }
        });
    }

    fn retire_routes(self: &Arc<Self>) {
        let plain = std::mem::take(&mut *self.plain_routes.lock());
        for (base, registration) in plain {
            self.room.rpc_router().remove(base, &self.destination, registration);
        }
        let keys = self.segment.lock().core.keys();
        for key in keys {
            self.retire_segment(key);
        }
    }

    async fn aclose(self: Arc<Self>) -> Closed {
        let deadline =
            self.clock.now().saturating_add(lk_agents_abi::utils::stream::HOST_TEARDOWN_TIMEOUT);
        let segment = {
            let mut segments = self.segment.lock();
            segments.core.close().and_then(|key| segments.writers.get(&key).cloned())
        };
        if let Some(segment) = segment {
            self.close_segment(segment);
        }
        self.tracker.reset();
        let parent = self.parent.lock().take();
        if let Some(parent) = parent {
            let parent: Arc<dyn PlaybackObserver> = parent;
            self.tracker.unobserve(&parent);
        }
        let replaced = self.clear_timer.lock().take();
        drop(replaced);
        self.retire_routes();
        self.tasks.close();
        if !self.tasks.wait_empty(&*self.clock, Some(deadline)).await {
            return Closed::TimedOut;
        }
        if !self.closing.wait_empty(&*self.clock, Some(deadline)).await {
            return Closed::TimedOut;
        }
        *self.close_outcome.lock()
    }
}

/// `DataStreamAudioOutput`: streams the agent's audio to a remote avatar worker over LiveKit
/// DataStream, which reports its playback back.
pub struct DataStreamAudioOutput {
    inner: Arc<Inner>,
}

/// Explicit close while the room is usable is the primary path. A dropped output stops its tasks
/// and routes at once, and its streams still settle on its executor, observed: this survives the
/// host loop that dropped it.
impl Drop for DataStreamAudioOutput {
    fn drop(&mut self) {
        self.inner.tasks.close();
        self.inner.retire_routes();
        let inner = Arc::clone(&self.inner);
        let spawner = Arc::clone(&inner.spawner);
        Arc::new(TaskSet::default()).spawn(&*spawner, "avatar_output_drop", async move {
            observe_close(inner.aclose().await);
        });
    }
}

fn observe_close(closed: Closed) -> Closed {
    if closed != Closed::Clean {
        log::error!(target: lk_agents_abi::LOG_TARGET, "avatar output cleanup did not settle cleanly: {closed:?}");
    }
    closed
}

impl lk_agents_abi::voice::avatar::DataStreamAudioOutput for DataStreamAudioOutput {
    type Env = lk_agents_net::env::ProcessEnv;

    fn new(
        env: &Self::Env,
        room: Arc<dyn RtcRoom>,
        destination_identity: String,
        sample_rate: Option<u32>,
        wait_remote_track: Option<i32>,
        clear_buffer_timeout: Option<f64>,
        wait_playback_start: bool,
    ) -> std::result::Result<Self, lk_agents_abi::voice::avatar::AvatarOutputError> {
        use lk_agents_abi::errors::{ApiError, ProviderError};
        use lk_agents_abi::voice::avatar::AvatarOutputError;
        let timeout = clear_buffer_timeout
            .map(|seconds| {
                Duration::try_from_secs_f64(seconds).map_err(|error| {
                    AvatarOutputError::Invalid(format!("clear_buffer_timeout: {error}"))
                })
            })
            .transpose()?;
        let spawner = (env.spawner)()
            .map_err(|error| ProviderError::from(ApiError::new(error.to_string())))?;
        Ok(Self::new(
            room,
            destination_identity,
            sample_rate,
            wait_remote_track,
            timeout,
            wait_playback_start,
            env.clock.clone(),
            spawner,
        ))
    }

    fn label(&self) -> String {
        AudioOutput::label(self).to_owned()
    }
    fn sample_rate(&self) -> Option<u32> {
        AudioOutput::sample_rate(self)
    }
    fn begin_segment(&self, segment: PlaybackSegment) -> Result<()> {
        AudioOutput::begin_segment(self, segment)
    }
    fn withdraw_capture(&self) -> Result<()> {
        AudioOutput::withdraw_capture(self)
    }
    fn playback_segment(&self) -> Option<PlaybackSegment> {
        self.inner.tracker.current_segment()
    }
    async fn capture_frame(&self, frame: AudioFrame) -> Result<()> {
        AudioOutputExt::capture_frame(self, &frame).await
    }
    fn flush(&self) {
        AudioOutputExt::flush(self);
    }
    fn clear_buffer(&self) {
        AudioOutput::clear_buffer(self);
    }
    fn can_pause(&self) -> bool {
        AudioOutputExt::can_pause(self)
    }
    fn pause(&self) {
        AudioOutput::pause(self);
    }
    fn resume(&self) {
        AudioOutput::resume(self);
    }
    fn on_attached(&self) {
        AudioOutput::on_attached(self);
    }
    fn on_detached(&self) {
        AudioOutput::on_detached(self);
    }
    async fn wait_for_playout(&self) -> PlaybackFinishedEvent {
        let ev = AudioOutputExt::wait_for_playout(self).await;
        PlaybackFinishedEvent {
            playback_position: ev.playback_position.as_secs_f64(),
            interrupted: ev.interrupted,
            synchronized_transcript: ev.synchronized_transcript,
            segment: None,
        }
    }
    fn set_parent(
        &self,
        stage: Arc<dyn PlaybackListener>,
    ) -> std::result::Result<(), OutputAlreadyChained> {
        let mut parent = self.inner.parent.lock();
        if let Some(parent) = &*parent {
            // a stage's tracker is its own: the same tracker is the same stage
            let registered = Arc::as_ptr(parent.stage.tracker());
            if std::ptr::addr_eq(registered, Arc::as_ptr(stage.tracker())) {
                return Ok(());
            }
            return Err(OutputAlreadyChained { label: AudioOutput::label(self).to_owned() });
        }
        let anchor = self.inner.tracker.anchor();
        let registered = Arc::new(Parent { stage, anchor });
        *parent = Some(Arc::clone(&registered));
        drop(parent);
        self.inner.tracker.observe(registered);
        Ok(())
    }
    async fn aclose(&self) {
        self.aclose().await;
    }
}

impl DataStreamAudioOutput {
    #[allow(
        clippy::too_many_arguments,
        reason = "Keep the operation’s existing explicit arguments"
    )]
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
        let (start, started) = signal::once();
        let tracker: Arc<dyn PlaybackTracker> = Arc::new(crate::voice::playback::Tracker::new());
        // Its reports are stamped on `clock`; a host stage over it reads them as wall seconds.
        tracker.anchor_to(WallAnchor::of(&*clock));
        let inner = Arc::new(Inner {
            room,
            destination: destination_identity,
            wait_remote_track,
            wait_playback_start,
            clear_buffer_timeout,
            sample_rate,
            tracker,
            clock,
            spawner: Arc::clone(&spawner),
            started,
            absence: Mutex::new(None),
            parent: Mutex::new(None),
            segment: Mutex::new(Segments::default()),
            tasks: Arc::new(TaskSet::default()),
            closing: Arc::new(TaskSet::default()),
            close_outcome: Arc::new(Mutex::new(Closed::Clean)),
            clear_timer: Mutex::new(None),
            plain_routes: Mutex::new(Vec::new()),
        });
        inner.tasks.spawn(&*spawner, "avatar_output_start", Arc::clone(&inner).start(start));
        DataStreamAudioOutput { inner }
    }

    /// Follow the avatar lifecycle's `absence`: while it holds a reason, audio fails with it
    /// instead of waiting for the avatar.
    pub fn follow_absence(&self, absence: signal::Cell<Option<String>>) {
        *self.inner.absence.lock() = Some(absence);
    }

    /// When the give-up timer after a clear fires, while one is armed.
    pub fn wake_at(&self) -> Option<Ts> {
        self.inner.clear_timer.lock().as_ref().map(|(_, at, _)| *at)
    }

    pub async fn aclose(&self) -> Closed {
        observe_close(Arc::clone(&self.inner).aclose().await)
    }
}

impl AudioOutput for DataStreamAudioOutput {
    fn label(&self) -> &str {
        "DataStreamIO"
    }

    fn tracker(&self) -> &Arc<dyn PlaybackTracker> {
        &self.inner.tracker
    }

    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: false }
    }

    fn sample_rate(&self) -> Option<u32> {
        self.inner.sample_rate
    }

    /// `capture_frame`: wait for the avatar, open the segment's stream on its first frame.
    fn capture<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        frame: &'a AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.inner.ready().await?;
            if self.inner.segment.lock().core.is_closed() {
                Err(RtcError::Closed("the avatar output"))?;
            }
            let segment = {
                let mut state = self.inner.segment.lock();
                let key = self
                    .inner
                    .tracker
                    .current_segment()
                    .ok_or(IoError::PlaybackSegmentUnavailable)?;
                let key = state.core.admit(key).map_err(output_error)?;

                Arc::clone(state.writers.entry(key).or_insert_with(|| {
                    Arc::new(Segment {
                        key,
                        writer: tokio::sync::Mutex::new(SegmentWriter::Unopened),
                    })
                }))
            };
            // The task retains the writer even if the capture caller is interrupted.
            let (tx, rx) = tokio::sync::oneshot::channel();
            let inner = Arc::clone(&self.inner);
            let frame = frame.clone();
            self.inner.tasks.try_spawn(&*self.inner.spawner, "avatar_audio_write", async move {
                let _ = tx.send(inner.write(&segment, &frame).await);
            });
            rx.await.unwrap_or_else(|_| Err(IoError::Rtc("the avatar output closed".into())))
        })
    }

    /// `flush`: close the segment's stream, its trailer ending the segment on the avatar.
    fn flush_segment(&self) {
        if !self.inner.is_started() {
            return;
        }
        let (key, segment) = {
            let state = self.inner.segment.lock();
            let key = state.core.active();
            (key, key.and_then(|key| state.writers.get(&key).cloned()))
        };
        if let Some(segment) = segment {
            self.inner.close_segment(segment);
        }
        let Some(key) = key else { return };
        let finished = self.inner.segment.lock().core.flush(key);
        // Every stream of the segment has finished already: its finish is due now.
        if let Some(played) = finished {
            let event = PlaybackFinished {
                playback_position: played,
                interrupted: false,
                synchronized_transcript: None,
            };
            if self.inner.tracker.playback_finished(key, event) {
                self.inner.retire_segment(key);
            }
        }
    }

    /// Once per pushed segment: the interruption's cut and the playout's both clear, and a second
    /// `lk.clear_buffer` with nothing pushed since gets no `lk.playback_finished`, so its timer
    /// would mark a later segment done.
    fn clear_buffer(&self) {
        if !self.inner.is_started() {
            return;
        }
        let pushed = self.inner.tracker.take_uncleared();
        if pushed.is_zero() {
            return;
        }
        let Some(segment) = self.inner.segment.lock().core.last() else {
            return;
        };
        self.inner.tasks.try_spawn(
            &*self.inner.spawner,
            "avatar_clear_buffer",
            Arc::clone(&self.inner).clear(segment, pushed),
        );
    }

    fn pause(&self) {
        log::warn!(target: lk_agents_abi::LOG_TARGET,
            "pause is not supported by DataStreamAudioOutput, disable `VoiceSession.resume_false_interruption` if you are using an avatar plugin.");
    }

    fn resume(&self) {}
}

/// Async reader wakeups around voice's synchronous queue and playback state machine.
pub struct QueueAudio {
    state: signal::Shared<Queue>,
}

impl lk_agents_abi::voice::avatar_queue::QueueAudio for QueueAudio {
    fn new(wait_playback_start: bool) -> Self {
        Self { state: signal::Shared::new(Queue::new(wait_playback_start)) }
    }
    fn check_open(&self) -> std::result::Result<(), QueueClosed> {
        self.state.with(|queue| queue.check_open())
    }
    fn capture(
        &self,
        frame: AudioFrame,
        segment: PlaybackSegment,
    ) -> std::result::Result<QueueEffects, QueueClosed> {
        self.state.modify(|queue| queue.capture(frame, segment))
    }
    fn flush(&self) -> std::result::Result<(), QueueClosed> {
        self.state.modify(Queue::flush)
    }
    fn clear(&self) -> QueueEffects {
        self.state.modify(Queue::clear)
    }
    fn playback_finished(
        &self,
        playback_position: f64,
        interrupted: bool,
        segment: Option<AvatarPlaybackSegment>,
    ) -> QueueEffects {
        self.state.with(|queue| queue.finished(playback_position, interrupted, segment))
    }
    fn playback_started(&self, segment: Option<AvatarPlaybackSegment>) -> QueueEffects {
        self.state.with(|queue| queue.started(segment))
    }
    async fn next_audio(&self) -> AudioReceiverObservation {
        self.wait(Queue::next_audio).await
    }
    fn close(&self) -> QueueEffects {
        self.state.modify(Queue::close)
    }
}

impl QueueAudio {
    /// What `read` finds in the queue, once it finds something.
    async fn wait<T>(&self, read: impl FnMut(&mut Queue) -> Option<T>) -> T {
        self.state.wait_until(read).await
    }
}

impl AvatarAudio for QueueAudio {
    fn next_audio(
        &self,
    ) -> BoxFuture<'_, std::result::Result<AudioReceiverObservation, ProviderError>> {
        Box::pin(async move { Ok(self.wait(Queue::next_audio).await) })
    }
    fn next_clear(&self, after: u64) -> BoxFuture<'_, Option<u64>> {
        Box::pin(self.wait(move |queue| match queue.clear_epoch() {
            Some(epoch) if epoch <= after => None,
            epoch => Some(epoch),
        }))
    }
}

#[cfg(test)]
mod queue_tests {
    use super::*;
    use lk_agents_abi::voice::avatar_queue::QueueAudio as QueueAudioClass;

    #[tokio::test]
    async fn close_releases_pending_read_and_remains_terminal() {
        let queue = QueueAudio::new(false);
        let mut pending = Box::pin(QueueAudioClass::next_audio(&queue));
        assert!(
            std::future::poll_fn(|cx| std::task::Poll::Ready(
                pending.as_mut().poll(cx).is_pending()
            ))
            .await
        );
        assert!(queue.close().reset);
        assert_eq!(pending.await, AudioReceiverObservation::default());
        assert_eq!(QueueAudioClass::next_audio(&queue).await, AudioReceiverObservation::default());
        assert!(!queue.close().reset);
    }
}
