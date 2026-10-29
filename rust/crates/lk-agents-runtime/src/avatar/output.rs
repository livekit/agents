//! Runtime avatar audio output: byte writers, RPC execution, task ownership and deadlines.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lk_agents_abi::voice::io::{
    AudioOutput, AvatarPlaybackSegment, IoError, PlaybackFinished, PlaybackSegment,
    PlaybackTracker, Result,
};
use lk_agents_abi::voice::rtc::{
    AudioOutputCapabilities, ByteStreamWriter, ParticipantIdentity, RpcHandler, RtcRoom,
    StreamBytesOptions,
};
use lk_agents_abi::{
    AbortHandle, AudioFrame, Clock, Closed, Spawner, SpawnerExt, TaskSet, Ts, unix_secs,
};
use lk_agents_voice::avatar::output::{Finish, OutputState, Reply, check_caller, report_routes};
use lk_agents_voice::avatar::{
    AUDIO_STREAM_TOPIC, LOG, RPC_CLEAR_BUFFER, RPC_PLAYBACK_FINISHED, playback_method, stream_name,
};
use tokio::sync::watch;

fn output_error(error: lk_agents_voice::avatar::output::OutputError) -> IoError {
    use lk_agents_voice::avatar::output::OutputError;
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
    tracker: Arc<PlaybackTracker>,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    /// `None` while starting, then whether the avatar became ready.
    started: watch::Sender<Option<std::result::Result<(), String>>>,
    /// The open segment. Its stream is opened, written and closed by tasks of their own, so a
    /// capture the playout drops mid-write leaves the stream to the segment's close.
    segment: Mutex<Segments>,
    tasks: Arc<TaskSet>,
    closing: Arc<TaskSet>,
    close_outcome: watch::Sender<Closed>,
    /// The give-up timer after a clear, and when it fires.
    clear_timer: Mutex<Option<(AbortHandle, Ts, PlaybackSegment)>>,
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

impl Inner {
    fn is_started(&self) -> bool {
        matches!(*self.started.borrow(), Some(Ok(())))
    }

    /// `_start_task`: the room connected, the RPC methods registered, the avatar in the room (and
    /// its track published, when asked).
    async fn start(self: Arc<Self>) {
        let outcome: lk_agents_abi::voice::rtc::Result<()> = async {
            let mut events = self.room.events();
            while !self.room.connection_state()?.is_connected() {
                if events.next().await?.is_none() {
                    return Err(lk_agents_abi::voice::rtc::RtcError::Closed("room before connection"));
                }
            }
            if self.segment.lock().unwrap_or_else(|p| p.into_inner()).core.is_closed() {
                self.retire_routes();
                return Err(lk_agents_abi::voice::rtc::RtcError::Closed("avatar output"));
            }
            let identity = ParticipantIdentity(self.destination.clone());
            log::debug!(target: LOG, identity = self.destination.as_str(); "waiting for the remote participant");
            while self.room.participant(&identity)?.is_none() {
                if events.next().await?.is_none() {
                    return Err(lk_agents_abi::voice::rtc::RtcError::Closed("room"));
                }
            }
            if let Some(kind) = self.wait_remote_track {
                log::debug!(target: LOG, identity = self.destination.as_str(), kind = kind; "waiting for the remote track");
                loop {
                    let published = match self.room.participant(&identity)? {
                        Some(participant) => {
                            participant.track_publications()?.iter().any(|track| track.kind() == kind)
                        }
                        None => false,
                    };
                    if published {
                        break;
                    }
                    if events.next().await?.is_none() {
                        return Err(lk_agents_abi::voice::rtc::RtcError::Closed("room"));
                    }
                }
            }
            log::debug!(target: LOG, identity = self.destination.as_str(); "remote participant ready");
            Ok(())

        }.await;
        self.started.send_replace(Some(outcome.map_err(|error| error.to_string())));
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
            let opened = self.room.local().stream_bytes(opts).await?;
            let token = AvatarPlaybackSegment {
                playback: segment.key,
                stream_id: Some(opened.id().to_owned()),
            };
            *writer = SegmentWriter::Open(opened);
            let continued = self.register_segment(segment, token)?;
            if !self.wait_playback_start && !continued {
                // approximated: the avatar does not report lk.playback_started
                self.tracker.report_started(segment.key, unix_secs(self.clock.wall()));
            }
        }
        if let SegmentWriter::Open(writer) = &*writer {
            writer.write(&frame.data).await?;
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
            let mut state = self.segment.lock().unwrap_or_else(|p| p.into_inner());
            state.core.register(segment.key, token.clone()).map_err(output_error)?;
            state.core.continued(segment.key)
        };
        for &base in report_routes(self.wait_playback_start) {
            let method =
                playback_method(base, &token).map_err(|error| IoError::Rtc(error.to_string()))?;
            let target = Arc::downgrade(self);
            let key = segment.key;
            let handler = RpcHandler::new(move |caller, payload| {
                let Some(inner) = target.upgrade() else { return Reply::Retired.wire() };
                if let Err(reply) = check_caller(caller, &inner.destination) {
                    return reply.wire();
                }
                let reply = if base == RPC_PLAYBACK_FINISHED {
                    inner.on_playback_finished(key, payload)
                } else if continued
                    || inner.tracker.report_started(key, unix_secs(inner.clock.wall()))
                {
                    Reply::Ok
                } else {
                    Reply::Stale
                };
                reply.wire()
            });
            self.room.local().register_rpc_method(method.clone(), handler)?;
            let retained =
                self.segment.lock().unwrap_or_else(|p| p.into_inner()).core.contains(segment.key);
            if !retained {
                // Retirement may remove the token before this registration returns.
                self.room.local().unregister_rpc_method(method)?;
                return Err(IoError::Rtc("the avatar segment retired".into()));
            }
        }
        Ok(continued)
    }

    fn on_playback_finished(&self, key: PlaybackSegment, payload: &str) -> Reply {
        let finish = self
            .segment
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .core
            .report_finished(key, payload);
        let event = match finish {
            Err(reply) => return reply,
            Ok(Finish::Continue { route }) => {
                self.reopen_segment(key, route);
                return Reply::Ok;
            }
            Ok(Finish::Done(event)) => event,
        };
        if !self.tracker.playback_finished(key, event) {
            return Reply::Stale;
        }
        let timer = {
            let mut timer = self.clear_timer.lock().unwrap_or_else(|p| p.into_inner());
            if timer.as_ref().is_some_and(|(_, _, segment)| *segment == key) {
                timer.take()
            } else {
                None
            }
        };
        if let Some((timer, _, _)) = timer {
            timer.abort();
        }
        self.retire_segment(key);
        Reply::Ok
    }

    fn retire_segment(&self, key: PlaybackSegment) {
        let (writer, token) = {
            let mut state = self.segment.lock().unwrap_or_else(|p| p.into_inner());
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
        let segment =
            self.segment.lock().unwrap_or_else(|p| p.into_inner()).writers.get(&key).cloned();
        if let Some(segment) = segment {
            let outcome = self.close_outcome.clone();
            self.closing.spawn(&*self.spawner, "avatar_stream_reopen", async move {
                let mut writer = segment.writer.lock().await;
                if let SegmentWriter::Open(_) = &*writer
                    && let SegmentWriter::Open(given_up) =
                        std::mem::replace(&mut *writer, SegmentWriter::Unopened)
                    && let Err(error) = given_up.close().await
                {
                    outcome.send_modify(|closed| *closed = closed.and(Closed::Failed));
                    log::error!(target: LOG, error = error.to_string().as_str();
                        "failed to close avatar audio stream");
                }
            });
        }
        self.retire_route(route);
    }

    fn retire_route(&self, token: Option<AvatarPlaybackSegment>) {
        let Some(token) = token else { return };
        for &base in report_routes(self.wait_playback_start) {
            let Ok(method) = playback_method(base, &token) else { continue };
            if let Err(error) = self.room.local().unregister_rpc_method(method) {
                log::error!(target: LOG, error = error.to_string().as_str(); "failed to retire avatar playback route");
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
        if !self.tracker.has_pending_segment(key) {
            self.retire_segment(key);
            return;
        }
        // a failed clear marks the playout done at once
        let timeout = match failed {
            Some(e) => {
                log::error!(target: LOG, error = e.as_str(); "failed to perform clear buffer rpc");
                Some(Duration::ZERO)
            }
            None => self.clear_buffer_timeout,
        };
        if let Some((timer, _, _)) =
            self.clear_timer.lock().unwrap_or_else(|p| p.into_inner()).take()
        {
            timer.abort();
        }
        let Some(timeout) = timeout else { return };
        let me = Arc::clone(&self);
        let at = self.clock.now().saturating_add(timeout);
        let timer = self.tasks.try_spawn(&*self.spawner, "avatar_clear_timeout", async move {
            me.clock.wait_until(at).await;
            {
                let mut timer = me.clear_timer.lock().unwrap_or_else(|p| p.into_inner());
                if timer.as_ref().is_some_and(|(_, _, segment)| *segment == key) { timer.take(); }
            }
            log::warn!(target: LOG,
                "playback clear was not acknowledged with segment identity; ending the captured segment unheard");
            me.tracker.playback_finished(key, PlaybackFinished {
                playback_position: Duration::ZERO,
                interrupted: true,
                synchronized_transcript: None,
            });
            me.retire_segment(key);
        });
        if let Some(timer) = timer {
            *self.clear_timer.lock().unwrap_or_else(|p| p.into_inner()) = Some((timer, at, key));
        }
    }

    fn close_segment(&self, segment: Arc<Segment>) {
        let outcome = self.close_outcome.clone();
        self.closing.spawn(&*self.spawner, "avatar_stream_close", async move {
            let writer =
                std::mem::replace(&mut *segment.writer.lock().await, SegmentWriter::Closed);
            if let SegmentWriter::Open(writer) = writer {
                if let Err(error) = writer.close().await {
                    outcome.send_modify(|closed| *closed = closed.and(Closed::Failed));
                    log::error!(target: LOG, error = error.to_string().as_str();
                        "failed to close avatar audio stream");
                }
            }
        });
    }

    fn retire_routes(self: &Arc<Self>) {
        let keys = self.segment.lock().unwrap_or_else(|p| p.into_inner()).core.keys();
        for key in keys {
            self.retire_segment(key);
        }
    }

    async fn aclose(self: Arc<Self>) -> Closed {
        let deadline =
            self.clock.now().saturating_add(lk_agents_abi::utils::stream::HOST_TEARDOWN_TIMEOUT);
        let segment = {
            let mut segments = self.segment.lock().unwrap_or_else(|p| p.into_inner());
            segments.core.close().and_then(|key| segments.writers.get(&key).cloned())
        };
        self.started.send_replace(Some(Err("the avatar output closed".into())));
        if let Some(segment) = segment {
            self.close_segment(segment);
        }
        self.tracker.reset();
        if let Some((timer, _, _)) =
            self.clear_timer.lock().unwrap_or_else(|p| p.into_inner()).take()
        {
            timer.abort();
        }
        self.retire_routes();
        self.tasks.close();
        if !self.tasks.wait_empty(&*self.clock, Some(deadline)).await {
            return Closed::TimedOut;
        }
        if !self.closing.wait_empty(&*self.clock, Some(deadline)).await {
            return Closed::TimedOut;
        }
        *self.close_outcome.borrow()
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
        log::error!("avatar output cleanup did not settle cleanly: {closed:?}");
    }
    closed
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
            segment: Mutex::new(Segments::default()),
            tasks: Arc::new(TaskSet::default()),
            closing: Arc::new(TaskSet::default()),
            close_outcome: watch::channel(Closed::Clean).0,
            clear_timer: Mutex::new(None),
        });
        inner.tasks.spawn(&*spawner, "avatar_output_start", Arc::clone(&inner).start());
        DataStreamAudioOutput { inner }
    }

    /// When the give-up timer after a clear fires, while one is armed.
    pub fn wake_at(&self) -> Option<Ts> {
        self.inner
            .clear_timer
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|(_, at, _)| *at)
    }

    pub async fn aclose(&self) -> Closed {
        observe_close(Arc::clone(&self.inner).aclose().await)
    }
}

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
    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut started = self.inner.started.subscribe();
            let ready = started.wait_for(Option::is_some).await.map(|s| s.clone()).ok().flatten();
            match ready {
                Some(Ok(())) => {}
                Some(Err(e)) => return Err(IoError::Rtc(e)),
                None => return Err(IoError::Rtc("the avatar output closed".into())),
            }
            let segment = {
                let mut state = self.inner.segment.lock().unwrap_or_else(|p| p.into_inner());
                let key = self
                    .inner
                    .tracker
                    .current_segment()
                    .ok_or(IoError::PlaybackSegmentUnavailable)?;
                let key = state.core.admit(key).map_err(output_error)?;
                let segment = Arc::clone(state.writers.entry(key).or_insert_with(|| {
                    Arc::new(Segment {
                        key,
                        writer: tokio::sync::Mutex::new(SegmentWriter::Unopened),
                    })
                }));
                segment
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
            let state = self.inner.segment.lock().unwrap_or_else(|p| p.into_inner());
            let key = state.core.active();
            (key, key.and_then(|key| state.writers.get(&key).cloned()))
        };
        if let Some(segment) = segment {
            self.inner.close_segment(segment);
        }
        let Some(key) = key else { return };
        let finished = self.inner.segment.lock().unwrap_or_else(|p| p.into_inner()).core.flush(key);
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
        let Some(segment) =
            self.inner.segment.lock().unwrap_or_else(|p| p.into_inner()).core.last()
        else {
            return;
        };
        self.inner.tasks.try_spawn(
            &*self.inner.spawner,
            "avatar_clear_buffer",
            Arc::clone(&self.inner).clear(segment, pushed),
        );
    }

    fn pause(&self) {
        log::warn!(target: LOG,
            "pause is not supported by DataStreamAudioOutput, disable `VoiceSession.resume_false_interruption` if you are using an avatar plugin.");
    }

    fn resume(&self) {}
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
