//! Independent deterministic avatar-output performer for the shared synchronous output model.
//! It uses fake RTC ports and the fixture's scheduler; it never calls the runtime executor.

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
use lk_agents_voice::avatar::output::{OutputError, OutputState};
use lk_agents_voice::avatar::{
    AUDIO_STREAM_TOPIC, RPC_CLEAR_BUFFER, RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED,
    playback_method, stream_name,
};

#[derive(Default)]
enum Writer {
    #[default]
    New,
    Open(Box<dyn ByteStreamWriter>),
    Closed,
}

#[derive(Default)]
struct State {
    core: OutputState,
    streams: BTreeMap<PlaybackSegment, Arc<tokio::sync::Mutex<Writer>>>,
}

struct Fixture {
    room: Arc<dyn RtcRoom>,
    identity: String,
    rate: Option<u32>,
    wait_track: Option<i32>,
    report_start: bool,
    clear_after: Option<Duration>,
    clock: Arc<dyn Clock>,
    scheduler: Arc<dyn Spawner>,
    tracker: Arc<PlaybackTracker>,
    state: Mutex<State>,
    writes: Arc<TaskSet>,
    closes: Arc<TaskSet>,
    timer: Mutex<Option<(AbortHandle, Ts, PlaybackSegment)>>,
}

fn refused(error: OutputError) -> IoError {
    IoError::Rtc(format!("avatar output: {error:?}"))
}

impl Fixture {
    fn retire(&self, key: PlaybackSegment) {
        let (token, stream) = {
            let mut state = self.state.lock().unwrap();
            (state.core.retire(key), state.streams.remove(&key))
        };
        if let Some(stream) = stream {
            self.close_stream(stream);
        }
        if let Some(token) = token {
            for base in [RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED] {
                if base == RPC_PLAYBACK_STARTED && !self.report_start {
                    continue;
                }
                if let Ok(method) = playback_method(base, &token) {
                    let _ = self.room.local().unregister_rpc_method(method);
                }
            }
        }
    }

    fn close_stream(&self, stream: Arc<tokio::sync::Mutex<Writer>>) {
        self.closes.spawn(&*self.scheduler, "fake_avatar_close", async move {
            let writer = std::mem::replace(&mut *stream.lock().await, Writer::Closed);
            if let Writer::Open(writer) = writer {
                let _ = writer.close().await;
            }
        });
    }

    fn routes(self: &Arc<Self>, key: PlaybackSegment, token: AvatarPlaybackSegment) -> Result<()> {
        self.state.lock().unwrap().core.register(key, token.clone()).map_err(refused)?;
        for base in [RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED] {
            if base == RPC_PLAYBACK_STARTED && !self.report_start {
                continue;
            }
            let method =
                playback_method(base, &token).map_err(|error| IoError::Rtc(error.to_string()))?;
            let owner = Arc::downgrade(self);
            self.room.local().register_rpc_method(
                method.clone(),
                RpcHandler::new(move |caller, payload| {
                    let Some(owner) = owner.upgrade() else {
                        return "reject: retired output".into();
                    };
                    if caller != owner.identity {
                        return "reject: unexpected participant".into();
                    }
                    if base == RPC_PLAYBACK_STARTED {
                        return if owner.tracker.report_started(key, unix_secs(owner.clock.wall())) {
                            "ok".into()
                        } else {
                            "reject: stale playback segment".into()
                        };
                    }
                    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
                        return "reject: malformed report".into();
                    };
                    let Some(position) = value
                        .get("playback_position")
                        .and_then(|value| value.as_f64())
                        .and_then(|value| Duration::try_from_secs_f64(value).ok())
                    else {
                        return "reject: malformed position".into();
                    };
                    let event = PlaybackFinished {
                        playback_position: position,
                        interrupted: value
                            .get("interrupted")
                            .and_then(|value| value.as_bool())
                            .unwrap_or(false),
                        synchronized_transcript: value
                            .get("synchronized_transcript")
                            .and_then(|value| value.as_str())
                            .map(str::to_owned),
                    };
                    if !owner.tracker.playback_finished(key, event) {
                        return "reject: stale playback segment".into();
                    }
                    let timer = {
                        let mut timer = owner.timer.lock().unwrap();
                        if timer.as_ref().is_some_and(|(_, _, timer_key)| *timer_key == key) {
                            timer.take()
                        } else {
                            None
                        }
                    };
                    if let Some((timer, _, _)) = timer {
                        timer.abort();
                    }
                    owner.retire(key);
                    "ok".into()
                }),
            )?;
            if !self.state.lock().unwrap().core.contains(key) {
                self.room.local().unregister_rpc_method(method)?;
                return Err(refused(OutputError::Retired));
            }
        }
        Ok(())
    }

    async fn write(
        self: Arc<Self>,
        key: PlaybackSegment,
        stream: Arc<tokio::sync::Mutex<Writer>>,
        frame: AudioFrame,
    ) -> Result<()> {
        let mut writer = stream.lock().await;
        match &*writer {
            Writer::Closed => return Err(refused(OutputError::Retired)),
            Writer::New => {
                let opened = self
                    .room
                    .local()
                    .stream_bytes(StreamBytesOptions {
                        name: stream_name(key),
                        topic: AUDIO_STREAM_TOPIC.into(),
                        destination_identities: vec![ParticipantIdentity(self.identity.clone())],
                        attributes: BTreeMap::from([
                            ("sample_rate".into(), frame.sample_rate.to_string()),
                            ("num_channels".into(), frame.num_channels.to_string()),
                        ]),
                    })
                    .await?;
                let token =
                    AvatarPlaybackSegment { playback: key, stream_id: Some(opened.id().into()) };
                *writer = Writer::Open(opened);
                self.routes(key, token)?;
                if !self.report_start {
                    self.tracker.report_started(key, unix_secs(self.clock.wall()));
                }
            }
            Writer::Open(_) => {}
        }
        if let Writer::Open(writer) = &*writer {
            writer.write(&frame.data).await?;
        }
        Ok(())
    }

    fn stop(&self) {
        let (keys, streams) = {
            let mut state = self.state.lock().unwrap();
            state.core.close();
            (state.core.keys(), state.streams.values().cloned().collect::<Vec<_>>())
        };
        self.writes.close();
        if let Some((timer, _, _)) = self.timer.lock().unwrap().take() {
            timer.abort();
        }
        for stream in streams {
            self.close_stream(stream);
        }
        self.tracker.reset();
        for key in keys {
            self.retire(key);
        }
    }
}

pub struct FakeAvatarOutput(Arc<Fixture>);

impl FakeAvatarOutput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        identity: String,
        rate: Option<u32>,
        wait_track: Option<i32>,
        clear_after: Option<Duration>,
        report_start: bool,
        clock: Arc<dyn Clock>,
        scheduler: Arc<dyn Spawner>,
    ) -> Self {
        Self(Arc::new(Fixture {
            room,
            identity,
            rate,
            wait_track,
            clear_after,
            report_start,
            clock,
            scheduler,
            tracker: Arc::new(PlaybackTracker::new()),
            state: Mutex::default(),
            writes: Arc::default(),
            closes: Arc::default(),
            timer: Mutex::default(),
        }))
    }

    pub fn wake_at(&self) -> Option<Ts> {
        self.0.timer.lock().unwrap().as_ref().map(|(_, at, _)| *at)
    }

    pub async fn aclose(&self) -> Closed {
        self.0.stop();
        let due = self.0.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        if self.0.closes.wait_empty(&*self.0.clock, Some(due)).await {
            Closed::Clean
        } else {
            Closed::TimedOut
        }
    }
}

impl Drop for FakeAvatarOutput {
    fn drop(&mut self) {
        self.0.stop();
    }
}

impl AudioOutput for FakeAvatarOutput {
    fn label(&self) -> &str {
        "DataStreamIO"
    }
    fn tracker(&self) -> &Arc<PlaybackTracker> {
        &self.0.tracker
    }
    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: false }
    }
    fn sample_rate(&self) -> Option<u32> {
        self.0.rate
    }
    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if self.0.state.lock().unwrap().core.is_closed() {
                return Err(refused(OutputError::Closed));
            }
            self.0.room.wait_for_participant(Some(self.0.identity.clone()), None, true).await?;
            if let Some(kind) = self.0.wait_track {
                self.0
                    .room
                    .wait_for_track_publication(
                        Some(self.0.identity.clone()),
                        Some(vec![kind]),
                        true,
                        false,
                    )
                    .await?;
            }
            let (key, stream) = {
                let mut state = self.0.state.lock().unwrap();
                let key =
                    self.tracker().current_segment().ok_or(IoError::PlaybackSegmentUnavailable)?;
                let key = state.core.admit(key).map_err(refused)?;
                let stream = state.streams.entry(key).or_default().clone();
                (key, stream)
            };
            let fixture = self.0.clone();
            let frame = frame.clone();
            let (sent, received) = tokio::sync::oneshot::channel();
            self.0.writes.try_spawn(&*self.0.scheduler, "fake_avatar_write", async move {
                let _ = sent.send(fixture.write(key, stream, frame).await);
            });
            received.await.unwrap_or_else(|_| Err(refused(OutputError::Closed)))
        })
    }
    fn flush_segment(&self) {
        let stream = {
            let mut state = self.0.state.lock().unwrap();
            let Some(key) = state.core.active() else { return };
            state.core.flush(key);
            state.streams.get(&key).cloned()
        };
        if let Some(stream) = stream {
            self.0.close_stream(stream);
        }
    }
    fn clear_buffer(&self) {
        if self.tracker().take_uncleared().is_zero() {
            return;
        }
        let Some(key) = self.0.state.lock().unwrap().core.last() else { return };
        let fixture = self.0.clone();
        self.0.writes.try_spawn(&*self.0.scheduler, "fake_avatar_clear", async move {
            let result = fixture
                .room
                .local()
                .perform_rpc(
                    ParticipantIdentity(fixture.identity.clone()),
                    RPC_CLEAR_BUFFER.into(),
                    String::new(),
                )
                .await;
            if !fixture.tracker.has_pending_segment(key) {
                fixture.retire(key);
                return;
            }
            let timeout = match result {
                Ok((0, _)) => fixture.clear_after,
                _ => Some(Duration::ZERO),
            };
            let Some(timeout) = timeout else { return };
            let at = fixture.clock.now().saturating_add(timeout);
            let owner = fixture.clone();
            let timer = fixture.writes.try_spawn(
                &*fixture.scheduler,
                "fake_avatar_clear_due",
                async move {
                    owner.clock.wait_until(at).await;
                    owner.tracker.playback_finished(
                        key,
                        PlaybackFinished {
                            playback_position: Duration::ZERO,
                            interrupted: true,
                            synchronized_transcript: None,
                        },
                    );
                    owner.retire(key);
                    let mut timer = owner.timer.lock().unwrap();
                    if timer.as_ref().is_some_and(|(_, _, timer_key)| *timer_key == key) {
                        timer.take();
                    }
                },
            );
            if let Some(timer) = timer {
                if let Some((prior, _, _)) = fixture.timer.lock().unwrap().replace((timer, at, key))
                {
                    prior.abort();
                }
            }
        });
    }
}
