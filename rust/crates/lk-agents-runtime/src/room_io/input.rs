//! Runtime ownership of participant input subscriptions and their bounded delivery queues.

use lk_agents_abi::voice::io::{AudioInput, IoError, Result, VideoInput};
use lk_agents_abi::voice::rtc::{
    AudioInputStream, AudioStreamOptions, AudioTrackFeature, ParticipantIdentity,
    RemoteParticipant, RoomEvent, RtcRoom, TrackPublication, TrackSid, TrackSource, VideoFrameRef,
    silence_frame,
};
use lk_agents_abi::{
    AudioFrame, BoundedLossy, Clock, Closed, LossyTx, Pushed, Spawner, StreamSource, TaskSet, Ts,
};
use lk_agents_voice::room_io::input_state::{InputIdentity, TrackSelection};
use std::future::Future;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

use super::pre_connect_audio::PreConnectAudioHandler;
use lk_agents_abi::voice::room_io::AudioInputConfig;

const AUDIO_SOURCES: &[TrackSource] = &[TrackSource::Microphone];
const VIDEO_SOURCES: &[TrackSource] = &[TrackSource::Camera, TrackSource::Screenshare];
const MAX_READERS: usize = 4;

struct InputFrame<T> {
    identity: InputIdentity,
    item: Result<T>,
}

struct ReaderState<T> {
    selection: TrackSelection,
    stop: Option<Arc<Notify>>,
    tx: Option<LossyTx<InputFrame<T>>>,
    cleanup: Closed,
    starting: usize,
}

struct Lease {
    generation: u64,
    stop: Arc<Notify>,
}

struct TrackReader<T> {
    state: Mutex<ReaderState<T>>,
    rx: tokio::sync::Mutex<BoundedLossy<InputFrame<T>>>,
    tasks: Arc<TaskSet>,
    started: Notify,
}

impl<T> TrackReader<T> {
    fn new(capacity: usize) -> Self {
        let (tx, rx) = BoundedLossy::channel(capacity);
        Self {
            state: Mutex::new(ReaderState {
                selection: TrackSelection::default(),
                stop: None,
                tx: Some(tx),
                cleanup: Closed::Clean,
                starting: 0,
            }),
            rx: tokio::sync::Mutex::new(rx),
            tasks: Arc::default(),
            started: Notify::new(),
        }
    }

    fn reading(&self) -> Option<TrackSid> {
        self.state.lock().unwrap().selection.reading().cloned()
    }

    fn link(&self, identity: Option<ParticipantIdentity>) -> bool {
        let mut state = self.state.lock().unwrap();
        if !state.selection.link(identity) {
            return false;
        }
        if let Some(stop) = state.stop.take() {
            stop.notify_one();
        }
        true
    }

    fn claim(
        &self,
        participant: &ParticipantIdentity,
        source: TrackSource,
        sid: &TrackSid,
        sources: &[TrackSource],
    ) -> Option<Lease> {
        let mut state = self.state.lock().unwrap();
        if self.tasks.names().len() + state.starting >= MAX_READERS {
            return None;
        }
        let generation = state.selection.claim(participant, source, sid, sources)?;
        if let Some(stop) = state.stop.take() {
            stop.notify_one();
        }
        let stop = Arc::new(Notify::new());
        state.stop = Some(stop.clone());
        state.starting += 1;
        Some(Lease { generation, stop })
    }

    fn release(&self, participant: &ParticipantIdentity, sid: &TrackSid) -> bool {
        let mut state = self.state.lock().unwrap();
        if !state.selection.release(participant, sid) {
            return false;
        }
        if let Some(stop) = state.stop.take() {
            stop.notify_one();
        }
        true
    }

    fn opening_failed(&self, lease: &Lease) {
        self.ended(lease);
        self.state.lock().unwrap().starting -= 1;
        self.started.notify_waiters();
    }

    fn spawn(
        &self,
        spawner: &dyn Spawner,
        name: &str,
        work: impl Future<Output = ()> + Send + 'static,
    ) {
        self.tasks.spawn(spawner, name, work);
    }

    fn started(&self) {
        self.state.lock().unwrap().starting -= 1;
        self.started.notify_waiters();
    }

    fn ended(&self, lease: &Lease) {
        let mut state = self.state.lock().unwrap();
        if state.selection.ended(lease.generation) {
            state.stop.take();
        }
    }

    fn finish(&self, lease: &Lease, cleanup: Closed) {
        let mut state = self.state.lock().unwrap();
        state.cleanup = state.cleanup.and(cleanup);
        if state.selection.ended(lease.generation) {
            state.stop.take();
        }
    }

    fn deliver(&self, lease: &Lease, item: Result<T>) -> bool {
        let state = self.state.lock().unwrap();
        if !state.selection.producing(lease.generation) {
            return false;
        }
        let Some(identity) = state.selection.admit(lease.generation) else {
            return true;
        };
        state.tx.as_ref().is_some_and(|tx| tx.push(InputFrame { identity, item }) != Pushed::Closed)
    }

    async fn recv(&self) -> Result<Option<T>> {
        let mut rx = self.rx.lock().await;
        while let Some(frame) = rx.recv().await {
            if self.state.lock().unwrap().selection.accepts(frame.identity) {
                return frame.item.map(Some);
            }
        }
        Ok(None)
    }

    fn set_attached(&self, attached: bool) {
        self.state.lock().unwrap().selection.set_attached(attached);
    }

    fn replay_pre_connect(
        &self,
        participant: ParticipantIdentity,
        participant_sid: String,
        sid: TrackSid,
    ) -> bool {
        self.state.lock().unwrap().selection.replay_pre_connect(
            participant,
            participant_sid,
            sid,
            lk_agents_voice::room_io::pre_connect_state::MAX_TRACKS,
        )
    }

    async fn close(&self, clock: &dyn Clock, deadline: Ts) -> Closed {
        {
            let mut state = self.state.lock().unwrap();
            state.selection.close();
            if let Some(stop) = state.stop.take() {
                stop.notify_one();
            }
            state.tx.take();
        }
        loop {
            let started = self.started.notified();
            tokio::pin!(started);
            started.as_mut().enable();
            if self.state.lock().unwrap().starting == 0 {
                break;
            }
            if lk_agents_abi::timeout(
                clock,
                deadline.saturating_duration_since(clock.now()),
                &mut started,
            )
            .await
            .is_err()
            {
                return Closed::TimedOut;
            }
        }
        if self.tasks.wait_empty(clock, Some(deadline)).await {
            self.state.lock().unwrap().cleanup
        } else {
            Closed::TimedOut
        }
    }
}

fn read_first(
    participant: &Arc<dyn RemoteParticipant>,
    skip: Option<&TrackSid>,
    mut try_read: impl FnMut(&Arc<dyn TrackPublication>) -> Result<bool>,
) -> Result<()> {
    for publication in participant.track_publications()? {
        if Some(publication.sid()) != skip
            && publication.is_subscribed()?
            && try_read(&publication)?
        {
            break;
        }
    }
    Ok(())
}

fn link_and_read<T>(
    reader: &TrackReader<T>,
    room: &dyn RtcRoom,
    identity: Option<ParticipantIdentity>,
    mut try_read: impl FnMut(&Arc<dyn RemoteParticipant>, &Arc<dyn TrackPublication>) -> Result<bool>,
) -> Result<()> {
    if !reader.link(identity.clone()) && reader.reading().is_some() {
        return Ok(());
    }
    if let Some(identity) = identity
        && let Some(participant) = room.participant(&identity)?
    {
        read_first(&participant, None, |track| try_read(&participant, track))?;
    }
    Ok(())
}

fn route<T>(
    reader: &TrackReader<T>,
    event: &RoomEvent,
    mut try_read: impl FnMut(&Arc<dyn RemoteParticipant>, &Arc<dyn TrackPublication>) -> Result<bool>,
) -> Result<()> {
    match event {
        RoomEvent::TrackSubscribed { participant, publication } => {
            try_read(participant, publication)?;
        }
        RoomEvent::TrackUnpublished { participant, publication } => {
            if reader.release(participant.identity(), publication.sid()) {
                read_first(participant, None, |track| try_read(participant, track))?;
            }
        }
        RoomEvent::TrackUnsubscribed { participant, publication } => {
            if reader.release(participant.identity(), publication.sid()) {
                read_first(participant, Some(publication.sid()), |track| {
                    try_read(participant, track)
                })?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub struct ParticipantVideoInput {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    reader: TrackReader<VideoFrameRef>,
}

impl ParticipantVideoInput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
    ) -> Arc<Self> {
        Arc::new(Self {
            room,
            spawner,
            clock,
            reader: TrackReader::new(lk_agents_abi::voice::rtc::VIDEO_QUEUE_FRAMES),
        })
    }

    pub fn reading(&self) -> Option<TrackSid> {
        self.reader.reading()
    }

    pub fn set_participant(self: &Arc<Self>, identity: Option<ParticipantIdentity>) -> Result<()> {
        link_and_read(&self.reader, self.room.as_ref(), identity, |p, t| self.try_read(p, t))
    }

    pub fn handle_event(self: &Arc<Self>, event: &RoomEvent) -> Result<()> {
        route(&self.reader, event, |p, t| self.try_read(p, t))
    }

    fn try_read(
        self: &Arc<Self>,
        participant: &Arc<dyn RemoteParticipant>,
        publication: &Arc<dyn TrackPublication>,
    ) -> Result<bool> {
        let Some(lease) = self.reader.claim(
            participant.identity(),
            publication.source(),
            publication.sid(),
            VIDEO_SOURCES,
        ) else {
            return Ok(false);
        };
        let stream = match publication.open_video_stream() {
            Ok(stream) => stream,
            Err(error) => {
                self.reader.opening_failed(&lease);
                return Err(error.into());
            }
        };
        let owner = self.clone();
        self.reader.spawn(self.spawner.as_ref(), "room video input", async move {
            owner.pump(lease, stream).await;
        });
        Ok(true)
    }

    async fn pump(self: Arc<Self>, lease: Lease, mut stream: Box<dyn StreamSource<VideoFrameRef>>) {
        self.reader.started();
        loop {
            let frame = tokio::select! {
                biased;
                () = lease.stop.notified() => break,
                frame = stream.next() => frame,
            };
            match frame {
                Ok(Some(frame)) => {
                    if !self.reader.deliver(&lease, Ok(frame)) {
                        break;
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    self.reader.deliver(&lease, Err(IoError::from(error)));
                    break;
                }
            }
        }
        let deadline = self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let cleanup = stream.close(deadline).await;
        self.reader.finish(&lease, cleanup);
    }

    pub async fn aclose(&self) -> Closed {
        self.reader
            .close(
                self.clock.as_ref(),
                self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT),
            )
            .await
    }
}

impl VideoInput for ParticipantVideoInput {
    fn label(&self) -> &str {
        "RoomIO"
    }
    fn next_frame<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Result<Option<VideoFrameRef>>> {
        Box::pin(async move { self.reader.recv().await })
    }
    fn on_attached(&self) {
        self.reader.set_attached(true);
    }
    fn on_detached(&self) {
        self.reader.set_attached(false);
    }
}

pub struct ParticipantAudioInput {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    config: AudioInputConfig,
    pre_connect: Option<Arc<PreConnectAudioHandler>>,
    reader: TrackReader<AudioFrame>,
}

impl ParticipantAudioInput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
        config: AudioInputConfig,
        pre_connect: Option<Arc<PreConnectAudioHandler>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            room,
            spawner,
            clock,
            config,
            pre_connect,
            reader: TrackReader::new(lk_agents_abi::AUDIO_QUEUE_FRAMES),
        })
    }

    pub fn reading(&self) -> Option<TrackSid> {
        self.reader.reading()
    }

    pub fn set_participant(self: &Arc<Self>, identity: Option<ParticipantIdentity>) -> Result<()> {
        link_and_read(&self.reader, self.room.as_ref(), identity, |p, t| self.try_read(p, t))
    }

    pub fn handle_event(self: &Arc<Self>, event: &RoomEvent) -> Result<()> {
        route(&self.reader, event, |p, t| self.try_read(p, t))
    }

    fn try_read(
        self: &Arc<Self>,
        participant: &Arc<dyn RemoteParticipant>,
        publication: &Arc<dyn TrackPublication>,
    ) -> Result<bool> {
        let Some(lease) = self.reader.claim(
            participant.identity(),
            publication.source(),
            publication.sid(),
            AUDIO_SOURCES,
        ) else {
            return Ok(false);
        };
        let input_stream = match publication.open_audio_stream(AudioStreamOptions {
            sample_rate: Some(self.config.sample_rate),
            num_channels: Some(self.config.num_channels),
            frame_size_ms: Some(self.config.frame_size_ms),
            noise_cancellation: self.config.noise_cancellation.clone(),
            auto_gain_control: self.config.auto_gain_control,
            participant_identity: participant.identity().clone(),
        }) {
            Ok(stream) => stream,
            Err(error) => {
                self.reader.opening_failed(&lease);
                return Err(error.into());
            }
        };
        let stream = match input_stream.frames() {
            Ok(stream) => stream,
            Err(error) => {
                self.reader.opening_failed(&lease);
                return Err(error.into());
            }
        };
        let wants_pre_connect =
            publication.audio_features().contains(&AudioTrackFeature::PreconnectBuffer)
                && self.reader.replay_pre_connect(
                    participant.identity().clone(),
                    participant.sid().to_owned(),
                    publication.sid().clone(),
                );
        let owner = self.clone();
        let sid = publication.sid().clone();
        self.reader.spawn(self.spawner.as_ref(), "room audio input", async move {
            owner.pump(lease, sid, stream, wants_pre_connect.then_some(input_stream)).await;
        });
        Ok(true)
    }

    async fn pump(
        self: Arc<Self>,
        lease: Lease,
        sid: TrackSid,
        mut stream: Box<dyn StreamSource<AudioFrame>>,
        pre_connect: Option<Arc<dyn AudioInputStream>>,
    ) {
        self.reader.started();
        let reading = async {
            if let Some(input_stream) = pre_connect
                && let Some(handler) = &self.pre_connect
            {
                match handler.wait_for_data(&sid).await {
                    Ok(frames) => {
                        for frame in frames {
                            match input_stream.process_audio(vec![frame]).await {
                                Ok(frames) => {
                                    for frame in frames {
                                        if !self.reader.deliver(&lease, Ok(frame)) {
                                            return;
                                        }
                                    }
                                }
                                Err(error) => {
                                    self.reader.deliver(&lease, Err(error.into()));
                                    return;
                                }
                            }
                            tokio::task::yield_now().await;
                        }
                    }
                    // An absent or expired head doesn't invalidate the live subscription.
                    Err(IoError::PreConnectTimeout(_)) => {}
                    Err(error) => {
                        self.reader.deliver(&lease, Err(error));
                        return;
                    }
                }
            }
            loop {
                match stream.next().await {
                    Ok(Some(frame)) => {
                        if !self.reader.deliver(&lease, Ok(frame)) {
                            return;
                        }
                    }
                    Ok(None) => {
                        self.reader.deliver(
                            &lease,
                            Ok(silence_frame(
                                self.config.sample_rate / 2,
                                self.config.sample_rate,
                                self.config.num_channels,
                            )),
                        );
                        return;
                    }
                    Err(error) => {
                        self.reader.deliver(&lease, Err(IoError::from(error)));
                        return;
                    }
                }
            }
        };
        tokio::select! {
            biased;
            () = lease.stop.notified() => {},
            () = reading => {},
        }
        let deadline = self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let cleanup = stream.close(deadline).await;
        self.reader.finish(&lease, cleanup);
    }

    pub async fn aclose(&self) -> Closed {
        self.reader
            .close(
                self.clock.as_ref(),
                self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT),
            )
            .await
    }
}

impl AudioInput for ParticipantAudioInput {
    fn label(&self) -> &str {
        "RoomIO"
    }
    fn next_frame<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Result<Option<AudioFrame>>> {
        Box::pin(async move { self.reader.recv().await })
    }
    fn on_attached(&self) {
        self.reader.set_attached(true);
    }
    fn on_detached(&self) {
        self.reader.set_attached(false);
    }
}

impl lk_agents_abi::voice::room_owner::RoomAudioInputPort for ParticipantAudioInput {
    fn reading(&self) -> Option<TrackSid> {
        ParticipantAudioInput::reading(self)
    }
    fn aclose<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Closed> {
        Box::pin(async move { ParticipantAudioInput::aclose(self).await })
    }
}

impl lk_agents_abi::voice::room_owner::RoomVideoInputPort for ParticipantVideoInput {
    fn reading(&self) -> Option<TrackSid> {
        ParticipantVideoInput::reading(self)
    }
    fn aclose<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Closed> {
        Box::pin(async move { ParticipantVideoInput::aclose(self).await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn replacing_a_track_rejects_its_buffered_input() {
        let reader = TrackReader::new(8);
        reader.link(Some("user".into()));
        let old = reader
            .claim(&"user".into(), TrackSource::Microphone, &"old".into(), AUDIO_SOURCES)
            .unwrap();
        assert!(reader.deliver(&old, Ok("old audio")));
        let new = reader
            .claim(&"user".into(), TrackSource::Microphone, &"new".into(), AUDIO_SOURCES)
            .unwrap();
        assert!(reader.deliver(&new, Ok("new audio")));
        assert_eq!(reader.recv().await.unwrap(), Some("new audio"));
        assert!(!reader.deliver(&old, Ok("late old audio")));
        reader.ended(&old);
        assert_eq!(reader.reading(), Some("new".into()));
    }

    #[tokio::test]
    async fn reattaching_the_same_input_rejects_its_previous_buffer() {
        let reader = TrackReader::new(8);
        reader.link(Some("user".into()));
        let track = reader
            .claim(&"user".into(), TrackSource::Microphone, &"mic".into(), AUDIO_SOURCES)
            .unwrap();
        assert!(reader.deliver(&track, Ok("previous attachment")));
        reader.set_attached(false);
        assert!(reader.deliver(&track, Ok("detached audio")));
        reader.set_attached(true);
        assert!(reader.deliver(&track, Ok("current attachment")));
        assert_eq!(reader.recv().await.unwrap(), Some("current attachment"));
    }

    #[tokio::test]
    async fn a_track_ending_preserves_its_accepted_audio() {
        let reader = TrackReader::new(8);
        reader.link(Some("user".into()));
        let track = reader
            .claim(&"user".into(), TrackSource::Microphone, &"mic".into(), AUDIO_SOURCES)
            .unwrap();
        assert!(reader.deliver(&track, Ok("last audio")));
        reader.ended(&track);
        assert_eq!(reader.recv().await.unwrap(), Some("last audio"));
    }
}

#[cfg(test)]
mod creation_tests {
    use super::*;
    use std::future::Future;
    use std::task::Poll;
    use std::time::Duration;

    #[tokio::test]
    async fn close_does_not_report_clean_while_an_admitted_reader_is_still_opening() {
        let reader = TrackReader::<u8>::new(8);
        reader.link(Some("user".into()));
        let _lease = reader
            .claim(&"user".into(), TrackSource::Microphone, &"mic".into(), AUDIO_SOURCES)
            .unwrap();
        let clock = lk_agents_abi::VirtualClock::new();
        let mut close = Box::pin(reader.close(&clock, Ts::from_nanos(1_000_000_000)));
        std::future::poll_fn(|cx| {
            assert!(
                close.as_mut().poll(cx).is_pending(),
                "a room source may still arrive and require retirement"
            );
            Poll::Ready(())
        })
        .await;
        clock.advance(Duration::from_secs(1));
        assert_eq!(close.await, Closed::TimedOut);
    }
}

#[cfg(test)]
#[path = "input_tests.rs"]
mod live_tests;
