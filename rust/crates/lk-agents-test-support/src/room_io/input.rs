//! Simulated subscriptions execute the same track and attachment admission as production.

use super::pre_connect_audio::PreConnectAudioHandler;
use lk_agents_abi::voice::io::{AudioInput, IoError, Result, VideoInput};
use lk_agents_abi::voice::room_io::AudioInputConfig;
use lk_agents_abi::voice::rtc::{
    AudioStreamOptions, AudioTrackFeature, ParticipantIdentity, RemoteParticipant, RoomEvent,
    RtcRoom, TrackPublication, TrackSid, TrackSource, VideoFrameRef, silence_frame,
};
use lk_agents_abi::{AudioFrame, BoundedLossy, Clock, Closed, LossyTx, Spawner, TaskSet};
use lk_agents_voice::room_io::input_state::{InputIdentity, TrackSelection};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

struct Delivery<T> {
    identity: InputIdentity,
    item: Result<T>,
}
struct Input<T> {
    selection: Mutex<TrackSelection>,
    sender: LossyTx<Delivery<T>>,
    receiver: tokio::sync::Mutex<BoundedLossy<Delivery<T>>>,
    changed: Notify,
    tasks: Arc<TaskSet>,
    cleanup: Mutex<Option<Closed>>,
    outcome: Mutex<Closed>,
}
impl<T> Input<T> {
    fn new(capacity: usize) -> Self {
        let (sender, receiver) = BoundedLossy::channel(capacity);
        Self {
            selection: Mutex::default(),
            sender,
            receiver: tokio::sync::Mutex::new(receiver),
            changed: Notify::new(),
            tasks: Arc::default(),
            cleanup: Mutex::new(None),
            outcome: Mutex::new(Closed::Clean),
        }
    }
    fn deliver(&self, generation: u64, item: Result<T>) {
        if let Some(identity) = self.selection.lock().unwrap().admit(generation) {
            self.sender.push(Delivery { identity, item });
        }
    }
    async fn next(&self) -> Result<Option<T>> {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.selection.lock().unwrap().is_closed() {
                return Ok(None);
            }
            let mut receiver = self.receiver.lock().await;
            let delivery = tokio::select! { item = receiver.recv() => item, () = &mut changed => {
                if self.selection.lock().unwrap().is_closed() {return Ok(None)}
                continue;
            }};
            let Some(delivery) = delivery else { return Ok(None) };
            if self.selection.lock().unwrap().accepts(delivery.identity) {
                return delivery.item.map(Some);
            }
        }
    }
    async fn close(&self, clock: &dyn Clock) -> Closed {
        if let Some(outcome) = *self.cleanup.lock().unwrap() {
            return outcome;
        }
        self.selection.lock().unwrap().close();
        self.changed.notify_waiters();
        {
            let mut receiver = self.receiver.lock().await;
            receiver.close();
            while receiver.recv().await.is_some() {}
        }
        let deadline = clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        let outcome = if self.tasks.wait_empty(clock, Some(deadline)).await {
            Closed::Clean
        } else {
            Closed::TimedOut
        };
        *self.cleanup.lock().unwrap().get_or_insert(outcome.and(*self.outcome.lock().unwrap()))
    }
    fn record_cleanup(&self, outcome: Closed) {
        let mut recorded = self.outcome.lock().unwrap();
        *recorded = recorded.and(outcome);
    }

    fn attached(&self, attached: bool) {
        self.selection.lock().unwrap().set_attached(attached);
        self.changed.notify_waiters();
    }
}

pub struct ParticipantAudioInput {
    input: Input<AudioFrame>,
    room: Arc<dyn RtcRoom>,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    config: AudioInputConfig,
    pre_connect: Option<Arc<PreConnectAudioHandler>>,
}
impl ParticipantAudioInput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        config: AudioInputConfig,
        pre_connect: Option<Arc<PreConnectAudioHandler>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            input: Input::new(lk_agents_abi::AUDIO_QUEUE_FRAMES),
            room,
            clock,
            spawner,
            config,
            pre_connect,
        })
    }
    pub fn set_participant(self: &Arc<Self>, who: Option<ParticipantIdentity>) -> Result<()> {
        self.input.selection.lock().unwrap().link(who.clone());
        self.input.changed.notify_waiters();
        if let Some(who) = who
            && let Some(participant) = self.room.participant(&who)?
        {
            for track in participant.track_publications()? {
                if self.subscribe(&participant, &track)? {
                    break;
                }
            }
        }
        Ok(())
    }
    pub fn handle_event(self: &Arc<Self>, event: &RoomEvent) -> Result<()> {
        match event {
            RoomEvent::TrackSubscribed { participant, publication } => {
                self.subscribe(participant, publication)?;
            }
            RoomEvent::TrackUnsubscribed { participant, publication }
            | RoomEvent::TrackUnpublished { participant, publication } => {
                if self
                    .input
                    .selection
                    .lock()
                    .unwrap()
                    .release(participant.identity(), publication.sid())
                {
                    self.input.changed.notify_waiters();
                    for track in participant.track_publications()? {
                        if track.sid() != publication.sid()
                            && self.subscribe(participant, &track)?
                        {
                            break;
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn subscribe(
        self: &Arc<Self>,
        participant: &Arc<dyn RemoteParticipant>,
        track: &Arc<dyn TrackPublication>,
    ) -> Result<bool> {
        let generation = self.input.selection.lock().unwrap().claim(
            participant.identity(),
            track.source(),
            track.sid(),
            &[TrackSource::Microphone],
        );
        let Some(generation) = generation else { return Ok(false) };
        self.input.changed.notify_waiters();
        let source = match track.open_audio_stream(AudioStreamOptions {
            participant_identity: participant.identity().clone(),
            sample_rate: Some(self.config.sample_rate),
            num_channels: Some(self.config.num_channels),
            frame_size_ms: Some(self.config.frame_size_ms),
            noise_cancellation: self.config.noise_cancellation.clone(),
            auto_gain_control: self.config.auto_gain_control,
        }) {
            Ok(source) => source,
            Err(error) => {
                self.input.selection.lock().unwrap().ended(generation);
                return Err(error.into());
            }
        };
        let mut frames = match source.frames() {
            Ok(frames) => frames,
            Err(error) => {
                self.input.selection.lock().unwrap().ended(generation);
                return Err(error.into());
            }
        };
        let owner = self.clone();
        let sid = track.sid().clone();
        let replay = track.audio_features().contains(&AudioTrackFeature::PreconnectBuffer)
            && self.input.selection.lock().unwrap().replay_pre_connect(
                participant.identity().clone(),
                participant.sid().into(),
                sid.clone(),
                lk_agents_voice::room_io::pre_connect_state::MAX_TRACKS,
            );
        self.input.tasks.spawn(self.spawner.as_ref(), "simulated room audio input", async move {
            let mut replay_failed = false;
            let replay_identity = owner.input.selection.lock().unwrap().admit(generation);
            if replay && let Some(head) = &owner.pre_connect {
                let changed = owner.input.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                let buffered = if owner.input.selection.lock().unwrap().producing(generation) {
                    tokio::select! {
                        data = head.wait_for_data(&sid) => Some(data),
                        () = &mut changed => None,
                    }
                } else {
                    None
                };
                match buffered {
                    Some(Ok(buffer)) => {
                        for frame in buffer {
                            if !replay_identity.is_some_and(|identity| {
                                owner.input.selection.lock().unwrap().accepts(identity)
                            }) {
                                break;
                            }
                            match source.process_audio(vec![frame]).await {
                                Ok(processed) => {
                                    for frame in processed {
                                        if let Some(identity) = replay_identity
                                            && owner
                                                .input
                                                .selection
                                                .lock()
                                                .unwrap()
                                                .accepts(identity)
                                        {
                                            owner
                                                .input
                                                .sender
                                                .push(Delivery { identity, item: Ok(frame) });
                                        }
                                    }
                                }
                                Err(error) => {
                                    owner.input.deliver(generation, Err(error.into()));
                                    replay_failed = true;
                                    break;
                                }
                            }
                        }
                    }
                    Some(Err(IoError::PreConnectTimeout(_))) => {}
                    Some(Err(error)) => {
                        owner.input.deliver(generation, Err(error));
                        replay_failed = true;
                    }
                    None => {
                        replay_failed = !owner.input.selection.lock().unwrap().producing(generation)
                    }
                }
            }
            loop {
                let changed = owner.input.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if replay_failed || !owner.input.selection.lock().unwrap().producing(generation) {
                    break;
                }
                let frame = tokio::select! {frame=frames.next()=>frame,()=&mut changed=>continue};
                match frame {
                    Ok(Some(frame)) => owner.input.deliver(generation, Ok(frame)),
                    Ok(None) => {
                        owner.input.deliver(
                            generation,
                            Ok(silence_frame(
                                owner.config.sample_rate / 2,
                                owner.config.sample_rate,
                                owner.config.num_channels,
                            )),
                        );
                        break;
                    }
                    Err(error) => {
                        owner.input.deliver(generation, Err(error.into()));
                        break;
                    }
                }
            }
            owner.input.selection.lock().unwrap().ended(generation);
            let cleanup = frames
                .close(owner.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
                .await;
            owner.input.record_cleanup(cleanup);
        });
        Ok(true)
    }
}
impl AudioInput for ParticipantAudioInput {
    fn label(&self) -> &str {
        "RoomIO"
    }
    fn next_frame(&self) -> lk_agents_abi::BoxFuture<'_, Result<Option<AudioFrame>>> {
        Box::pin(self.input.next())
    }
    fn on_attached(&self) {
        self.input.attached(true)
    }
    fn on_detached(&self) {
        self.input.attached(false)
    }
}
impl lk_agents_abi::voice::room_owner::RoomAudioInputPort for ParticipantAudioInput {
    fn reading(&self) -> Option<TrackSid> {
        self.input.selection.lock().unwrap().reading().cloned()
    }
    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, Closed> {
        Box::pin(async { self.input.close(self.clock.as_ref()).await })
    }
}

pub struct ParticipantVideoInput {
    input: Input<VideoFrameRef>,
    room: Arc<dyn RtcRoom>,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
}
impl ParticipantVideoInput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Arc<Self> {
        Arc::new(Self {
            input: Input::new(lk_agents_abi::voice::rtc::VIDEO_QUEUE_FRAMES),
            room,
            clock,
            spawner,
        })
    }
    pub fn set_participant(self: &Arc<Self>, who: Option<ParticipantIdentity>) -> Result<()> {
        self.input.selection.lock().unwrap().link(who.clone());
        self.input.changed.notify_waiters();
        if let Some(who) = who
            && let Some(p) = self.room.participant(&who)?
        {
            for t in p.track_publications()? {
                if self.subscribe(&p, &t)? {
                    break;
                }
            }
        }
        Ok(())
    }
    pub fn handle_event(self: &Arc<Self>, event: &RoomEvent) -> Result<()> {
        match event {
            RoomEvent::TrackSubscribed { participant, publication } => {
                self.subscribe(participant, publication)?;
            }
            RoomEvent::TrackUnsubscribed { participant, publication }
            | RoomEvent::TrackUnpublished { participant, publication } => {
                self.input
                    .selection
                    .lock()
                    .unwrap()
                    .release(participant.identity(), publication.sid());
                self.input.changed.notify_waiters();
            }
            _ => {}
        }
        Ok(())
    }
    fn subscribe(
        self: &Arc<Self>,
        p: &Arc<dyn RemoteParticipant>,
        t: &Arc<dyn TrackPublication>,
    ) -> Result<bool> {
        let generation = self.input.selection.lock().unwrap().claim(
            p.identity(),
            t.source(),
            t.sid(),
            &[TrackSource::Camera, TrackSource::Screenshare],
        );
        let Some(generation) = generation else { return Ok(false) };
        self.input.changed.notify_waiters();
        let mut source = match t.open_video_stream() {
            Ok(source) => source,
            Err(error) => {
                self.input.selection.lock().unwrap().ended(generation);
                return Err(error.into());
            }
        };
        let owner = self.clone();
        self.input.tasks.spawn(self.spawner.as_ref(), "simulated room video input", async move {
            loop {
                let changed = owner.input.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if !owner.input.selection.lock().unwrap().producing(generation) {
                    break;
                }
                let frame = tokio::select! {frame=source.next()=>frame,()=&mut changed=>continue};
                match frame {
                    Ok(Some(frame)) => owner.input.deliver(generation, Ok(frame)),
                    Ok(None) => break,
                    Err(error) => {
                        owner.input.deliver(generation, Err(error.into()));
                        break;
                    }
                }
            }
            owner.input.selection.lock().unwrap().ended(generation);
            let cleanup = source
                .close(owner.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
                .await;
            owner.input.record_cleanup(cleanup);
        });
        Ok(true)
    }
}
impl VideoInput for ParticipantVideoInput {
    fn label(&self) -> &str {
        "RoomIO"
    }
    fn next_frame(&self) -> lk_agents_abi::BoxFuture<'_, Result<Option<VideoFrameRef>>> {
        Box::pin(self.input.next())
    }
    fn on_attached(&self) {
        self.input.attached(true)
    }
    fn on_detached(&self) {
        self.input.attached(false)
    }
}
impl lk_agents_abi::voice::room_owner::RoomVideoInputPort for ParticipantVideoInput {
    fn reading(&self) -> Option<TrackSid> {
        self.input.selection.lock().unwrap().reading().cloned()
    }
    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, Closed> {
        Box::pin(async { self.input.close(self.clock.as_ref()).await })
    }
}
