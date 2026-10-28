//! The simulation's executor of voice's track selection over supplied-clock peers.

use super::pre_connect_audio::PreConnectAudioHandler;
use lk_agents_abi::voice::io::{AudioInput, IoError, Result, VideoInput};
use lk_agents_abi::voice::room_io::AudioInputConfig;
use lk_agents_abi::voice::rtc::{
    AudioStreamOptions, AudioTrackFeature, ParticipantIdentity, RemoteParticipant, RoomEvent,
    RtcRoom, TrackPublication, TrackSid, VideoFrameRef,
};
use lk_agents_abi::{
    AudioFrame, BoundedLossy, Clock, Closed, HOST_TEARDOWN_TIMEOUT, LossyTx, Pushed, Spawner,
    SpawnerExt, StreamSource,
};
use lk_agents_voice::room_io::input_state::{
    AUDIO_SOURCES, Admit, InputIdentity, Open, Owner, Publication, TrackReaders, VIDEO_SOURCES,
    end_of_track,
};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

struct Delivery<T> {
    identity: InputIdentity,
    item: Result<T>,
}

struct Input<T> {
    readers: Mutex<(TrackReaders, Option<LossyTx<Delivery<T>>>)>,
    queue: tokio::sync::Mutex<BoundedLossy<Delivery<T>>>,
    changed: Notify,
}

impl<T> Input<T> {
    fn new(sources: &'static [lk_agents_abi::voice::rtc::TrackSource], capacity: usize) -> Self {
        let (tx, rx) = BoundedLossy::channel(capacity);
        Self {
            readers: Mutex::new((TrackReaders::new(sources), Some(tx))),
            queue: tokio::sync::Mutex::new(rx),
            changed: Notify::new(),
        }
    }

    fn decide<R>(&self, decision: impl FnOnce(&mut TrackReaders) -> R) -> R {
        let outcome = decision(&mut self.readers.lock().unwrap().0);
        self.changed.notify_waiters();
        outcome
    }

    fn deliver(&self, generation: u64, item: Result<T>) -> bool {
        let readers = self.readers.lock().unwrap();
        match readers.0.admit(generation) {
            Admit::Stop => false,
            Admit::Drop => true,
            Admit::Deliver(identity) => readers
                .1
                .as_ref()
                .is_some_and(|tx| tx.push(Delivery { identity, item }) != Pushed::Closed),
        }
    }

    /// Resolves once voice retires the reader of `generation`.
    async fn retired(&self, generation: u64) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if !self.readers.lock().unwrap().0.producing(generation) {
                return;
            }
            changed.await;
        }
    }

    async fn next(&self) -> Result<Option<T>> {
        let mut queue = self.queue.lock().await;
        while let Some(delivery) = queue.recv().await {
            if self.readers.lock().unwrap().0.accepts(delivery.identity) {
                return delivery.item.map(Some);
            }
        }
        Ok(None)
    }

    async fn close(&self, clock: &dyn Clock) -> Closed {
        {
            let mut readers = self.readers.lock().unwrap();
            readers.0.close();
            readers.1.take();
        }
        self.changed.notify_waiters();
        let settled = async {
            loop {
                let changed = self.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if let Some(outcome) = self.readers.lock().unwrap().0.settled() {
                    return outcome;
                }
                changed.await;
            }
        };
        lk_agents_abi::timeout(clock, HOST_TEARDOWN_TIMEOUT, settled)
            .await
            .unwrap_or(Closed::TimedOut)
    }

    async fn pump(
        &self,
        clock: &dyn Clock,
        generation: u64,
        mut stream: Box<dyn StreamSource<T>>,
        mut end: Option<T>,
    ) {
        loop {
            let frame = tokio::select! {
                biased;
                () = self.retired(generation) => break,
                frame = stream.next() => frame,
            };
            let delivered = match frame {
                Ok(Some(frame)) => self.deliver(generation, Ok(frame)),
                Ok(None) => {
                    if let Some(frame) = end.take() {
                        self.deliver(generation, Ok(frame));
                    }
                    false
                }
                Err(error) => {
                    self.deliver(generation, Err(error.into()));
                    false
                }
            };
            if !delivered {
                break;
            }
        }
        let cleanup = stream.close(clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
        self.decide(|readers| readers.retired(generation, cleanup));
    }
}

fn listed(
    participant: &dyn RemoteParticipant,
) -> Result<(Vec<Publication>, Vec<Arc<dyn TrackPublication>>)> {
    let handles = participant.track_publications()?;
    let listed = handles
        .iter()
        .map(|track| {
            Ok(Publication {
                sid: track.sid().clone(),
                source: track.source(),
                subscribed: track.is_subscribed()?,
            })
        })
        .collect::<Result<_>>()?;
    Ok((listed, handles))
}

/// Which track voice opens, after a link or a room event.
fn select<T>(
    input: &Input<T>,
    room: &dyn RtcRoom,
    change: Change<'_>,
) -> Result<Option<(Arc<dyn RemoteParticipant>, Arc<dyn TrackPublication>, Open)>> {
    let (participant, handles, open) = match change {
        Change::Link(who) => {
            let participant = match &who {
                Some(who) => room.participant(who)?,
                None => None,
            };
            let (listed, handles) = match &participant {
                Some(p) => listed(p.as_ref())?,
                None => Default::default(),
            };
            let owner =
                participant.as_ref().map(|p| Owner { identity: p.identity(), sid: p.sid() });
            let open = input.decide(|readers| readers.link(who.clone(), owner, &listed));
            (participant, handles, open)
        }
        Change::Event(RoomEvent::TrackSubscribed { participant, publication }) => {
            let owner = Owner { identity: participant.identity(), sid: participant.sid() };
            let track = Publication {
                sid: publication.sid().clone(),
                source: publication.source(),
                subscribed: true,
            };
            let open = input.decide(|readers| readers.subscribed(owner, &track));
            (Some(participant.clone()), vec![publication.clone()], open)
        }
        Change::Event(
            RoomEvent::TrackUnpublished { participant, publication }
            | RoomEvent::TrackUnsubscribed { participant, publication },
        ) => {
            let unsubscribed = matches!(change, Change::Event(RoomEvent::TrackUnsubscribed { .. }));
            let (listed, handles) = listed(participant.as_ref())?;
            let owner = Owner { identity: participant.identity(), sid: participant.sid() };
            let open = input
                .decide(|readers| readers.removed(owner, publication.sid(), unsubscribed, &listed));
            (Some(participant.clone()), handles, open)
        }
        Change::Event(_) => return Ok(None),
    };
    let (Some(participant), Some(open)) = (participant, open) else { return Ok(None) };
    match handles.into_iter().find(|track| *track.sid() == open.sid) {
        Some(track) => Ok(Some((participant, track, open))),
        None => {
            input.decide(|readers| readers.retired(open.generation, Closed::Clean));
            Ok(None)
        }
    }
}

enum Change<'a> {
    Link(Option<ParticipantIdentity>),
    Event(&'a RoomEvent),
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
            input: Input::new(AUDIO_SOURCES, lk_agents_abi::AUDIO_QUEUE_FRAMES),
            room,
            clock,
            spawner,
            config,
            pre_connect,
        })
    }

    pub fn set_participant(self: &Arc<Self>, who: Option<ParticipantIdentity>) -> Result<()> {
        self.open(Change::Link(who))
    }

    pub fn handle_event(self: &Arc<Self>, event: &RoomEvent) -> Result<()> {
        self.open(Change::Event(event))
    }

    fn open(self: &Arc<Self>, change: Change<'_>) -> Result<()> {
        let Some((participant, track, open)) = select(&self.input, self.room.as_ref(), change)?
        else {
            return Ok(());
        };
        let opened = track
            .open_audio_stream(AudioStreamOptions {
                participant_identity: participant.identity().clone(),
                sample_rate: Some(self.config.sample_rate),
                num_channels: Some(self.config.num_channels),
                frame_size_ms: Some(self.config.frame_size_ms),
                noise_cancellation: self.config.noise_cancellation.clone(),
                auto_gain_control: self.config.auto_gain_control,
            })
            .and_then(|source| Ok((source.frames()?, source)));
        let (frames, source) = match opened {
            Ok(opened) => opened,
            Err(error) => {
                self.input.decide(|readers| readers.retired(open.generation, Closed::Clean));
                return Err(error.into());
            }
        };
        let owner = Owner { identity: participant.identity(), sid: participant.sid() };
        let replay = track.audio_features().contains(&AudioTrackFeature::PreconnectBuffer)
            && self.input.decide(|readers| readers.replay_pre_connect(owner, &open.sid));
        let input = self.clone();
        self.spawner.spawn_unboxed(async move {
            let generation = open.generation;
            let mut live = true;
            if replay && let Some(head) = &input.pre_connect {
                let buffered = tokio::select! {
                    biased;
                    () = input.input.retired(generation) => None,
                    data = head.wait_for_data(&open.sid) => Some(data),
                };
                live = match buffered {
                    None => false,
                    Some(Err(IoError::PreConnectTimeout(_))) => true,
                    Some(Err(error)) => {
                        input.input.deliver(generation, Err(error));
                        false
                    }
                    Some(Ok(frames)) => {
                        let mut replayed = true;
                        for frame in frames {
                            replayed = match source.process_audio(vec![frame]).await {
                                Ok(processed) => processed
                                    .into_iter()
                                    .all(|frame| input.input.deliver(generation, Ok(frame))),
                                Err(error) => {
                                    input.input.deliver(generation, Err(error.into()));
                                    false
                                }
                            };
                            if !replayed {
                                break;
                            }
                        }
                        replayed
                    }
                };
            }
            if !live {
                let cleanup =
                    frames.close(input.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
                input.input.decide(|readers| readers.retired(generation, cleanup));
                return;
            }
            let end = end_of_track(input.config.sample_rate, input.config.num_channels);
            input.input.pump(input.clock.as_ref(), generation, frames, Some(end)).await;
        });
        Ok(())
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
        self.input.decide(|readers| readers.set_attached(true));
    }
    fn on_detached(&self) {
        self.input.decide(|readers| readers.set_attached(false));
    }
}

impl lk_agents_abi::voice::room_owner::RoomAudioInputPort for ParticipantAudioInput {
    fn reading(&self) -> Option<TrackSid> {
        self.input.readers.lock().unwrap().0.reading().cloned()
    }
    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, Closed> {
        Box::pin(self.input.close(self.clock.as_ref()))
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
            input: Input::new(VIDEO_SOURCES, lk_agents_abi::voice::rtc::VIDEO_QUEUE_FRAMES),
            room,
            clock,
            spawner,
        })
    }

    pub fn set_participant(self: &Arc<Self>, who: Option<ParticipantIdentity>) -> Result<()> {
        self.open(Change::Link(who))
    }

    pub fn handle_event(self: &Arc<Self>, event: &RoomEvent) -> Result<()> {
        self.open(Change::Event(event))
    }

    fn open(self: &Arc<Self>, change: Change<'_>) -> Result<()> {
        let Some((_, track, open)) = select(&self.input, self.room.as_ref(), change)? else {
            return Ok(());
        };
        let frames = match track.open_video_stream() {
            Ok(frames) => frames,
            Err(error) => {
                self.input.decide(|readers| readers.retired(open.generation, Closed::Clean));
                return Err(error.into());
            }
        };
        let input = self.clone();
        self.spawner.spawn_unboxed(async move {
            input.input.pump(input.clock.as_ref(), open.generation, frames, None).await;
        });
        Ok(())
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
        self.input.decide(|readers| readers.set_attached(true));
    }
    fn on_detached(&self) {
        self.input.decide(|readers| readers.set_attached(false));
    }
}

impl lk_agents_abi::voice::room_owner::RoomVideoInputPort for ParticipantVideoInput {
    fn reading(&self) -> Option<TrackSid> {
        self.input.readers.lock().unwrap().0.reading().cloned()
    }
    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, Closed> {
        Box::pin(self.input.close(self.clock.as_ref()))
    }
}
