//! The avatar worker the agent's voice streams to, in its own room, on the production wiring: the
//! agent's `DataStreamAudioReceiver`, the runner (`lk_agents_voice::avatar_runner`, its ready
//! events taken in the run's seeded order), and in place of the user's model and the SDK's
//! `rtc.AVSynchronizer` a generator and a synchronizer that keep real time:
//!
//! - the generator ticks at its frame rate whether or not it has audio, each tick a video frame
//!   and the audio it lip-syncs (up to one frame period), ready after its generation latency;
//!   what a tick took from the buffer is in flight, and a clear drops only what was not taken;
//! - the synchronizer plays audio back to back and blocks a push past its queue, and shows video
//!   on the frame clock from a queue of `fps * queue` frames, as `rtc.AVSynchronizer` does.
//!
//! Frames are the host's: each is an entry in this worker's frame table, which a `MediaRef`
//! releases when dropped, so a frame the runner keeps is an entry left at the close.
//!
//! The two rooms are relayed in the order the agent sent things over its data channel: a segment's
//! chunks as they are written, a `lk.clear_buffer` between them, as a real room delivers them.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, OnceLock, Weak},
    time::Duration,
};

use async_trait::async_trait;
use lk_agents_abi::{
    Clock, MediaRef, RefRegistry, Ts,
    errors::{ApiError, ProviderError},
};
use lk_agents_rtc::{FakeRoom, ParticipantKind, Sent, StreamInfo};
use lk_agents_voice::{
    avatar::{DataStreamAudioReceiver, RPC_CLEAR_BUFFER, Received, ReceiverEnd},
    avatar_runner::{
        AvatarAudioReceiver, AvatarFrame, AvatarIo, AvatarMedia, AvatarRunner, AvatarVideoGenerator,
    },
};
use tokio::sync::mpsc::UnboundedSender;

use crate::record::{AvatarFrameLabel, AvatarStep, Observed, Stamps};

/// `rtc.AVSynchronizer`'s `queue_size_ms`, `AvatarRunner`'s default.
pub const QUEUE_MS: u64 = 100;
/// `_FPSController`'s `max_delay_tolerance_ms`.
const MAX_DELAY_MS: u64 = 300;
/// How long the avatar's audio track takes to be subscribed once published.
const SUBSCRIBE_MS: u64 = 60;

type Log = Arc<Mutex<Vec<(Observed, AvatarStep)>>>;

fn now_ms(clock: &dyn Clock) -> u64 {
    clock.now().saturating_duration_since(Ts::ZERO).as_millis() as u64
}

async fn until_ms(clock: &dyn Clock, ms: u64) {
    clock.wait_until(Ts::ZERO.saturating_add(Duration::from_millis(ms))).await;
}

// ----------------------------------------------------------------------------- the frame table

#[derive(Clone, Copy, Debug)]
enum Entry {
    /// Audio the receiver handed over: its segment, when, and how many samples at what rate.
    Received { segment: u32, from: Observed, samples: u32, rate: u32 },
    /// A frame the generator made, and how long it plays if audio.
    Generated { label: AvatarFrameLabel, ms: u64 },
}

/// The host's side table, for this worker's frames: an entry lives while a `MediaRef` holds it.
#[derive(Default)]
struct Frames {
    entries: Mutex<(u64, HashMap<u64, Entry>)>,
}

impl RefRegistry for Frames {
    fn release(&self, id: u64) {
        self.entries.lock().unwrap().1.remove(&id);
    }
}

impl Frames {
    fn add(self: &Arc<Self>, entry: Entry) -> MediaRef {
        let id = {
            let mut entries = self.entries.lock().unwrap();
            entries.0 += 1;
            let id = entries.0;
            entries.1.insert(id, entry);
            id
        };
        let registry: Arc<dyn RefRegistry> = self.clone();
        MediaRef::new(id, &registry)
    }

    fn get(&self, frame: &MediaRef) -> Option<Entry> {
        self.entries.lock().unwrap().1.get(&frame.id()).copied()
    }

    fn left(&self) -> usize {
        self.entries.lock().unwrap().1.len()
    }
}

fn record(log: &Log, stamps: &Stamps, step: AvatarStep) -> Observed {
    let at = stamps.next();
    log.lock().unwrap().push((at, step));
    at
}

// ----------------------------------------------------------------------------- the receiver

/// `DataStreamAudioReceiver` as a host `AudioReceiver` hands it to the runner (`voice/avatar.py`):
/// a frame becomes a host frame, a segment end `AudioSegmentEnd`, the end of the stream the end
/// of iteration.
struct Receiver {
    rx: Arc<DataStreamAudioReceiver>,
    frames: Arc<Frames>,
    log: Log,
    stamps: Stamps,
    /// The segment being received, and whether any of its audio was.
    segment: Mutex<(u32, bool)>,
    closed: Mutex<bool>,
}

impl Receiver {
    /// A segment end or a clear closes the segment being received.
    fn next_segment(&self) {
        let mut segment = self.segment.lock().unwrap();
        if segment.1 {
            *segment = (segment.0 + 1, false);
        }
    }
}

#[async_trait]
impl AvatarAudioReceiver for Receiver {
    async fn open_audio(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    async fn next_audio(&self) -> Result<Option<AvatarFrame>, ProviderError> {
        match self.rx.recv().await {
            Ok(Received::Frame(frame)) => {
                let segment = {
                    let mut segment = self.segment.lock().unwrap();
                    segment.1 = true;
                    segment.0
                };
                let from = record(&self.log, &self.stamps, AvatarStep::Received { segment });
                let entry = Entry::Received {
                    segment,
                    from,
                    samples: frame.samples_per_channel,
                    rate: frame.sample_rate,
                };
                Ok(Some(AvatarFrame::Audio {
                    frame: self.frames.add(entry),
                    duration: frame.duration(),
                }))
            }
            Ok(Received::SegmentEnd) => {
                record(&self.log, &self.stamps, AvatarStep::ReceivedEnd);
                self.next_segment();
                Ok(Some(AvatarFrame::SegmentEnd))
            }
            Err(ReceiverEnd::Closed) => Ok(None),
            Err(e) => Err(ProviderError::Local(ApiError::new(format!("{e:?}")))),
        }
    }

    async fn notify_playback_started(&self) -> Result<(), ProviderError> {
        record(&self.log, &self.stamps, AvatarStep::ReportedStarted);
        self.rx.notify_playback_started();
        Ok(())
    }

    async fn notify_playback_finished(
        &self,
        position: f64,
        interrupted: bool,
    ) -> Result<(), ProviderError> {
        record(&self.log, &self.stamps, AvatarStep::ReportedFinished { interrupted });
        self.rx.notify_playback_finished(position, interrupted);
        Ok(())
    }

    async fn aclose(&self) -> Result<(), ProviderError> {
        *self.closed.lock().unwrap() = true;
        self.rx.aclose();
        Ok(())
    }
}

// ----------------------------------------------------------------------------- the generator

#[derive(Clone, Copy, Debug)]
enum Buffered {
    Audio { segment: u32, from: Observed, samples: u32, rate: u32 },
    End,
}

#[derive(Default)]
struct GenState {
    buffer: VecDeque<Buffered>,
    /// What the last tick generated and the runner has not taken yet.
    ready: VecDeque<(Option<MediaRef>, Option<AvatarFrameLabel>, Option<u64>, Option<u32>)>,
    next_tick_ms: Option<u64>,
    tick: u64,
    /// The segment of the last audio pushed: what a segment end ends.
    last_segment: Option<u32>,
    closed: bool,
    pulls_after_close: u32,
}

/// A video model at a real frame rate with a real generation latency.
struct Generator {
    period_ms: u64,
    latency_ms: u64,
    clock: Arc<dyn Clock>,
    frames: Arc<Frames>,
    log: Log,
    stamps: Stamps,
    state: Mutex<GenState>,
}

impl Generator {
    /// One tick's frames, from what is buffered: a video frame, the audio it lip-syncs, and the
    /// segment's end when its last audio has gone.
    fn tick(
        &self,
        state: &mut GenState,
    ) -> Vec<(Option<MediaRef>, Option<AvatarFrameLabel>, Option<u64>, Option<u32>)> {
        let tick = state.tick;
        state.tick += 1;
        let (mut taken, mut rate, mut segment, mut from) = (0u32, 0u32, None, None::<Observed>);
        while let Some(Buffered::Audio { segment: s, from: f, samples, rate: r }) =
            state.buffer.front().copied()
        {
            if segment.is_some_and(|seg| seg != s) {
                break;
            }
            let want = (u64::from(r) * self.period_ms / 1000) as u32 - taken;
            let take = samples.min(want);
            taken += take;
            rate = r;
            segment = Some(s);
            from = Some(from.map_or(f, |earliest| earliest.min(f)));
            if take == samples {
                state.buffer.pop_front();
            } else if let Some(Buffered::Audio { samples, .. }) = state.buffer.front_mut() {
                *samples -= take;
            }
            if taken as u64 >= u64::from(r) * self.period_ms / 1000 {
                break;
            }
        }
        let mut out = Vec::new();
        let video = AvatarFrameLabel { video: true, tick, segment, from };
        out.push((
            Some(self.frames.add(Entry::Generated { label: video, ms: 0 })),
            Some(video),
            None,
            None,
        ));
        if taken > 0 {
            let audio = AvatarFrameLabel { video: false, ..video };
            let ms = u64::from(taken) * 1000 / u64::from(rate.max(1));
            out.push((
                Some(self.frames.add(Entry::Generated { label: audio, ms })),
                Some(audio),
                Some(ms),
                None,
            ));
        }
        if matches!(state.buffer.front(), Some(Buffered::End)) {
            state.buffer.pop_front();
            if let Some(segment) = state.last_segment {
                out.push((None, None, None, Some(segment)));
            }
        }
        out
    }

    /// How long until what is buffered has been generated.
    fn due_in(&self) -> Option<Duration> {
        let state = self.state.lock().unwrap();
        let ms: u64 = state
            .buffer
            .iter()
            .map(|b| match b {
                Buffered::Audio { samples, rate, .. } => {
                    u64::from(*samples) * 1000 / u64::from((*rate).max(1))
                }
                Buffered::End => self.period_ms,
            })
            .sum();
        (ms > 0 || !state.ready.is_empty())
            .then(|| Duration::from_millis(ms + self.period_ms + self.latency_ms))
    }
}

#[async_trait]
impl AvatarVideoGenerator for Generator {
    async fn push_audio(&self, frame: Option<MediaRef>) -> Result<(), ProviderError> {
        let mut state = self.state.lock().unwrap();
        match frame.as_ref().and_then(|f| self.frames.get(f)) {
            Some(Entry::Received { segment, from, samples, rate }) => {
                state.last_segment = Some(segment);
                state.buffer.push_back(Buffered::Audio { segment, from, samples, rate });
            }
            Some(Entry::Generated { .. }) => {}
            None => state.buffer.push_back(Buffered::End),
        }
        Ok(())
    }

    async fn clear_buffer(&self) -> Result<(), ProviderError> {
        self.state.lock().unwrap().buffer.clear();
        record(&self.log, &self.stamps, AvatarStep::GeneratorCleared);
        Ok(())
    }

    async fn open_frames(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    async fn next_frame(&self) -> Result<Option<AvatarFrame>, ProviderError> {
        loop {
            let at = {
                let mut state = self.state.lock().unwrap();
                if state.closed {
                    state.pulls_after_close += 1;
                }
                if let Some((frame, label, ms, end)) = state.ready.pop_front() {
                    drop(state);
                    return Ok(Some(match (frame, label) {
                        (Some(frame), Some(label)) => {
                            record(&self.log, &self.stamps, AvatarStep::Generated(label));
                            match ms {
                                Some(ms) => {
                                    AvatarFrame::Audio { frame, duration: ms as f64 / 1000.0 }
                                }
                                None => AvatarFrame::Video(frame),
                            }
                        }
                        _ => {
                            let segment = end.unwrap_or_default();
                            record(&self.log, &self.stamps, AvatarStep::GeneratedEnd { segment });
                            AvatarFrame::SegmentEnd
                        }
                    }));
                }
                let now = now_ms(&*self.clock);
                let at = state.next_tick_ms.unwrap_or(now).max(now);
                state.next_tick_ms = Some(at + self.period_ms);
                at
            };
            until_ms(&*self.clock, at).await;
            // taken from the buffer now: in flight for the generation's latency
            let made = self.tick(&mut self.state.lock().unwrap());
            until_ms(&*self.clock, at + self.latency_ms).await;
            self.state.lock().unwrap().ready.extend(made);
        }
    }
}

// ----------------------------------------------------------------------------- the synchronizer

#[derive(Default)]
struct AvState {
    audio_end_ms: u64,
    /// When each queued video frame shows.
    shows: VecDeque<u64>,
    next_show_ms: Option<u64>,
    closes: u32,
}

/// `rtc.AVSynchronizer` and the avatar's two tracks.
struct Synchronizer {
    period_ms: u64,
    clock: Arc<dyn Clock>,
    frames: Arc<Frames>,
    log: Log,
    stamps: Stamps,
    generator: Arc<Generator>,
    state: Mutex<AvState>,
}

#[async_trait]
impl AvatarMedia for Synchronizer {
    async fn publish(&self) -> Result<(), ProviderError> {
        let at = now_ms(&*self.clock) + SUBSCRIBE_MS;
        until_ms(&*self.clock, at).await;
        Ok(())
    }

    async fn push(&self, frame: MediaRef) -> Result<(), ProviderError> {
        let at = self.stamps.next();
        let Some(Entry::Generated { label, ms }) = self.frames.get(&frame) else {
            return Ok(());
        };
        // When the push returns (a full queue holds it) and when the frame plays, decided now:
        // pushes come one at a time, so nothing else moves the queues meanwhile.
        let (returns_ms, plays_ms) = {
            let now = now_ms(&*self.clock);
            let mut state = self.state.lock().unwrap();
            if label.video {
                let slots = (1000 / self.period_ms * QUEUE_MS / 1000).max(1) as usize;
                state.shows.retain(|show| *show > now);
                let returns = if state.shows.len() >= slots {
                    state.shows[state.shows.len() - slots]
                } else {
                    now
                };
                let next = state.next_show_ms.unwrap_or(returns);
                let show = if returns > next + MAX_DELAY_MS { returns } else { next.max(returns) };
                state.next_show_ms = Some(show + self.period_ms);
                state.shows.push_back(show);
                (returns, show)
            } else {
                let queued = state.audio_end_ms.saturating_sub(now);
                let returns =
                    if queued + ms > QUEUE_MS { state.audio_end_ms + ms - QUEUE_MS } else { now };
                let start = state.audio_end_ms.max(returns);
                state.audio_end_ms = start + ms;
                (returns, start)
            }
        };
        self.log.lock().unwrap().push((at, AvatarStep::Pushed { frame: label, plays_ms }));
        drop(frame);
        until_ms(&*self.clock, returns_ms).await;
        Ok(())
    }

    async fn aclose(&self) -> Result<(), ProviderError> {
        self.state.lock().unwrap().closes += 1;
        self.generator.state.lock().unwrap().closed = true;
        Ok(())
    }
}

// ----------------------------------------------------------------------------- the worker

/// Where the relay has got to on each side.
#[derive(Default)]
struct Relay {
    sent: usize,
    streams: HashMap<usize, UnboundedSender<Vec<u8>>>,
    reports: usize,
}

pub struct FakeAvatarWorker {
    room: Arc<FakeRoom>,
    runner: Arc<AvatarRunner>,
    receiver: Arc<Receiver>,
    generator: Arc<Generator>,
    media: Arc<Synchronizer>,
    frames: Arc<Frames>,
    log: Log,
    stamps: Stamps,
    relay: Mutex<Relay>,
    /// The agent's room, and the avatar's identity in it.
    agent: Arc<FakeRoom>,
    avatar: String,
    relaying: Mutex<Option<tokio::task::AbortHandle>>,
}

impl FakeAvatarWorker {
    pub const AGENT: &'static str = "agent";

    /// The worker in its room `name`, joined to the agent's room as `avatar`, generating at `fps`
    /// with `latency_ms` per frame; the runner takes its ready events in the order `schedule`
    /// seeds.
    #[allow(clippy::too_many_arguments)]
    pub async fn start(
        agent: Arc<FakeRoom>,
        avatar: &str,
        clock: Arc<dyn Clock>,
        stamps: Stamps,
        name: &str,
        schedule: u64,
        fps: u64,
        latency_ms: u64,
    ) -> Arc<FakeAvatarWorker> {
        let room = lk_agents_rtc::FakeRtc::new(clock.clone()).room(name);
        room.add_participant(Self::AGENT, ParticipantKind::Agent);
        let spawner: Arc<dyn lk_agents_abi::Spawner> =
            Arc::new(lk_agents_abi::TokioSpawner::current());
        let (frames, log): (Arc<Frames>, Log) = Default::default();
        let period_ms = 1000 / fps;
        record(&log, &stamps, AvatarStep::Started { period_ms, latency_ms, queue_ms: QUEUE_MS });

        // the receiver's `clear_buffer` event, which the host wires to the runner
        let worker: Arc<OnceLock<Weak<FakeAvatarWorker>>> = Arc::default();
        let on_clear: Arc<dyn Fn() + Send + Sync> = {
            let worker = worker.clone();
            Arc::new(move || {
                if let Some(w) = worker.get().and_then(Weak::upgrade) {
                    record(&w.log, &w.stamps, AvatarStep::Clear);
                    w.receiver.next_segment();
                    w.runner.clear_buffer();
                }
            })
        };
        let rx = Arc::new(DataStreamAudioReceiver::new(
            room.clone(),
            Some(Self::AGENT.into()),
            20,
            1,
            Some(on_clear),
            clock.clone(),
            spawner.clone(),
        ));
        let receiver = Arc::new(Receiver {
            rx: rx.clone(),
            frames: frames.clone(),
            log: log.clone(),
            stamps: stamps.clone(),
            segment: Mutex::default(),
            closed: Mutex::default(),
        });
        let generator = Arc::new(Generator {
            period_ms,
            latency_ms,
            clock: clock.clone(),
            frames: frames.clone(),
            log: log.clone(),
            stamps: stamps.clone(),
            state: Mutex::default(),
        });
        let media = Arc::new(Synchronizer {
            period_ms,
            clock: clock.clone(),
            frames: frames.clone(),
            log: log.clone(),
            stamps: stamps.clone(),
            generator: generator.clone(),
            state: Mutex::default(),
        });
        let io = AvatarIo {
            receiver: receiver.clone(),
            generator: generator.clone(),
            media: media.clone(),
        };
        let mut order = fastrand::Rng::with_seed(schedule ^ 0xa7a7);
        let runner = Arc::new(AvatarRunner::scheduled(
            room.clone(),
            io,
            true,
            clock,
            spawner,
            move |ready| order.usize(..ready),
        ));
        let me = Arc::new(FakeAvatarWorker {
            room,
            runner,
            receiver,
            generator,
            media,
            frames,
            log,
            stamps,
            relay: Mutex::default(),
            agent,
            avatar: avatar.to_owned(),
            relaying: Mutex::default(),
        });
        let _ = worker.set(Arc::downgrade(&me));
        // what either side sends crosses as it is sent
        let relaying = {
            let weak = Arc::downgrade(&me);
            let mut agent_sent = me.agent.local_participant().sent_count();
            let mut worker_sent = me.room.local_participant().sent_count();
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        changed = agent_sent.changed() => if changed.is_err() { return },
                        changed = worker_sent.changed() => if changed.is_err() { return },
                    }
                    match weak.upgrade() {
                        Some(worker) => worker.relay(),
                        None => return,
                    }
                }
            })
            .abort_handle()
        };
        *me.relaying.lock().unwrap() = Some(relaying);
        // `AvatarRunner.start()`: the receiver's own start, then the runner's
        let _ = rx.start().await;
        let _ = me.runner.start().await;
        me
    }

    /// Carry what each side sent since the last relay to the other, in the order it was sent: the
    /// agent's segment streams and its clears to the worker, the worker's reports to the agent.
    pub fn relay(&self) {
        let avatar = self.avatar.as_str();
        let mut relay = self.relay.lock().unwrap();
        let local = self.agent.local_participant();
        let sent = local.sent_since(relay.sent);
        relay.sent += sent.len();
        for item in sent {
            match item {
                Sent::StreamOpened(index) => {
                    let Some(stream) = local.byte_streams().into_iter().nth(index) else {
                        continue;
                    };
                    if !stream.opts.destination_identities.iter().any(|d| d.0 == avatar) {
                        continue;
                    }
                    let info = StreamInfo {
                        id: stream.opts.name.clone(),
                        topic: stream.opts.topic.clone(),
                        mime_type: "application/octet-stream".into(),
                        total_length: None,
                        attributes: stream.opts.attributes.clone(),
                        timestamp: 0,
                        attachments: Vec::new(),
                    };
                    if let Ok(tx) = self.room.open_byte_stream(Self::AGENT, info) {
                        relay.streams.insert(index, tx);
                    }
                }
                Sent::StreamChunk(index, chunk) => {
                    if let Some(tx) = relay.streams.get(&index) {
                        let _ = tx.send(chunk);
                    }
                }
                Sent::StreamClosed(index) => {
                    relay.streams.remove(&index);
                }
                Sent::Rpc { destination, method, .. } => {
                    if destination.0 == avatar && method == RPC_CLEAR_BUFFER {
                        self.room.local_participant().invoke_rpc(RPC_CLEAR_BUFFER, Self::AGENT, "");
                    }
                }
            }
        }
        let reports = self.room.local_participant().sent_since(relay.reports);
        relay.reports += reports.len();
        for item in reports {
            if let Sent::Rpc { method, payload, .. } = item {
                local.invoke_rpc(&method, avatar, &payload);
            }
        }
    }

    /// How much has been relayed, which only grows.
    pub fn progress(&self) -> (usize, usize) {
        let relay = self.relay.lock().unwrap();
        (relay.sent, relay.reports)
    }

    /// How long until the generator has played what it holds; `None` when it holds nothing.
    pub fn due_in(&self) -> Option<Duration> {
        self.generator.due_in()
    }

    /// What the worker did since the last look, in the order it happened.
    pub fn drain(&self) -> Vec<(Observed, AvatarStep)> {
        let mut steps = std::mem::take(&mut *self.log.lock().unwrap());
        steps.sort_by_key(|(at, _)| at.seq);
        steps
    }

    /// `AvatarRunner.aclose()`, then what is left of it.
    pub async fn close(&self) {
        record(&self.log, &self.stamps, AvatarStep::CloseAsked);
        let _ = self.runner.aclose().await;
        if let Some(relaying) = self.relaying.lock().unwrap().take() {
            relaying.abort();
        }
        self.relay.lock().unwrap().streams.clear();
        let pulls_after_close = self.generator.state.lock().unwrap().pulls_after_close;
        let media_closes = self.media.state.lock().unwrap().closes;
        // what the generator itself still holds is the user's model's, not the runner's
        let held =
            self.generator.state.lock().unwrap().ready.iter().filter(|r| r.0.is_some()).count();
        let step = AvatarStep::Closed {
            frames_left: self.frames.left().saturating_sub(held),
            pulls_after_close,
            media_closes,
            receiver_closed: *self.receiver.closed.lock().unwrap(),
            io_in_flight: self.runner.io_in_flight(),
        };
        record(&self.log, &self.stamps, step);
    }
}
