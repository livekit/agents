use crate::room_io::_pre_connect_audio::PreConnectAudioHandler;
use crate::room_io::decide_track;
use crate::room_io::types::{AUDIO_INPUT_SOURCES, AudioInputConfig};
use async_trait::async_trait;
use lk_agents_abi::AbortHandle;
use lk_agents_abi::voice::io::AudioInput;
#[cfg(feature = "python")]
use lk_agents_abi::voice::io::VideoInput;
use lk_agents_abi::{AudioFrame, BoundedLossy, LossyTx, Pushed, Spawner, SpawnerExt, Ts};
use lk_agents_media::{AudioFormat, FrameResampler, Quality};
use lk_agents_rtc::{
    AudioStreamOptions, AudioTrackFeature, ParticipantIdentity, RemoteParticipant, RoomEvent,
    RtcRoom, TrackPublication, TrackSid, TrackSource, VideoFrameRef, silence_frame,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

const VIDEO_INPUT_SOURCES: &[TrackSource] = &[TrackSource::Camera, TrackSource::Screenshare];

/// `_ParticipantVideoInputStream`: the user's camera into the session. The audio leg's
/// `TrackReader`, but two accepted sources, no silence flush at track end, and no options. The
/// queue is short and drop-oldest: each entry pins a decoded host frame.
pub struct ParticipantVideoInput {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    reader: TrackReader<VideoFrameRef>,
}

/// `session.input.video` when the room is the source, as `_core.RoomVideoInput`: built only by
/// the session (the leg shares the audio leg's participant link), so there is no constructor.
#[lk_agents_abi::host(object(name = "RoomVideoInput", cell = "arc", as_dyn(VideoInput = as_video_input)))]
impl ParticipantVideoInput {
    /// The leg itself where a session takes an `Arc<dyn VideoInput>`, never proxied back.
    pub fn as_video_input(self: &Arc<Self>) -> Arc<dyn lk_agents_abi::voice::io::VideoInput> {
        Arc::clone(self) as Arc<dyn lk_agents_abi::voice::io::VideoInput>
    }

    pub fn new(room: Arc<dyn RtcRoom>, spawner: Arc<dyn Spawner>) -> Arc<Self> {
        Arc::new(ParticipantVideoInput {
            room,
            spawner,
            reader: TrackReader::new(BoundedLossy::channel(lk_agents_rtc::VIDEO_QUEUE_FRAMES)),
        })
    }

    /// The track currently being read, if any.
    #[host(getter)]
    pub fn reading(&self) -> Option<TrackSid> {
        self.reader.reading()
    }

    /// Switch to a different participant, or to nobody.
    pub fn set_participant(self: &Arc<Self>, identity: Option<ParticipantIdentity>) {
        self.reader.link_and_read(&*self.room, identity, |p, t| self.try_read(p, t));
    }

    /// Route one room event. `track_subscribed` and `track_unpublished` are the two that matter.
    pub fn handle_event(self: &Arc<Self>, event: &RoomEvent) {
        self.reader.route(event, |p, t| self.try_read(p, t));
    }

    /// `_on_track_available`; returns whether this track became the one read. A newly subscribed
    /// acceptable track replaces the current one (screenshare replaces the face), as the reference does.
    fn try_read(
        self: &Arc<Self>,
        participant: &Arc<dyn RemoteParticipant>,
        publication: &Arc<dyn TrackPublication>,
    ) -> bool {
        let Some(lease) = self.reader.claim(
            participant.identity(),
            publication.source(),
            publication.sid(),
            VIDEO_INPUT_SOURCES,
        ) else {
            return false;
        };
        let stream = publication.video_stream();
        let this = Arc::clone(self);
        let handle = self.spawner.spawn_unboxed(async move {
            this.pump(lease, stream).await;
        });
        self.reader.keep(handle);
        true
    }

    /// Read one track until it ends, is superseded, or the consumer hangs up. Returning drops the
    /// stream, which releases the subscription; hence the stop signal.
    async fn pump(self: Arc<Self>, lease: Lease, mut stream: BoundedLossy<VideoFrameRef>) {
        loop {
            let received = tokio::select! {
                // Stopping wins over a simultaneous frame.
                biased;
                () = lease.stopped() => None,
                frame = stream.recv() => frame,
            };
            let Some(frame) = received else { return };
            if !self.reader.deliver(&lease, frame) {
                return;
            }
        }
    }

    /// The next frame, or `None` once closed. The host's `VideoInput` wraps this.
    pub async fn next_frame(&self) -> Option<VideoFrameRef> {
        self.reader.recv().await
    }

    /// The next frame as the host's own `rtc.VideoFrame` (the interned object, never a copy), or
    /// `None` once the input is closed (the host's `__anext__` turns that into
    /// `StopAsyncIteration`).
    #[host(method(name = "next_frame"))]
    pub async fn next_host_frame(&self) -> Option<lk_agents_abi::MediaRef> {
        self.next_frame().await.map(|frame| frame.media)
    }

    /// Stop feeding frames during a hand-off; the track stays subscribed.
    pub fn on_attached(&self) {
        self.reader.set_attached(true);
    }

    pub fn on_detached(&self) {
        self.reader.set_attached(false);
    }

    #[host(method)]
    pub async fn aclose(&self) {
        self.reader.link(None);
        self.reader.close();
    }
}

/// The room's camera as the driver's video source (`ActivityCommand::SetVideoInput`, the pump).
#[async_trait::async_trait]
impl lk_agents_abi::voice::io::VideoInput for ParticipantVideoInput {
    fn label(&self) -> &str {
        "RoomIO"
    }

    async fn next_frame(&self) -> Option<VideoFrameRef> {
        ParticipantVideoInput::next_frame(self).await
    }

    fn on_attached(&self) {
        ParticipantVideoInput::on_attached(self);
    }

    fn on_detached(&self) {
        ParticipantVideoInput::on_detached(self);
    }
}

/// `VoiceActivityVideoSampler`: sample video faster while the user is talking. The last-pass
/// instant moves only on a pass, so the rate doesn't depend on the camera's frame rate.
#[lk_agents_abi::class]
pub struct VideoSampler {
    state: Mutex<SamplerState>,
}

struct SamplerState {
    speaking_fps: f64,
    silent_fps: f64,
    /// On the caller's clock. `None` until the first frame passes.
    last_sampled: Option<Ts>,
}

#[lk_agents_abi::host(object(pid = false))]
impl VideoSampler {
    /// The reference's defaults (1.0 and 0.3) live at the call site.
    #[host(new(kw_only = "speaking_fps"))]
    pub fn new(speaking_fps: f64, silent_fps: f64) -> Self {
        VideoSampler {
            state: Mutex::new(SamplerState { speaking_fps, silent_fps, last_sampled: None }),
        }
    }

    #[host(getter)]
    pub fn speaking_fps(&self) -> f64 {
        self.state.lock().unwrap().speaking_fps
    }

    #[host(getter)]
    pub fn silent_fps(&self) -> f64 {
        self.state.lock().unwrap().silent_fps
    }

    /// Settable, like the reference's public attributes.
    #[host(setter)]
    pub fn set_speaking_fps(&self, fps: f64) {
        self.state.lock().unwrap().speaking_fps = fps;
    }

    #[host(setter)]
    pub fn set_silent_fps(&self, fps: f64) {
        self.state.lock().unwrap().silent_fps = fps;
    }

    /// Should this frame reach the model? `now` is the session clock. Unlike the reference, a
    /// non-positive rate is off (not every frame), and a clock stepping backwards re-anchors.
    pub fn should_sample_at(&self, speaking: bool, now: Ts) -> bool {
        let mut s = self.state.lock().unwrap();
        let target_fps = if speaking { s.speaking_fps } else { s.silent_fps };
        if !(target_fps > 0.0) {
            return false;
        }
        // A rate too small to express as a `Duration` is an interval longer than any session.
        let min_interval = Duration::try_from_secs_f64(1.0 / target_fps).unwrap_or(Duration::MAX);
        match s.last_sampled {
            // The first frame always passes.
            None => {
                s.last_sampled = Some(now);
                true
            }
            Some(last) => {
                let stepped_back = now < last;
                let due = now.saturating_duration_since(last) >= min_interval;
                if stepped_back || due {
                    s.last_sampled = Some(now);
                }
                stepped_back || due
            }
        }
    }

    /// [`should_sample_at`](Self::should_sample_at) for a seconds clock like `time.time()`.
    #[host(method)]
    pub fn should_sample(&self, speaking: bool, now_secs: f64) -> bool {
        self.should_sample_at(speaking, secs_to_ts(now_secs))
    }
}

/// Seconds as a [`Ts`] on the same clock; saturating, so a bad host value can't panic.
fn secs_to_ts(secs: f64) -> Ts {
    Ts::from_nanos((secs * 1e9) as u64)
}

fn read_first(
    participant: &Arc<dyn RemoteParticipant>,
    mut try_read: impl FnMut(&Arc<dyn TrackPublication>) -> bool,
) {
    for publication in participant.track_publications() {
        if publication.is_subscribed() && try_read(&publication) {
            return;
        }
    }
}

struct ReaderState {
    linked: Option<ParticipantIdentity>,
    reading: Option<TrackSid>,
    generation: u64,
    /// Wakes the reading pump so it returns and drops its queue; the generation counter alone
    /// can't reach a pump parked in `recv`.
    stop: Option<Arc<Notify>>,
}

impl ReaderState {
    /// Supersede the reading pump: bump the generation, forget the track, wake it. `notify_one`
    /// leaves a permit, so a pump between frames still sees it.
    fn stop_pump(&mut self) {
        self.generation += 1;
        self.reading = None;
        if let Some(stop) = self.stop.take() {
            stop.notify_one();
        }
    }
}

/// A leg's `try_read(participant, publication)`: whether this track became the one read.
trait TryRead: Fn(&Arc<dyn RemoteParticipant>, &Arc<dyn TrackPublication>) -> bool {}
impl<F: Fn(&Arc<dyn RemoteParticipant>, &Arc<dyn TrackPublication>) -> bool> TryRead for F {}

pub(crate) struct Lease {
    generation: u64,
    stop: Arc<Notify>,
}

impl Lease {
    /// Resolves when this pump has been superseded. Leaves no permit behind for a later pump.
    pub(crate) async fn stopped(&self) {
        self.stop.notified().await;
    }
}

/// Which track a leg reads and the one pump reading it: linking, claiming, releasing, the
/// generation check and the bounded queue.
pub(crate) struct TrackReader<T> {
    state: Mutex<ReaderState>,
    /// Drop-oldest. `Option` so [`close`](Self::close) can drop the sender, ending the consumer's
    /// `recv`.
    tx: Mutex<Option<LossyTx<T>>>,
    rx: tokio::sync::Mutex<BoundedLossy<T>>,
    attached: AtomicBool,
    /// The running pump's handle; a superseded pump returns on its own stop.
    pump: Mutex<Option<AbortHandle>>,
}

impl<T> TrackReader<T> {
    pub(crate) fn new((tx, rx): (LossyTx<T>, BoundedLossy<T>)) -> Self {
        TrackReader {
            state: Mutex::new(ReaderState {
                linked: None,
                reading: None,
                generation: 0,
                stop: None,
            }),
            tx: Mutex::new(Some(tx)),
            rx: tokio::sync::Mutex::new(rx),
            attached: AtomicBool::new(true),
            pump: Mutex::new(None),
        }
    }

    /// [`link`](Self::link), then pick up what the participant already published (possibly before
    /// the core reported the link).
    fn link_and_read(
        &self,
        room: &dyn RtcRoom,
        identity: Option<ParticipantIdentity>,
        try_read: impl TryRead,
    ) {
        if !self.link(identity.clone()) {
            return;
        }
        if let Some(p) = identity.and_then(|i| room.participant(&i)) {
            read_first(&p, |t| try_read(&p, t));
        }
    }

    /// Route one room event to the leg's `try_read` (`_on_track_available`).
    fn route(&self, event: &RoomEvent, try_read: impl TryRead) {
        match event {
            RoomEvent::TrackSubscribed { participant, publication } => {
                try_read(participant, publication);
            }
            // `_on_track_unavailable`: fall back to the participant's next track.
            RoomEvent::TrackUnpublished { participant, publication } => {
                if self.release(participant.identity(), publication.sid()) {
                    read_first(participant, |t| try_read(participant, t));
                }
            }
            RoomEvent::TrackPublished { .. }
            | RoomEvent::LocalTrackPublished { .. }
            | RoomEvent::ParticipantConnected { .. }
            | RoomEvent::ParticipantDisconnected { .. }
            | RoomEvent::ConnectionStateChanged { .. }
            | RoomEvent::Close { .. } => {}
        }
    }

    pub(crate) fn reading(&self) -> Option<TrackSid> {
        self.state.lock().unwrap().reading.clone()
    }

    /// Link to `identity` (or nobody), stopping the pump; `false` if already linked to it.
    pub(crate) fn link(&self, identity: Option<ParticipantIdentity>) -> bool {
        let mut s = self.state.lock().unwrap();
        if s.linked == identity {
            return false;
        }
        s.linked = identity;
        s.stop_pump();
        true
    }

    /// Claim `sid` if [`decide_track`] says so, superseding the current pump. `None`: not ours.
    pub(crate) fn claim(
        &self,
        participant: &ParticipantIdentity,
        source: TrackSource,
        sid: &TrackSid,
        sources: &[TrackSource],
    ) -> Option<Lease> {
        let mut s = self.state.lock().unwrap();
        if !decide_track(s.linked.as_ref(), s.reading.as_ref(), participant, source, sid, sources) {
            return None;
        }
        // Wake the old pump even though this call replaces it.
        s.stop_pump();
        let stop = Arc::new(Notify::new());
        s.stop = Some(Arc::clone(&stop));
        s.reading = Some(sid.clone());
        Some(Lease { generation: s.generation, stop })
    }

    /// `sid` went away; `true` if it was the track read, so the leg looks for another.
    pub(crate) fn release(&self, participant: &ParticipantIdentity, sid: &TrackSid) -> bool {
        let mut s = self.state.lock().unwrap();
        if s.reading.as_ref() != Some(sid) || s.linked.as_ref() != Some(participant) {
            return false;
        }
        s.stop_pump();
        true
    }

    /// One item from the pump holding `lease`; `false` when superseded. The generation check is the
    /// stop the signal can't cover: the pre-connect leg never polls `stopped()`, and the live
    /// `select!` isn't atomic with `stop_pump`. Both are decided under the same mutex.
    pub(crate) fn deliver(&self, lease: &Lease, item: T) -> bool {
        if self.state.lock().unwrap().generation != lease.generation {
            return false;
        }
        if !self.attached.load(Ordering::Relaxed) {
            // Detached: drop rather than close the track, so a re-attach resumes mid-stream.
            return true;
        }
        match self.tx.lock().unwrap().as_ref() {
            Some(tx) => tx.push(item) != Pushed::Closed,
            None => false,
        }
    }

    pub(crate) async fn recv(&self) -> Option<T> {
        self.rx.lock().await.recv().await
    }

    pub(crate) fn set_attached(&self, attached: bool) {
        self.attached.store(attached, Ordering::Relaxed);
    }

    pub(crate) fn keep(&self, handle: AbortHandle) {
        *self.pump.lock().unwrap() = Some(handle);
    }

    /// Stop the pump and hang up on the consumer; the abort bounds a pump whose stop doesn't land.
    pub(crate) fn close(&self) {
        self.state.lock().unwrap().stop_pump();
        if let Some(h) = self.pump.lock().unwrap().take() {
            h.abort();
        }
        if let Some(tx) = self.tx.lock().unwrap().take() {
            tx.close();
        }
    }
}

/// `_ParticipantAudioInputStream`.
pub struct ParticipantAudioInput {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    config: AudioInputConfig,
    pre_connect: Option<Arc<PreConnectAudioHandler>>,
    /// The one audio-in leg; drop-oldest (the output leg must not drop).
    reader: TrackReader<AudioFrame>,
}

impl ParticipantAudioInput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        config: AudioInputConfig,
        pre_connect: Option<Arc<PreConnectAudioHandler>>,
    ) -> Arc<Self> {
        Arc::new(ParticipantAudioInput {
            room,
            spawner,
            config,
            pre_connect,
            reader: TrackReader::new(BoundedLossy::audio()),
        })
    }

    pub fn set_participant(self: &Arc<Self>, identity: Option<ParticipantIdentity>) {
        self.reader.link_and_read(&*self.room, identity, |p, t| self.try_read(p, t));
    }

    /// Route one room event. `track_subscribed` and `track_unpublished` are the two that matter.
    pub fn handle_event(self: &Arc<Self>, event: &RoomEvent) {
        self.reader.route(event, |p, t| self.try_read(p, t));
    }

    /// `_on_track_available`. Returns whether this track became the one being read.
    fn try_read(
        self: &Arc<Self>,
        participant: &Arc<dyn RemoteParticipant>,
        publication: &Arc<dyn TrackPublication>,
    ) -> bool {
        let Some(lease) = self.reader.claim(
            participant.identity(),
            publication.source(),
            publication.sid(),
            AUDIO_INPUT_SOURCES,
        ) else {
            return false;
        };

        let stream = publication.audio_stream(AudioStreamOptions {
            sample_rate: Some(self.config.sample_rate),
            num_channels: Some(self.config.num_channels),
            frame_size_ms: Some(self.config.frame_size_ms),
            // Holding the ref keeps the Krisp side-table entry alive for every stream that may open
            // with it.
            noise_cancellation: self.config.noise_cancellation.clone(),
            // Forwarded to the host's stream; dropping it would silently lose AGC.
            auto_gain_control: self.config.auto_gain_control,
        });
        let wants_pre_connect =
            publication.audio_features().contains(&AudioTrackFeature::PreconnectBuffer);
        let sid = publication.sid().clone();

        let this = self.clone();
        let publication = Arc::clone(publication);
        let handle = self.spawner.spawn_unboxed(async move {
            this.pump(lease, sid, stream, wants_pre_connect.then_some(publication)).await;
        });
        self.reader.keep(handle);
        true
    }

    async fn pump(
        self: Arc<Self>,
        lease: Lease,
        sid: TrackSid,
        mut stream: BoundedLossy<AudioFrame>,
        pre_connect: Option<Arc<dyn TrackPublication>>,
    ) {
        if let Some(publication) = pre_connect
            && let Some(handler) = &self.pre_connect
        {
            // A timeout or malformed buffer isn't fatal: the live stream follows.
            if let Ok(frames) = handler.wait_for_data(&sid).await {
                // Through the stream's noise cancellation first, as the live frames are.
                let frames = match publication.process_audio(frames.clone()).await {
                    Ok(processed) => processed,
                    Err(_) => frames,
                };
                for frame in self.resample_to_input_rate(frames) {
                    if !self.reader.deliver(&lease, frame) {
                        return;
                    }
                }
            }
        }

        loop {
            tokio::select! {
                // Stopping wins over a simultaneous frame.
                biased;
                // `return`, not `break`: a superseded track must not get the end-of-track silence flush.
                () = lease.stopped() => return,
                frame = stream.recv() => match frame {
                    // `false`: superseded.
                    Some(frame) => {
                        if !self.reader.deliver(&lease, frame) {
                            return;
                        }
                    }
                    // The track itself ended: fall through to the flush.
                    None => break,
                },
            }
        }

        // Track ended: half a second of silence flushes a pending STT final.
        let samples = self.config.sample_rate / 2;
        let silence = silence_frame(samples, self.config.sample_rate, self.config.num_channels);
        self.reader.deliver(&lease, silence);
    }

    /// Convert the pre-connect buffer to the live stream's rate once.
    fn resample_to_input_rate(&self, frames: Vec<AudioFrame>) -> Vec<AudioFrame> {
        let target = self.config.sample_rate;
        let Some(first) = frames.first() else { return frames };
        if first.sample_rate == target {
            return frames;
        }
        // The first frame's format is the buffer's; every frame is read as it.
        let format = AudioFormat::new(first.sample_rate, first.num_channels);
        let Ok(mut resampler) = FrameResampler::fixed(format, target, Quality::default()) else {
            // The resampler refused the rate pair. The head is lost; the live stream is not.
            return Vec::new();
        };

        let mut out = Vec::with_capacity(frames.len());
        for frame in &frames {
            match resampler.push(frame) {
                Ok(ready) => out.extend(ready),
                Err(_) => return Vec::new(),
            }
        }
        if let Ok(Some(tail)) = resampler.flush() {
            out.push(tail);
        }
        out
    }

    pub async fn aclose(&self) {
        self.reader.close();
    }
}

#[async_trait]
impl AudioInput for ParticipantAudioInput {
    fn label(&self) -> &str {
        "RoomIO"
    }

    async fn next_frame(&self) -> Option<AudioFrame> {
        self.reader.recv().await
    }

    fn on_attached(&self) {
        self.reader.set_attached(true);
    }

    fn on_detached(&self) {
        self.reader.set_attached(false);
    }
}
