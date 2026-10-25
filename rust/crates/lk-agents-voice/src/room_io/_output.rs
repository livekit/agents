//! `voice/room_io/_output.py`: `_ParticipantAudioOutput`, the agent's audio on its published track,
//! and its host object; the transcription output is [`transcription`].

use crate::room_io::types::AudioOutputConfig;
use async_trait::async_trait;
use lk_agents_abi::AbortHandle;
use lk_agents_abi::voice::io::Result;
use lk_agents_abi::voice::io::{AudioOutput, IoError, PlaybackFinished, PlaybackTracker};
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AudioByteStream, AudioFrame, Clock, Monotone, Spawner, SpawnerExt, Ts};
use lk_agents_rtc::{AudioSource, PlaybackInfo, RtcRoom, TrackSource, frame_duration};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

pub mod transcription;

/// Where the current segment is. A pause is beside it (`Playback::paused`), so resume plays on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Capturing,
    /// Flushed: closed to new frames, the source still playing what it holds.
    Flushed,
    /// A clear landed: backlog gone, source cut; the segment finishes interrupted.
    Cutting,
}

/// Frames handed to the source that may still be queued there, oldest first. A pause's cut returns
/// the last `discarded` of these to the backlog; pruned as the queue drains.
#[derive(Default)]
struct Handed {
    frames: VecDeque<AudioFrame>,
    total: Duration,
}

impl Handed {
    fn push(&mut self, frame: AudioFrame) {
        self.total += frame_duration(&frame);
        self.frames.push_back(frame);
    }

    /// Drop every frame that has fully played: `queued` is what the source still holds.
    fn prune(&mut self, queued: Duration) {
        while let Some(front) = self.frames.front() {
            let len = frame_duration(front);
            if self.total.saturating_sub(len) < queued {
                break;
            }
            self.total -= len;
            self.frames.pop_front();
        }
    }

    fn clear(&mut self) {
        self.frames.clear();
        self.total = Duration::ZERO;
    }

    /// The last `discarded` of audio handed, boundary frame trimmed to its unheard part.
    fn take_last(&mut self, discarded: Duration) -> Vec<AudioFrame> {
        let mut out = Vec::new();
        let mut taken = Duration::ZERO;
        while taken < discarded {
            let Some(frame) = self.frames.pop_back() else { break };
            let len = frame_duration(&frame);
            let need = discarded - taken;
            if len > need {
                out.push(tail_of(&frame, need));
                taken = discarded;
            } else {
                out.push(frame);
                taken += len;
            }
        }
        self.clear();
        out.reverse();
        out
    }
}

/// The last `keep` of `frame`, on a sample boundary.
fn tail_of(frame: &AudioFrame, keep: Duration) -> AudioFrame {
    let channels = usize::from(frame.num_channels.max(1));
    let keep_samples = ((keep.as_secs_f64() * f64::from(frame.sample_rate)).round() as u32)
        .min(frame.samples_per_channel);
    let drop_bytes =
        ((frame.samples_per_channel - keep_samples) as usize * channels * 2).min(frame.data.len());
    AudioFrame::new(
        frame.data[drop_bytes..].to_vec(),
        frame.sample_rate,
        frame.num_channels,
        keep_samples,
    )
}

/// The progressed-run ledger, in the segment's captured timeline; the playhead comes from
/// [`PlaybackInfo`].
#[derive(Debug, Default)]
struct Ledger {
    /// A frame of this segment reached the source (`playback_started` reported); not while paused.
    started: bool,
    run_offset: Duration,
    /// When the source runs dry if nothing more is captured.
    dry_at: Option<Ts>,
    /// What this segment has played: the closed runs' durations. A cut source reports nothing, so
    /// the finished position is read here.
    played: Duration,
    /// Audio a pause cut and re-fed: counted twice by the source, once here.
    refed: Duration,
    /// The source-side position the current window started at (the source resets per
    /// `wait_for_playout`, and a pause can drain it more than once).
    window_base: Duration,
}

struct Run {
    started_at: Ts,
    offset: Duration,
    duration: Duration,
}

impl Ledger {
    /// `_report_run`: close the open run at `playhead` and open the next at `resumes_at` (past a
    /// cut's discard).
    fn report_run(
        &mut self,
        playhead: Duration,
        ended_at: Ts,
        resumes_at: Option<Duration>,
    ) -> Option<Run> {
        // A run cannot outlast the audio the source actually held, however late we noticed.
        let ended_at = match self.dry_at {
            Some(dry) if dry < ended_at => dry,
            Some(_) | None => ended_at,
        };
        let duration = playhead.saturating_sub(self.run_offset);
        self.played += duration;
        let run = (!duration.is_zero()).then(|| Run {
            started_at: ended_at.saturating_sub(duration),
            offset: self.run_offset.saturating_sub(self.refed),
            duration,
        });
        self.run_offset = resumes_at.unwrap_or(playhead).max(self.run_offset);
        self.dry_at = None;
        run
    }
}

/// Everything the source has taken that has played or been discarded.
fn playhead(info: PlaybackInfo) -> Duration {
    info.played + info.discarded
}

/// The leg's whole state, under its one lock. Never held across an await, and the tracker is only
/// told after it is released.
struct Playback {
    phase: Phase,
    paused: bool,
    /// Frames accepted and not yet handed to the source (`_audio_buf`). Unbounded: the source's
    /// queue is the backpressure, and dropping would cut words.
    backlog: VecDeque<AudioFrame>,
    /// The pump holds a frame inside `capture`, so "drained" is honest.
    in_flight: bool,
    closed: bool,
    /// Frames at the backlog's front a pause put back, not yet taken; the in-flight frame parks
    /// behind them.
    refed: usize,
    handed: Handed,
    ledger: Ledger,
    /// `_audio_bstream`: rebuffers TTS output into 50 ms frames, progressively (the first is
    /// 20 ms). Bypassed when already framed and nothing is buffered (`pending` is zero).
    bstream: AudioByteStream,
    pending: usize,
    source: Option<Arc<dyn AudioSource>>,
}

impl Playback {
    fn drained(&self) -> bool {
        self.backlog.is_empty() && !self.in_flight
    }

    fn discard_backlog(&mut self) {
        self.backlog.clear();
        self.refed = 0;
    }

    fn extend_backlog(&mut self, frames: Vec<AudioFrame>) -> bool {
        if frames.is_empty() || self.closed {
            return false;
        }
        self.backlog.extend(frames);
        true
    }
}

struct Shared {
    clock: Arc<dyn Clock>,
    tracker: Arc<PlaybackTracker>,
    state: Mutex<Playback>,
    /// One wake for every waiter; each re-checks its own condition.
    wake: Notify,
    /// The track is published *and* a remote peer has subscribed. `_subscribed_fut`.
    subscribed: Monotone<bool>,
    sample_rate: u32,
    num_channels: u16,
}

impl Shared {
    /// Never held across an await, nor across a call into the source: a host source takes the GIL,
    /// which a host thread calling into this leg already holds.
    fn lock(&self) -> std::sync::MutexGuard<'_, Playback> {
        self.state.lock().unwrap()
    }

    /// `clear_queue` on the source, closing the open run at what passed the speaker. A pause
    /// (`park`) resumes the timeline past the re-fed discard, which is where this run ended; a cut
    /// resumes where the source does. Returns what the cut discarded.
    fn clear_source(&self, src: &dyn AudioSource, park: bool) -> (Option<Run>, Duration) {
        let before = playhead(src.playback_info());
        src.clear_queue();
        let after = playhead(src.playback_info());
        let now = self.clock.now();
        let mut st = self.lock();
        let base = st.ledger.window_base;
        let (before, after) = (base + before, base + after);
        let discarded = after.saturating_sub(before);
        let resumes_at = if park { before + discarded } else { after };
        let run = st.ledger.report_run(before, now, Some(resumes_at));
        if park {
            st.ledger.refed += discarded;
        }
        (run, discarded)
    }

    fn report(&self, run: Option<Run>) {
        if let Some(run) = run {
            self.tracker.playback_progressed(run.started_at, run.offset, run.duration);
        }
    }

    fn framed(&self, f: lk_agents_abi::Frame) -> AudioFrame {
        AudioFrame::new(f.data, self.sample_rate, self.num_channels, f.samples_per_channel)
    }

    /// Into the backlog, re-chunked to 50 ms unless the emitter already did that.
    fn enqueue(&self, st: &mut Playback, frame: &AudioFrame) -> bool {
        let already_framed = st.pending == 0
            && frame.sample_rate == self.sample_rate
            && frame.num_channels == self.num_channels
            && frame.samples_per_channel == self.sample_rate / 20;
        let frames = if already_framed {
            vec![frame.clone()]
        } else {
            st.pending += frame.data.len();
            let out = st.bstream.push(&frame.data);
            let emitted: usize = out.iter().map(|f| f.data.len()).sum();
            st.pending = st.pending.saturating_sub(emitted);
            out.into_iter().map(|f| self.framed(f)).collect()
        };
        st.extend_backlog(frames)
    }

    /// Park until `ready` holds. Armed before the check so a wake between isn't lost.
    async fn wait_until(&self, ready: impl Fn(&Playback) -> bool) {
        loop {
            let parked = self.wake.notified();
            tokio::pin!(parked);
            parked.as_mut().enable();
            if ready(&self.lock()) {
                return;
            }
            parked.await;
        }
    }

    /// `_forward_audio`: one frame at a time from the backlog into the source, paced by the source.
    async fn forward_audio(self: Arc<Self>) {
        loop {
            let frame = loop {
                let parked = self.wake.notified();
                tokio::pin!(parked);
                parked.as_mut().enable();
                {
                    let mut st = self.lock();
                    if st.closed && st.backlog.is_empty() {
                        return;
                    }
                    if !st.paused
                        && let Some(frame) = st.backlog.pop_front()
                    {
                        st.in_flight = true;
                        st.refed = st.refed.saturating_sub(1);
                        break frame;
                    }
                }
                parked.await;
            };

            let src = self.lock().source.clone();
            if let Some(src) = src {
                self.submit(&src, frame).await;
            }
            self.lock().in_flight = false;
            self.wake.notify_waiters();
        }
    }

    async fn submit(&self, src: &Arc<dyn AudioSource>, frame: AudioFrame) {
        let now = self.clock.now();
        let (first, dry) = {
            let mut st = self.lock();
            (!std::mem::replace(&mut st.ledger.started, true), st.ledger.dry_at)
        };
        if first {
            // The segment's first frame leaves for the speaker now, not when it was pushed.
            self.tracker.playback_started(now);
        }
        // The source ran out before this frame arrived; the run ended when it did.
        if let Some(dry) = dry.filter(|dry| now >= *dry) {
            let at = playhead(src.playback_info());
            let mut st = self.lock();
            let at = st.ledger.window_base + at;
            let run = st.ledger.report_run(at, dry, None);
            drop(st);
            self.report(run);
        }

        // The await that paces everything.
        let _ = src.capture(&frame).await;

        let queued = src.queued_duration();
        let (paused, cutting) = {
            let mut st = self.lock();
            st.handed.push(frame.clone());
            st.handed.prune(queued);
            (st.paused, st.phase == Phase::Cutting)
        };
        // A pause or cut may have landed while parked on backpressure: cut this frame too (the
        // reference lets it play). After a pause it is owed back, behind the frames already put
        // back.
        if paused {
            let (run, discarded) = self.clear_source(src.as_ref(), true);
            self.report(run);
            let mut st = self.lock();
            st.handed.clear();
            if !discarded.is_zero() && !st.closed {
                let owed = if discarded < frame_duration(&frame) {
                    tail_of(&frame, discarded)
                } else {
                    frame
                };
                let at = st.refed.min(st.backlog.len());
                st.backlog.insert(at, owed);
                st.refed = at + 1;
            }
        } else if cutting {
            src.clear_queue();
            let at = playhead(src.playback_info());
            let mut st = self.lock();
            let at = st.ledger.window_base + at;
            st.ledger.run_offset = st.ledger.run_offset.max(at);
            st.handed.clear();
        }
        let dry_at = self.clock.now().saturating_add(src.queued_duration());
        self.lock().ledger.dry_at = Some(dry_at);
    }

    /// `_wait_for_playout`, per closed segment: wait for the drain, then finish the segment.
    async fn wait_for_playout(self: Arc<Self>) {
        loop {
            self.wait_until(|s| s.closed || matches!(s.phase, Phase::Flushed | Phase::Cutting))
                .await;
            if self.lock().closed {
                return;
            }
            // Summed over waits: a pause can make the source run dry more than once.
            let mut info = PlaybackInfo::default();
            loop {
                self.wait_until(|s| s.closed || s.drained()).await;
                let src = {
                    let st = self.lock();
                    if st.closed {
                        return;
                    }
                    st.source.clone()
                };
                let more = match src {
                    Some(src) => src.wait_for_playout().await,
                    None => PlaybackInfo::default(),
                };
                info.played += more.played;
                info.discarded += more.discarded;
                // The source's counters start over from here; the ledger keeps counting.
                self.lock().ledger.window_base += playhead(more);
                // A pause holds the finish until the sink plays again or the pause becomes a cut,
                // unless the pause put audio back; then measure again until nothing is queued.
                self.wait_until(|s| s.closed || !(s.paused && s.phase == Phase::Flushed)).await;
                let (drained, src) = {
                    let st = self.lock();
                    if st.closed {
                        return;
                    }
                    (st.drained(), st.source.clone())
                };
                if !drained || src.is_some_and(|src| !src.queued_duration().is_zero()) {
                    continue;
                }
                let mut st = self.lock();
                let interrupted = match st.phase {
                    Phase::Idle | Phase::Capturing => break,
                    // Paused: not finished; re-offered at resume or cut.
                    Phase::Flushed if st.paused => continue,
                    // A clean drain; a pause is not an interruption, only a clear is.
                    Phase::Flushed => false,
                    // A clear after everything played is still an interruption.
                    Phase::Cutting => true,
                };
                st.phase = Phase::Idle;
                let run = st.ledger.report_run(playhead(info), self.clock.now(), None);
                let played = std::mem::take(&mut st.ledger).played;
                st.handed.clear();
                drop(st);
                self.report(run);
                self.tracker.playback_finished(PlaybackFinished {
                    playback_position: played.max(info.played),
                    interrupted,
                    synchronized_transcript: None,
                });
                // Idle again: held captures may retry.
                self.wake.notify_waiters();
                break;
            }
        }
    }
}

/// `_ParticipantAudioOutput`: frames rebuffered into a backlog, forwarded into the track's source by
/// one pump, each flushed segment finished by one watcher.
pub struct ParticipantAudioOutput {
    room: Arc<dyn RtcRoom>,
    spawner: Arc<dyn Spawner>,
    config: AudioOutputConfig,
    shared: Arc<Shared>,
    /// The pump and the watcher. Two tasks for the life of the output, not one per segment.
    tasks: Mutex<Vec<AbortHandle>>,
}

impl ParticipantAudioOutput {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        config: AudioOutputConfig,
    ) -> Arc<Self> {
        let sample_rate = config.publish.sample_rate;
        let num_channels = config.publish.num_channels;
        let state = Playback {
            phase: Phase::Idle,
            paused: false,
            backlog: VecDeque::new(),
            in_flight: false,
            closed: false,
            refed: 0,
            handed: Handed::default(),
            ledger: Ledger::default(),
            // 50 ms frames, as `_output.py`: `samples_per_channel=sample_rate // 20`.
            bstream: AudioByteStream::new(sample_rate, num_channels, Some(sample_rate / 20), true),
            pending: 0,
            source: None,
        };
        Arc::new(ParticipantAudioOutput {
            room,
            spawner,
            shared: Arc::new(Shared {
                clock,
                tracker: Arc::new(PlaybackTracker::new()),
                state: Mutex::new(state),
                wake: Notify::new(),
                subscribed: Monotone::new(false),
                sample_rate,
                num_channels,
            }),
            tasks: Mutex::new(Vec::new()),
            config,
        })
    }

    pub async fn start(&self) -> Result<()> {
        let pump = self.spawner.spawn_unboxed(self.shared.clone().forward_audio());
        let watcher = self.spawner.spawn_unboxed(self.shared.clone().wait_for_playout());
        *self.tasks.lock().unwrap() = vec![pump, watcher];

        let (source, publication) =
            self.room.local().publish_audio_track(self.config.publish.clone()).await?;
        self.shared.lock().source = Some(source);

        publication.wait_for_subscription().await?;
        self.shared.subscribed.set(true);
        Ok(())
    }

    pub async fn aclose(&self) {
        let src = {
            let mut st = self.shared.lock();
            st.phase = Phase::Idle;
            st.paused = false;
            st.discard_backlog();
            st.closed = true;
            st.handed.clear();
            st.source.clone()
        };
        self.shared.wake.notify_waiters();
        if let Some(src) = src {
            let _ = src.close().await;
        }
        // The tasks return on `closed` by themselves; the abort is a bound on one that does not.
        for h in self.tasks.lock().unwrap().drain(..) {
            h.abort();
        }
    }
}

#[async_trait]
impl AudioOutput for ParticipantAudioOutput {
    fn label(&self) -> &str {
        "RoomIO"
    }

    fn tracker(&self) -> &Arc<PlaybackTracker> {
        &self.shared.tracker
    }

    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: true }
    }

    fn sample_rate(&self) -> Option<u32> {
        Some(self.shared.sample_rate)
    }

    async fn capture(&self, frame: &AudioFrame) -> Result<()> {
        self.shared.subscribed.wait_for(|s| *s).await;
        loop {
            let parked = self.shared.wake.notified();
            tokio::pin!(parked);
            parked.as_mut().enable();
            let queued = {
                let mut st = self.shared.lock();
                match st.phase {
                    Phase::Idle | Phase::Capturing => {
                        st.phase = Phase::Capturing;
                        Some(self.shared.enqueue(&mut st, frame))
                    }
                    // Letting the frame in would land its playout in the previous segment's window:
                    // held until the watcher finishes that segment.
                    Phase::Flushed | Phase::Cutting => None,
                }
            };
            match queued {
                Some(woke) => {
                    if woke {
                        self.shared.wake.notify_waiters();
                    }
                    return Ok(());
                }
                None => parked.await,
            }
        }
    }

    fn flush_segment(&self) {
        let (tail, watch) = {
            let mut st = self.shared.lock();
            // The tail first, so the watcher sees it in the backlog when the segment closes.
            st.pending = 0;
            let frames =
                st.bstream.flush().frames.into_iter().map(|f| self.shared.framed(f)).collect();
            let tail = st.extend_backlog(frames);
            // Nothing captured is no playout to wait for; a cut's measurement is the one that counts.
            let watch = st.phase == Phase::Capturing;
            if watch {
                st.phase = Phase::Flushed;
            }
            (tail, watch)
        };
        if tail || watch {
            self.shared.wake.notify_waiters();
        }
    }

    fn clear_buffer(&self) {
        let src = {
            let mut st = self.shared.lock();
            // Resets progressive sizing too, so the first frame after a interruption is 20 ms again.
            st.bstream.clear();
            st.pending = 0;
            if !matches!(st.phase, Phase::Capturing | Phase::Flushed) {
                return;
            }
            st.phase = Phase::Cutting;
            st.discard_backlog();
            st.source.clone()
        };
        self.shared.wake.notify_waiters();
        if let Some(src) = src {
            // Before the cut: what passed the speaker. After: where the timeline resumes.
            self.shared.report(self.shared.clear_source(src.as_ref(), false).0);
        }
        self.shared.lock().handed.clear();
    }

    /// Cut now: a pause means the user started speaking. What the cut discarded goes back at the
    /// backlog's front so resume plays on with no hole or repeat (the reference lets the source
    /// queue play out under a pause).
    fn pause(&self) {
        let src = {
            let mut st = self.shared.lock();
            if std::mem::replace(&mut st.paused, true) {
                return;
            }
            st.source.clone()
        };
        let Some(src) = src else { return };
        let (run, discarded) = self.shared.clear_source(src.as_ref(), true);
        self.shared.report(run);
        let mut st = self.shared.lock();
        let frames = st.handed.take_last(discarded);
        if !st.closed {
            st.refed += frames.len();
            for frame in frames.into_iter().rev() {
                st.backlog.push_front(frame);
            }
        }
    }

    fn resume(&self) {
        if std::mem::replace(&mut self.shared.lock().paused, false) {
            self.shared.wake.notify_waiters();
        }
    }
}

// The session's own leg (`of`, one `Arc`, so assigning it back swaps nothing), or a standalone leg
// over a host room built by the constructor.

/// `_core.RoomAudioOutput`: a core audio leg as a host object.
pub struct RoomAudioLeg {
    inner: Arc<dyn AudioOutput>,
    /// The concrete leg when the constructor built it: `start`/`aclose` are its.
    owned: Option<Arc<ParticipantAudioOutput>>,
    anchor: lk_agents_abi::WallAnchor,
    listener: Mutex<Option<Arc<dyn lk_agents_abi::voice::io::PlaybackObserver>>>,
}

impl RoomAudioLeg {
    /// A leg the session (or a synchroniser) already runs, stamped against its clock's `anchor`.
    pub fn of(inner: Arc<dyn AudioOutput>, anchor: lk_agents_abi::WallAnchor) -> Self {
        RoomAudioLeg { inner, owned: None, anchor, listener: Mutex::new(None) }
    }

    /// The leg itself, for a host handing it back as `session.output.audio`.
    pub fn leg(&self) -> Arc<dyn AudioOutput> {
        Arc::clone(&self.inner)
    }
}

#[lk_agents_abi::host(object(name = "RoomAudioOutput", cell = "arc", as_dyn(AudioOutput = leg)))]
impl RoomAudioLeg {
    /// A standalone leg over a host room (the reference's `_ParticipantAudioOutput(room,
    /// sample_rate=, num_channels=, track_publish_options=, track_name=)`); `start()` publishes.
    #[host(new(kw_only = "sample_rate"))]
    pub fn standalone(
        room: Arc<dyn RtcRoom>,
        sample_rate: u32,
        num_channels: u16,
        track_source: Option<TrackSource>,
        track_name: Option<String>,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Self {
        let clock: Arc<dyn Clock> = Arc::new(lk_agents_abi::RealClock::new());
        let anchor = lk_agents_abi::WallAnchor::of(&*clock);
        let mut publish =
            lk_agents_rtc::TrackPublishOptions { sample_rate, num_channels, ..Default::default() };
        if let Some(name) = track_name {
            publish.name = name;
        }
        if let Some(source) = track_source {
            publish.source = source;
        }
        let leg = ParticipantAudioOutput::new(room, clock, spawner, AudioOutputConfig { publish });
        RoomAudioLeg { inner: leg.clone(), owned: Some(leg), anchor, listener: Mutex::new(None) }
    }

    /// Publish the track; resolves once a remote peer has subscribed. The session's leg is started
    /// by its `RoomIo`.
    #[host(method)]
    pub async fn start(&self) -> std::result::Result<(), IoError> {
        match &self.owned {
            Some(leg) => leg.start().await,
            None => Ok(()),
        }
    }

    #[host(method)]
    pub async fn aclose(&self) {
        if let Some(leg) = &self.owned {
            leg.aclose().await;
        }
    }

    #[host(getter)]
    pub fn sample_rate(&self) -> Option<u32> {
        self.inner.sample_rate()
    }

    #[host(getter)]
    pub fn can_pause(&self) -> bool {
        lk_agents_abi::voice::io::AudioOutputExt::can_pause(&*self.inner)
    }

    /// One frame, downstream. Awaited: the leg blocks on the track's own backpressure.
    #[host(method)]
    pub async fn capture_frame(&self, frame: AudioFrame) -> std::result::Result<(), IoError> {
        lk_agents_abi::voice::io::AudioOutputExt::capture_frame(&*self.inner, &frame).await
    }

    #[host(method)]
    pub fn flush(&self) {
        lk_agents_abi::voice::io::AudioOutputExt::flush(&*self.inner);
    }

    #[host(method)]
    pub fn clear_buffer(&self) {
        self.inner.clear_buffer();
    }

    #[host(method)]
    pub fn pause(&self) {
        self.inner.pause();
    }

    #[host(method)]
    pub fn resume(&self) {
        self.inner.resume();
    }

    /// Hand the leg's playback events to `listener` (the wrapper re-emits them as its own). One
    /// subscription: a second replaces the first.
    #[host(method)]
    pub fn subscribe(&self, listener: Arc<dyn lk_agents_abi::voice::io::PlaybackListener>) {
        let observer: Arc<dyn lk_agents_abi::voice::io::PlaybackObserver> =
            Arc::new(lk_agents_abi::voice::io::ListenerObserver { listener, anchor: self.anchor });
        let mut slot = self.listener.lock().unwrap();
        if let Some(old) = slot.take() {
            self.inner.tracker().unobserve(&old);
        }
        self.inner.tracker().observe(observer.clone());
        *slot = Some(observer);
    }
}
