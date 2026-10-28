//! `voice/room_io/_output.py`: `_ParticipantAudioOutput`, the agent's audio on its published track,
//! and its host object; the transcription output is [`transcription`].

use crate::room_io::{AudioOutputConfig, publication::Publication};
use lk_agents_abi::voice::io::Result;
use lk_agents_abi::voice::io::{AudioOutput, PlaybackFinished, PlaybackTracker};
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AbortHandle, Closed, HOST_TEARDOWN_TIMEOUT, timeout};
use lk_agents_abi::{AudioByteStream, AudioFrame, Clock, Spawner, SpawnerExt, Ts};
use lk_agents_rtc::{AudioSource, PlaybackInfo, RtcRoom, frame_duration};
use std::collections::VecDeque;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
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
    /// Heard audio from source windows already consumed by wait_for_playout.
    settled_played: Duration,
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

/// How often a playing segment reports what passed the speaker. A synchroniser downstream releases
/// its words on these reports, so the room's transcript follows the audio as it plays (the
/// reference's synchroniser paces each word on its own timer) rather than all at the segment's end.
const PROGRESS_TICK: Duration = Duration::from_millis(20);

/// Everything the source has taken that has played or been discarded.
fn playhead(info: PlaybackInfo) -> Duration {
    info.played + info.discarded
}

/// The leg's whole state, under its one lock. Never held across an await, and the tracker is only
/// told after it is released.
struct Playback {
    phase: Phase,
    paused: bool,
    /// Frames accepted and not yet handed to the source. Capture waits at the configured bound.
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
    publication: Publication,
    cleanup_finished: bool,
    cleanup_outcome: Option<Closed>,
    cleanup_deadline: Option<Ts>,
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
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    state: Mutex<Playback>,
    /// One wake for every waiter; each re-checks its own condition.
    wake: Notify,
    sample_rate: u32,
    num_channels: u16,
    max_buffer_bytes: usize,
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

    fn fail(&self, error: lk_agents_abi::voice::io::IoError, source_played: Duration) {
        let (captured, played, run) = {
            let mut state = self.lock();
            if state.closed || state.publication.outcome().is_some_and(|outcome| outcome.is_err()) {
                return;
            }
            state.publication.fail(error.clone());
            let captured = state.phase != Phase::Idle;
            state.phase = Phase::Idle;
            state.paused = false;
            state.discard_backlog();
            state.handed.clear();
            state.bstream.clear();
            state.pending = 0;
            let played = state.ledger.settled_played + source_played;
            let unreported = played.saturating_sub(state.ledger.played);
            let at = state.ledger.run_offset + unreported;
            let run = state.ledger.report_run(at, self.clock.now(), None);
            state.ledger = Ledger::default();
            (captured, played, run)
        };
        log::error!(target: "livekit.agents", "room audio source failed: {error}");
        self.report(run);
        if captured {
            self.tracker.note_flush();
            self.tracker.playback_finished(PlaybackFinished {
                playback_position: played,
                interrupted: true,
                synchronized_transcript: None,
            });
        }
        self.wake.notify_waiters();
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
        if let Err(error) = src.capture(&frame).await {
            if !self.lock().closed {
                let played = src.playback_info().played;
                src.clear_queue();
                self.fail(error.into(), played);
            }
            return;
        }

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

    /// While a segment plays, its progress every [`PROGRESS_TICK`], as the run so far. The source
    /// is the playhead's one origin; a pause, a cut and the finish report their own runs.
    async fn report_progress(self: Arc<Self>) {
        let playing = |s: &Playback| s.ledger.started && !s.paused && s.phase != Phase::Idle;
        loop {
            self.wait_until(|s| s.closed || playing(s)).await;
            let at = self.clock.now().saturating_add(PROGRESS_TICK);
            self.clock.wait_until(at).await;
            let src = {
                let st = self.lock();
                if st.closed {
                    return;
                }
                if !playing(&st) {
                    continue;
                }
                st.source.clone()
            };
            let Some(src) = src else { continue };
            let played = playhead(src.playback_info());
            let run = {
                let mut st = self.lock();
                if !playing(&st) {
                    continue;
                }
                let at = st.ledger.window_base + played;
                // Not a dry spell: the next capture still owes the run it may close.
                let dry = st.ledger.dry_at;
                let run = st.ledger.report_run(at, self.clock.now(), None);
                st.ledger.dry_at = dry;
                run
            };
            self.report(run);
        }
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
                {
                    let mut state = self.lock();
                    state.ledger.window_base += playhead(more);
                    state.ledger.settled_played += more.played;
                }
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
                self.lock().phase = Phase::Idle;
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
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// The pump, the watcher and the progress reporter: for the life of the output, not one per
    /// segment.
    tasks: Mutex<Vec<AbortHandle>>,
}

impl ParticipantAudioOutput {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
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
            publication: Publication::default(),
            cleanup_finished: false,
            cleanup_outcome: None,
            cleanup_deadline: None,
        };
        Arc::new(ParticipantAudioOutput {
            room,
            spawner,
            shared: Arc::new(Shared {
                clock,
                tracker: Arc::new(PlaybackTracker::new()),
                state: Mutex::new(state),
                wake: Notify::new(),
                sample_rate,
                num_channels,
                // Leave room for a partial 50 ms frame and the next 50 ms write.
                max_buffer_bytes: (sample_rate as usize * usize::from(num_channels) * 2)
                    .saturating_mul(config.publish.queue_size_ms.max(100) as usize)
                    / 1000,
            }),
            tasks: Mutex::new(Vec::new()),
            config,
        })
    }

    fn begin_start(&self) {
        let mut tasks = self.tasks.lock().unwrap();
        if self.shared.lock().publication.begin() {
            tasks.push(self.spawner.spawn_unboxed(self.shared.clone().forward_audio()));
            tasks.push(self.spawner.spawn_unboxed(self.shared.clone().wait_for_playout()));
            tasks.push(self.spawner.spawn_unboxed(self.shared.clone().report_progress()));
            let room = self.room.clone();
            let shared = self.shared.clone();
            let options = self.config.publish.clone();
            let self_spawner = self.spawner.clone();
            // Creation belongs to the output, even when the caller abandons start(). Close drains
            // this task before releasing its handles, so a late SID can always be unpublished.
            tasks.push(self.spawner.spawn_unboxed(async move {
                let (source, publication) = match room.local().publish_audio_track(options).await {
                    Ok(published) => published,
                    Err(error) => {
                        shared.lock().publication.fail(error.into());
                        shared.wake.notify_waiters();
                        return;
                    }
                };
                let adopted = {
                    let mut state = shared.lock();
                    if state.publication.adopt(publication.sid().clone()) {
                        state.source = Some(source.clone());
                        true
                    } else {
                        false
                    }
                };
                if !adopted {
                    let outcome = retire_publication(
                        room.clone(),
                        source,
                        publication.sid().clone(),
                        self_spawner,
                    )
                    .await;
                    let mut state = shared.lock();
                    state.cleanup_outcome =
                        Some(state.cleanup_outcome.unwrap_or(Closed::Clean).and(outcome));
                    drop(state);
                    shared.lock().publication.creation_retired();
                    shared.wake.notify_waiters();
                    return;
                }
                shared.wake.notify_waiters();
                match publication.wait_for_subscription().await {
                    Ok(()) => shared.lock().publication.ready(),
                    Err(error) => shared.lock().publication.fail(error.into()),
                }
                shared.wake.notify_waiters();
            }));
        }
    }

    pub async fn start(&self) -> Result<()> {
        self.begin_start();
        self.shared.wait_until(|s| s.publication.outcome().is_some()).await;
        self.shared.lock().publication.outcome().expect("readiness settled")
    }

    async fn capture_chunk(&self, frame: &AudioFrame) -> Result<()> {
        self.shared.wait_until(|s| s.publication.outcome().is_some()).await;
        loop {
            let parked = self.shared.wake.notified();
            tokio::pin!(parked);
            parked.as_mut().enable();
            let queued = {
                let mut st = self.shared.lock();
                if let Some(Err(error)) = st.publication.outcome() {
                    drop(st);
                    // capture_frame may already have opened a segment for this refused frame.
                    self.shared.tracker.note_flush();
                    self.shared.tracker.playback_finished(PlaybackFinished {
                        playback_position: Duration::ZERO,
                        interrupted: true,
                        synchronized_transcript: None,
                    });
                    return Err(error);
                }
                match st.phase {
                    Phase::Idle | Phase::Capturing => {
                        let buffered = st.pending
                            + st.backlog.iter().map(|frame| frame.data.len()).sum::<usize>();
                        if frame.data.len() <= self.shared.max_buffer_bytes.saturating_sub(buffered)
                        {
                            st.phase = Phase::Capturing;
                            Some(self.shared.enqueue(&mut st, frame))
                        } else {
                            None
                        }
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

    pub async fn aclose(&self) -> Closed {
        let (first, captured, deadline) = {
            let mut state = self.shared.lock();
            let first = !state.closed;
            let captured = state.phase != Phase::Idle;
            state.closed = true;
            state.publication.close();
            state.phase = Phase::Idle;
            state.paused = false;
            state.discard_backlog();
            state.handed.clear();
            state.bstream.clear();
            state.pending = 0;
            let deadline = *state.cleanup_deadline.get_or_insert_with(|| {
                self.shared.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)
            });
            (first, captured, deadline)
        };
        self.shared.wake.notify_waiters();
        if first {
            // Stop frame/progress producers before the terminal playback event. Creation keeps its
            // separate obligation to deliver or retire the track, even if start() was cancelled.
            {
                let mut tasks = self.tasks.lock().unwrap();
                let workers = tasks.len().min(3);
                for handle in tasks.drain(..workers) {
                    handle.abort();
                }
            }
            let shared = self.shared.clone();
            let room = self.room.clone();
            let spawner = self.spawner.clone();
            self.spawner.spawn_unboxed(async move {
                shared.wait_until(|s| !s.publication.creating()).await;
                let (source, sid) = {
                    let mut state = shared.lock();
                    (state.source.take(), state.publication.retire())
                };
                let mut outcome = Closed::Clean;
                if let Some(source) = source {
                    if captured {
                        shared.report(shared.clear_source(source.as_ref(), false).0);
                    }
                    if captured {
                        let played = shared.lock().ledger.played;
                        shared.tracker.note_flush();
                        shared.tracker.playback_finished(PlaybackFinished {
                            playback_position: played,
                            interrupted: true,
                            synchronized_transcript: None,
                        });
                    }
                    if let Some(sid) = sid {
                        outcome = retire_publication(room, source, sid, spawner).await;
                    } else {
                        outcome = cleanup_outcome(source.close().await, "audio source");
                    }
                }
                {
                    let mut state = shared.lock();
                    state.cleanup_finished = true;
                    state.cleanup_outcome =
                        Some(state.cleanup_outcome.unwrap_or(Closed::Clean).and(outcome));
                }
                shared.wake.notify_waiters();
            });
        }
        let remaining = deadline.saturating_duration_since(self.shared.clock.now());
        let settled = timeout(
            self.shared.clock.as_ref(),
            remaining,
            self.shared.wait_until(|s| s.cleanup_finished),
        )
        .await
        .is_ok();
        let outcome = {
            let mut state = self.shared.lock();
            if !settled {
                state.cleanup_outcome = Some(Closed::TimedOut);
            }
            state.cleanup_outcome.unwrap_or(Closed::TimedOut)
        };
        if settled {
            for handle in self.tasks.lock().unwrap().drain(..) {
                handle.abort();
            }
        }
        outcome
    }
}

async fn retire_publication(
    room: Arc<dyn RtcRoom>,
    source: Arc<dyn AudioSource>,
    sid: lk_agents_rtc::TrackSid,
    spawner: Arc<dyn Spawner>,
) -> Closed {
    // Source closure and SID retirement are separate obligations: either SDK call may hang.
    // This task retains both receipts after a close waiter reports its deadline outcome.
    let (closed, closing) = tokio::sync::oneshot::channel();
    spawner.spawn_unboxed(async move {
        let outcome = cleanup_outcome(source.close().await, "audio source");
        let _ = closed.send(outcome);
    });
    let unpublished =
        cleanup_outcome(room.local().unpublish_track(&sid).await, "audio publication");
    unpublished.and(closing.await.unwrap_or(Closed::Failed))
}

fn cleanup_outcome(result: lk_agents_rtc::Result<()>, resource: &str) -> Closed {
    match result {
        Ok(()) => Closed::Clean,
        Err(error) => {
            log::error!(target: "livekit.agents", "room {resource} cleanup failed: {error}");
            Closed::Failed
        }
    }
}

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

    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if frame.sample_rate != self.shared.sample_rate
                || frame.num_channels != self.shared.num_channels
                || frame.num_channels == 0
                || self.shared.max_buffer_bytes == 0
                || frame.data.len()
                    != frame.samples_per_channel as usize * usize::from(frame.num_channels) * 2
            {
                return Err(lk_agents_abi::voice::io::IoError::Rtc(
                    "room audio frame must match the output format".into(),
                ));
            }
            let sample_bytes = usize::from(frame.num_channels) * 2;
            let chunk_bytes = (frame.sample_rate as usize / 20).max(1) * sample_bytes;
            if frame.data.len() <= chunk_bytes {
                return self.capture_chunk(frame).await;
            }
            for bytes in frame.data.chunks(chunk_bytes) {
                let chunk = AudioFrame::new(
                    bytes.to_vec(),
                    frame.sample_rate,
                    frame.num_channels,
                    (bytes.len() / sample_bytes) as u32,
                );
                self.capture_chunk(&chunk).await?;
            }
            Ok(())
        })
    }

    fn flush_segment(&self) {
        let (tail, watch) = {
            let mut st = self.shared.lock();
            // The tail first, so the watcher sees it in the backlog when the segment closes.
            st.pending = 0;
            let frames = st.bstream.flush().into_iter().map(|f| self.shared.framed(f)).collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::TokioSpawner;
    use lk_agents_rtc::{FakeRtc, RtcError};

    #[tokio::test]
    async fn publication_arriving_after_close_is_retired() {
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room("late-publish");
        let local = room.local_participant();
        local.flow_publications(false);
        let output = ParticipantAudioOutput::new(
            room,
            clock,
            Arc::new(TokioSpawner::current()),
            AudioOutputConfig::default(),
        );
        let mut start = Box::pin(output.start());
        crate::room_io::tests::pending(start.as_mut()).await;
        let mut close = Box::pin(output.aclose());
        crate::room_io::tests::pending(close.as_mut()).await;
        local.flow_publications(true);
        close.await;
        let result = start.await;
        assert!(local.publications().is_empty(), "the late SID must be unpublished");
        assert!(result.is_err(), "closed output must refuse adoption");
        assert!(output.shared.lock().source.is_none());
    }

    #[tokio::test]
    async fn publication_failure_releases_capture_waiters() {
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room("failed-publish");
        room.local_participant().fail_with(RtcError::host("publish failed"));
        let output = ParticipantAudioOutput::new(
            room,
            clock,
            Arc::new(TokioSpawner::current()),
            AudioOutputConfig::default(),
        );
        assert!(output.start().await.is_err());
        let result = lk_agents_abi::timeout(
            &lk_agents_abi::RealClock::new(),
            Duration::from_millis(50),
            output.capture(&lk_agents_rtc::silence_frame(480, 24000, 1)),
        )
        .await;
        assert!(matches!(result, Ok(Err(_))), "capture must receive the publication failure");
        output.aclose().await;
    }
    #[tokio::test]
    async fn paused_room_output_backpressures_capture() {
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room("backpressure");
        let output = ParticipantAudioOutput::new(
            room,
            clock,
            Arc::new(TokioSpawner::current()),
            AudioOutputConfig::default(),
        );
        output.start().await.unwrap();
        output.pause();
        let result = lk_agents_abi::timeout(
            &lk_agents_abi::RealClock::new(),
            Duration::from_millis(50),
            async {
                output.capture(&lk_agents_rtc::silence_frame(24000, 24000, 1)).await.unwrap();
            },
        )
        .await;
        assert!(
            result.is_err(),
            "a paused sink must bound accepted audio instead of buffering a second"
        );
        output.aclose().await;
    }

    #[tokio::test]
    async fn concurrent_starts_share_one_publication() {
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room("joined-start");
        let local = room.local_participant();
        let output = ParticipantAudioOutput::new(
            room,
            clock,
            Arc::new(TokioSpawner::current()),
            AudioOutputConfig::default(),
        );
        local.flow_publications(false);
        let mut first = Box::pin(output.start());
        let mut second = Box::pin(output.start());
        crate::room_io::tests::pending(first.as_mut()).await;
        crate::room_io::tests::pending(second.as_mut()).await;
        local.flow_publications(true);
        first.await.unwrap();
        second.await.unwrap();
        assert_eq!(local.publications().len(), 1, "start joins its existing creation");
        output.aclose().await;
        assert!(local.publications().is_empty());
    }
    #[tokio::test]
    async fn close_finishes_a_captured_segment_as_interrupted() {
        use lk_agents_abi::voice::io::AudioOutputExt;
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let output = ParticipantAudioOutput::new(
            rtc.room("close-segment"),
            clock,
            Arc::new(TokioSpawner::current()),
            AudioOutputConfig::default(),
        );
        output.start().await.unwrap();
        output.pause();
        output.capture_frame(&lk_agents_rtc::silence_frame(480, 24000, 1)).await.unwrap();
        output.aclose().await;
        let finished = lk_agents_abi::timeout(
            &lk_agents_abi::RealClock::new(),
            Duration::from_millis(50),
            output.tracker().wait_for_playout(),
        )
        .await
        .expect("close must settle the captured segment");
        assert!(finished.interrupted);
        assert_eq!(finished.playback_position, Duration::ZERO);
    }

    #[tokio::test]
    async fn cancelled_start_keeps_ownership_of_the_created_track() {
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room("cancel-start");
        let local = room.local_participant();
        local.flow_publications(false);
        let output = ParticipantAudioOutput::new(
            room,
            clock,
            Arc::new(TokioSpawner::current()),
            AudioOutputConfig::default(),
        );
        let mut start = Box::pin(output.start());
        crate::room_io::tests::pending(start.as_mut()).await;
        tokio::task::yield_now().await;
        assert_eq!(
            local.publications().len(),
            1,
            "the server has created the track before acknowledging it"
        );
        drop(start);
        let mut close = Box::pin(output.aclose());
        crate::room_io::tests::pending(close.as_mut()).await;
        local.flow_publications(true);
        close.await;
        assert!(local.publications().is_empty(), "cancellation must not orphan a created SID");
    }
    #[tokio::test]
    async fn source_capture_failure_releases_playout_and_refuses_further_audio() {
        use lk_agents_abi::voice::io::AudioOutputExt;

        async fn finished(output: &ParticipantAudioOutput) -> PlaybackFinished {
            lk_agents_abi::timeout(
                &lk_agents_abi::RealClock::new(),
                Duration::from_secs(1),
                output.tracker().wait_for_playout(),
            )
            .await
            .expect("source failure must settle the captured segment")
        }

        let mut outcomes = Vec::new();
        for mode in ["fresh", "queued", "heard", "blocked"] {
            let (rtc, clock) = FakeRtc::with_virtual_clock();
            let room = rtc.room(mode);
            let local = room.local_participant();
            let output = ParticipantAudioOutput::new(
                room,
                clock.clone(),
                Arc::new(TokioSpawner::current()),
                AudioOutputConfig::default(),
            );
            output.start().await.unwrap();
            let frame = lk_agents_rtc::silence_frame(480, 24_000, 1);
            let (refused, playback) = if mode == "blocked" {
                output.pause();
                for _ in 0..10 {
                    output.capture_frame(&frame).await.unwrap();
                }
                let mut waiting = Box::pin(output.capture_frame(&frame));
                crate::room_io::tests::pending(waiting.as_mut()).await;
                local.sources()[0].close().await.unwrap();
                output.resume();
                let refused = waiting.await;
                (refused, finished(&output).await)
            } else {
                if mode == "queued" || mode == "heard" {
                    output.capture_frame(&frame).await.unwrap();
                    for _ in 0..8 {
                        tokio::task::yield_now().await;
                    }
                    assert_eq!(local.sources()[0].captured().len(), 1);
                    if mode == "heard" {
                        clock.advance(Duration::from_millis(10));
                    }
                }
                local.sources()[0].close().await.unwrap();
                output.capture_frame(&frame).await.unwrap();
                output.flush();
                for _ in 0..8 {
                    tokio::task::yield_now().await;
                }
                let playback = finished(&output).await;
                (output.capture_frame(&frame).await, playback)
            };
            output.flush_segment();
            output.aclose().await;
            let settled = lk_agents_abi::timeout(
                &lk_agents_abi::RealClock::new(),
                Duration::from_secs(1),
                output.tracker().wait_for_playout(),
            )
            .await
            .is_ok();
            outcomes.push((
                mode,
                playback.interrupted,
                playback.playback_position,
                refused.is_err(),
                settled,
            ));
        }
        assert_eq!(
            outcomes,
            [
                ("fresh", true, Duration::ZERO, true, true),
                ("queued", true, Duration::ZERO, true, true),
                ("heard", true, Duration::from_millis(10), true, true),
                ("blocked", true, Duration::ZERO, true, true),
            ],
            "source failure must preserve heard media, refuse parked and later captures, and settle every waiter"
        );
    }
}
