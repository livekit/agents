//! Room audio buffering, pause/refeed and playout accounting over supplied observations.

use lk_agents_abi::voice::io::{IoError, PlaybackFinished, PlaybackSegment, Result};
use lk_agents_abi::voice::rtc::{PlaybackInfo, TrackSid, frame_duration};
use lk_agents_abi::{AudioFrame, Ts};
use lk_agents_media::AudioByteStream;
use std::borrow::Cow;
use std::collections::VecDeque;
use std::time::Duration;

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

/// A captured frame in 50 ms chunks, so one long frame cannot hold the whole backlog; a frame
/// that fits passes through unchanged.
pub fn capture_chunks(frame: &AudioFrame) -> Vec<Cow<'_, AudioFrame>> {
    let sample_bytes = usize::from(frame.num_channels) * 2;
    let chunk_bytes = (frame.sample_rate as usize / 20).max(1) * sample_bytes;
    if frame.data.len() <= chunk_bytes {
        return vec![Cow::Borrowed(frame)];
    }
    frame
        .data
        .chunks(chunk_bytes)
        .map(|bytes| {
            Cow::Owned(AudioFrame::new(
                bytes.to_vec(),
                frame.sample_rate,
                frame.num_channels,
                (bytes.len() / sample_bytes) as u32,
            ))
        })
        .collect()
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

/// Speaker progress in the segment's captured timeline.
#[derive(Debug)]
pub struct Progress {
    pub started_at: Ts,
    pub offset: Duration,
    pub duration: Duration,
}

/// A source observation retains the segment that requested it across asynchronous execution.
#[derive(Debug)]
pub enum PlaybackReport {
    Started { segment: PlaybackSegment, at: Ts },
    Progressed { segment: PlaybackSegment, progress: Progress },
    Finished { segment: PlaybackSegment, event: PlaybackFinished },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureAdmission {
    Accepted { wake: bool },
    Wait,
}

/// The room output data plane. Resource handles and waits live in its performer.
pub struct Playback {
    segment: Option<PlaybackSegment>,
    phase: Phase,
    paused: bool,
    backlog: VecDeque<AudioFrame>,
    in_flight: bool,
    closed: bool,
    refed: usize,
    handed: Handed,
    ledger: Ledger,
    bstream: AudioByteStream,
    pending: usize,
    sample_rate: u32,
    num_channels: u16,
    max_buffer_bytes: usize,
}

impl Playback {
    pub fn new(sample_rate: u32, num_channels: u16, queue_size_ms: u32) -> Self {
        Self {
            segment: None,
            phase: Phase::Idle,
            paused: false,
            backlog: VecDeque::new(),
            in_flight: false,
            closed: false,
            refed: 0,
            handed: Handed::default(),
            ledger: Ledger::default(),
            bstream: AudioByteStream::new(sample_rate, num_channels, Some(sample_rate / 20), true),
            pending: 0,
            sample_rate,
            num_channels,
            max_buffer_bytes: (sample_rate as usize * usize::from(num_channels) * 2)
                .saturating_mul(queue_size_ms.max(100) as usize)
                / 1000,
        }
    }

    pub fn begin_segment(&mut self, segment: PlaybackSegment) {
        if self.phase == Phase::Idle {
            self.segment = Some(segment);
        }
    }

    pub fn segment(&self) -> Option<PlaybackSegment> {
        self.segment
    }
    pub fn closed(&self) -> bool {
        self.closed
    }
    pub fn drained(&self) -> bool {
        self.backlog.is_empty() && !self.in_flight
    }
    pub fn ready_to_finish(&self) -> bool {
        matches!(self.phase, Phase::Flushed | Phase::Cutting)
    }
    pub fn finish_held(&self) -> bool {
        self.paused && self.phase == Phase::Flushed
    }
    pub fn playing(&self) -> bool {
        self.ledger.started && !self.paused && self.phase != Phase::Idle
    }
    pub fn paused(&self) -> bool {
        self.paused
    }
    pub fn cutting(&self) -> bool {
        self.phase == Phase::Cutting
    }
    pub fn dry_at(&self) -> Option<Ts> {
        self.ledger.dry_at
    }
    pub fn window_position(&self, info: PlaybackInfo) -> Duration {
        self.ledger.window_base + playhead(info)
    }

    pub fn validate(&self, frame: &AudioFrame) -> Result<()> {
        if frame.sample_rate != self.sample_rate
            || frame.num_channels != self.num_channels
            || frame.num_channels == 0
            || self.max_buffer_bytes == 0
            || frame.data.len()
                != frame.samples_per_channel as usize * usize::from(frame.num_channels) * 2
        {
            return Err(IoError::Rtc("room audio frame must match the output format".into()));
        }
        Ok(())
    }

    pub fn capture(
        &mut self,
        segment: Option<PlaybackSegment>,
        frame: &AudioFrame,
    ) -> CaptureAdmission {
        if self.closed || !matches!(self.phase, Phase::Idle | Phase::Capturing) {
            return CaptureAdmission::Wait;
        }
        let buffered =
            self.pending + self.backlog.iter().map(|frame| frame.data.len()).sum::<usize>();
        if frame.data.len() > self.max_buffer_bytes.saturating_sub(buffered) {
            return CaptureAdmission::Wait;
        }
        if self.phase == Phase::Idle {
            self.segment = segment;
        }
        self.phase = Phase::Capturing;
        let already_framed = self.pending == 0
            && frame.sample_rate == self.sample_rate
            && frame.num_channels == self.num_channels
            && frame.samples_per_channel == self.sample_rate / 20;
        let frames = if already_framed {
            vec![frame.clone()]
        } else {
            self.pending += frame.data.len();
            let frames = self.bstream.push(&frame.data);
            let emitted: usize = frames.iter().map(|frame| frame.data.len()).sum();
            self.pending = self.pending.saturating_sub(emitted);
            frames
                .into_iter()
                .map(|frame| {
                    AudioFrame::new(
                        frame.data,
                        self.sample_rate,
                        self.num_channels,
                        frame.samples_per_channel,
                    )
                })
                .collect()
        };
        let wake = !frames.is_empty();
        self.backlog.extend(frames);
        CaptureAdmission::Accepted { wake }
    }

    pub fn next_frame(&mut self) -> Option<AudioFrame> {
        if self.paused || self.closed {
            return None;
        }
        let frame = self.backlog.pop_front()?;
        self.in_flight = true;
        self.refed = self.refed.saturating_sub(1);
        Some(frame)
    }

    pub fn captured(&mut self, frame: AudioFrame, queued: Duration) {
        self.handed.push(frame);
        self.handed.prune(queued);
    }

    pub fn capture_settled(&mut self) {
        self.in_flight = false;
    }

    pub fn submitted(&mut self, now: Ts) -> Option<PlaybackReport> {
        let first = !std::mem::replace(&mut self.ledger.started, true);
        if first {
            self.segment.map(|segment| PlaybackReport::Started { segment, at: now })
        } else {
            None
        }
    }

    pub fn progressed(
        &mut self,
        info: PlaybackInfo,
        now: Ts,
        preserve_dry: bool,
    ) -> Option<PlaybackReport> {
        let dry = self.ledger.dry_at;
        let at = self.window_position(info);
        let run = self.ledger.report_run(at, now, None);
        if preserve_dry {
            self.ledger.dry_at = dry;
        }
        self.report(run)
    }

    pub fn cleared(
        &mut self,
        before: PlaybackInfo,
        after: PlaybackInfo,
        now: Ts,
        park: bool,
    ) -> (Option<PlaybackReport>, Duration) {
        let (before, after) = (self.window_position(before), self.window_position(after));
        let discarded = after.saturating_sub(before);
        let resumes_at = if park { before + discarded } else { after };
        let run = self.ledger.report_run(before, now, Some(resumes_at));
        if park {
            self.ledger.refed += discarded;
        }
        (self.report(run), discarded)
    }

    pub fn park_captured(&mut self, frame: AudioFrame, discarded: Duration) {
        self.handed.clear();
        if discarded.is_zero() || self.closed {
            return;
        }
        let owed =
            if discarded < frame_duration(&frame) { tail_of(&frame, discarded) } else { frame };
        let at = self.refed.min(self.backlog.len());
        self.backlog.insert(at, owed);
        self.refed = at + 1;
    }

    pub fn cut_captured(&mut self, info: PlaybackInfo) {
        self.ledger.run_offset = self.ledger.run_offset.max(self.window_position(info));
        self.handed.clear();
    }

    pub fn source_queued(&mut self, now: Ts, queued: Duration) {
        self.ledger.dry_at = Some(now.saturating_add(queued));
    }

    /// One `wait_for_playout` window of `segment` drained. False when the segment is no longer
    /// the one finishing.
    pub fn settled(&mut self, segment: Option<PlaybackSegment>, info: PlaybackInfo) -> bool {
        if self.segment != segment {
            return false;
        }
        self.ledger.window_base += playhead(info);
        self.ledger.settled_played += info.played;
        true
    }

    /// `_wait_for_playout`'s finish over every window settled so far (a pause can make the source
    /// run dry more than once). `None` while the source still holds `queued` audio: the pause put
    /// some back, so the watcher waits for playout again.
    pub fn finish(
        &mut self,
        segment: Option<PlaybackSegment>,
        queued: Duration,
        now: Ts,
    ) -> Option<Vec<PlaybackReport>> {
        if !queued.is_zero() {
            return None;
        }
        if self.segment != segment || !self.ready_to_finish() || self.finish_held() {
            return Some(Vec::new());
        }
        let interrupted = self.phase == Phase::Cutting;
        let run = self.ledger.report_run(self.ledger.window_base, now, None);
        let mut reports = self.report(run).into_iter().collect::<Vec<_>>();
        let ledger = std::mem::take(&mut self.ledger);
        self.handed.clear();
        self.phase = Phase::Idle;
        if let Some(segment) = self.segment.take() {
            reports.push(PlaybackReport::Finished {
                segment,
                event: PlaybackFinished {
                    playback_position: ledger.played.max(ledger.settled_played),
                    interrupted,
                    synchronized_transcript: None,
                },
            });
        }
        Some(reports)
    }

    pub fn flush(&mut self) -> bool {
        self.pending = 0;
        let frames = self
            .bstream
            .flush()
            .into_iter()
            .map(|frame| {
                AudioFrame::new(
                    frame.data,
                    self.sample_rate,
                    self.num_channels,
                    frame.samples_per_channel,
                )
            })
            .collect::<Vec<_>>();
        let tail = !frames.is_empty() && !self.closed;
        if !self.closed {
            self.backlog.extend(frames);
        }
        let watch = self.phase == Phase::Capturing;
        if watch {
            self.phase = Phase::Flushed;
        }
        tail || watch
    }

    pub fn clear(&mut self) -> bool {
        self.bstream.clear();
        self.pending = 0;
        if !matches!(self.phase, Phase::Capturing | Phase::Flushed) {
            return false;
        }
        self.phase = Phase::Cutting;
        self.discard_backlog();
        true
    }

    pub fn pause(&mut self) -> bool {
        !std::mem::replace(&mut self.paused, true)
    }
    pub fn paused_clear(&mut self, discarded: Duration) {
        let frames = self.handed.take_last(discarded);
        if !self.closed {
            self.refed += frames.len();
            for frame in frames.into_iter().rev() {
                self.backlog.push_front(frame);
            }
        }
    }
    pub fn resume(&mut self) -> bool {
        std::mem::replace(&mut self.paused, false)
    }

    pub fn close(&mut self) -> bool {
        let captured = self.phase != Phase::Idle;
        self.closed = true;
        self.phase = Phase::Idle;
        self.paused = false;
        self.discard_backlog();
        self.handed.clear();
        self.bstream.clear();
        self.pending = 0;
        captured
    }

    pub fn interrupted(&mut self, source_played: Option<Duration>, now: Ts) -> Vec<PlaybackReport> {
        let captured = self.phase != Phase::Idle;
        self.phase = Phase::Idle;
        self.paused = false;
        self.discard_backlog();
        self.handed.clear();
        self.bstream.clear();
        self.pending = 0;
        let played = source_played
            .map(|played| self.ledger.settled_played + played)
            .unwrap_or(self.ledger.played)
            .max(self.ledger.played);
        let at = self.ledger.run_offset + played.saturating_sub(self.ledger.played);
        let run = self.ledger.report_run(at, now, None);
        let mut reports = self.report(run).into_iter().collect::<Vec<_>>();
        self.ledger = Ledger::default();
        if captured && let Some(segment) = self.segment.take() {
            reports.push(PlaybackReport::Finished {
                segment,
                event: PlaybackFinished {
                    playback_position: played,
                    interrupted: true,
                    synchronized_transcript: None,
                },
            });
        }
        reports
    }

    /// A capture the failed publication refused: the segment the capture may already have
    /// opened finishes interrupted at 0.
    pub fn capture_refused(open: Option<PlaybackSegment>) -> Option<PlaybackReport> {
        Some(PlaybackReport::Finished {
            segment: open?,
            event: PlaybackFinished {
                playback_position: Duration::ZERO,
                interrupted: true,
                synchronized_transcript: None,
            },
        })
    }

    pub fn close_finished(&mut self) -> Option<PlaybackReport> {
        let segment = self.segment.take()?;
        Some(PlaybackReport::Finished {
            segment,
            event: PlaybackFinished {
                playback_position: self.ledger.played,
                interrupted: true,
                synchronized_transcript: None,
            },
        })
    }

    fn report(&self, run: Option<Run>) -> Option<PlaybackReport> {
        let segment = self.segment?;
        let run = run?;
        Some(PlaybackReport::Progressed {
            segment,
            progress: Progress {
                started_at: run.started_at,
                offset: run.offset,
                duration: run.duration,
            },
        })
    }
    fn discard_backlog(&mut self) {
        self.backlog.clear();
        self.refed = 0;
    }
}

fn playhead(info: PlaybackInfo) -> Duration {
    info.played + info.discarded
}

#[derive(Default)]
enum PublicationPhase {
    #[default]
    Idle,
    Creating,
    Subscribing(TrackSid),
    Ready(TrackSid),
    Failed {
        error: IoError,
        sid: Option<TrackSid>,
    },
    Closed {
        creating: bool,
        sid: Option<TrackSid>,
    },
}

#[derive(Default)]
pub struct Publication {
    phase: PublicationPhase,
}

impl Publication {
    pub fn begin(&mut self) -> bool {
        if !matches!(self.phase, PublicationPhase::Idle) {
            return false;
        }
        self.phase = PublicationPhase::Creating;
        true
    }

    /// Install the source in the same transition, or retain it for retirement.
    pub fn adopt(&mut self, sid: TrackSid) -> bool {
        if !matches!(self.phase, PublicationPhase::Creating) {
            return false;
        }
        self.phase = PublicationPhase::Subscribing(sid);
        true
    }

    /// The published track, once published and until closed.
    pub fn sid(&self) -> Option<&TrackSid> {
        match &self.phase {
            PublicationPhase::Subscribing(sid) | PublicationPhase::Ready(sid) => Some(sid),
            _ => None,
        }
    }

    pub fn creation_retired(&mut self) {
        if let PublicationPhase::Closed { creating, .. } = &mut self.phase {
            *creating = false;
        }
    }

    pub fn creating(&self) -> bool {
        matches!(
            self.phase,
            PublicationPhase::Creating | PublicationPhase::Closed { creating: true, .. }
        )
    }

    pub fn ready(&mut self) {
        if let PublicationPhase::Subscribing(sid) = &self.phase {
            self.phase = PublicationPhase::Ready(sid.clone());
        }
    }

    pub fn fail(&mut self, error: IoError) {
        self.phase = match std::mem::take(&mut self.phase) {
            PublicationPhase::Creating => PublicationPhase::Failed { error, sid: None },
            PublicationPhase::Subscribing(sid) | PublicationPhase::Ready(sid) => {
                PublicationPhase::Failed { error, sid: Some(sid) }
            }
            PublicationPhase::Closed { sid, .. } => {
                PublicationPhase::Closed { creating: false, sid }
            }
            other => other,
        };
    }

    pub fn outcome(&self) -> Option<Result<()>> {
        match &self.phase {
            PublicationPhase::Ready(_) => Some(Ok(())),
            PublicationPhase::Failed { error, .. } => Some(Err(error.clone())),
            PublicationPhase::Closed { .. } => {
                Some(Err(IoError::Rtc("room audio output is closed".into())))
            }
            PublicationPhase::Idle
            | PublicationPhase::Creating
            | PublicationPhase::Subscribing(_) => None,
        }
    }

    pub fn close(&mut self) {
        self.phase = match std::mem::take(&mut self.phase) {
            PublicationPhase::Creating => PublicationPhase::Closed { creating: true, sid: None },
            PublicationPhase::Subscribing(sid) | PublicationPhase::Ready(sid) => {
                PublicationPhase::Closed { creating: false, sid: Some(sid) }
            }
            PublicationPhase::Failed { sid, .. } => {
                PublicationPhase::Closed { creating: false, sid }
            }
            PublicationPhase::Idle => PublicationPhase::Closed { creating: false, sid: None },
            closed @ PublicationPhase::Closed { .. } => closed,
        };
    }

    pub fn retire(&mut self) -> Option<TrackSid> {
        match &mut self.phase {
            PublicationPhase::Closed { sid, .. } => sid.take(),
            _ => None,
        }
    }
}
