//! Playback accounting and acceptance of a speech forwarding's sink reports.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::Now;
use crate::trace::PlayoutStep;
use lk_agents_abi::Ts;
use lk_agents_abi::voice::io::{PlaybackEvent, PlaybackFinished, PlaybackSegment};
use std::time::Duration;

const DEFAULT_FRAME_LEN: Duration = Duration::from_millis(20);
/// The transport allowance once all accepted audio should have played.
const LOST_FINISH_MARGIN: Duration = Duration::from_secs(2);

/// A physical sink observation, delivered in arrival order.
pub enum SinkReport {
    Playback(PlaybackEvent),
    Finished { segment: PlaybackSegment, ev: PlaybackFinished },
    Shown(lk_agents_abi::TimedString),
    Reset,
}

/// An accepted observation's publications.
pub enum PlayoutEffect {
    Trace(PlayoutStep),
    Started,
}

/// Accepted audio and playback receipts, without a sink, clock or task.
pub struct Playout {
    speech: SpeechId,
    pub segment_key: Option<PlaybackSegment>,
    pub finish_due: Option<Ts>,
    pub started: bool,
    /// The first-frame instant owed to the segment's `started_speaking`, until it is free.
    owed: Option<f64>,
    /// The frame length `AudioPlayed` counts in.
    pub pushed: u64,
    pushed_len: Duration,
    pub played: Duration,
    /// What the segments the sink reported played out held: the forwarding's heard audio before
    /// the segment now playing.
    pub completed: Duration,
    reported: u64,
    /// Pushed since the last boundary: one `SinkCall::Push` record per stretch.
    pub batch: u64,
    /// The tracker's number for this forwarding's first segment, once a frame opened it.
    first_segment: Option<PlaybackSegment>,
    /// The sink reported one of this forwarding's segments cut under it (a reset): not all of
    /// it played, whatever the last segment reports.
    pub cut_below: bool,
}

impl Playout {
    pub fn new(speech: SpeechId, segment_key: Option<PlaybackSegment>) -> Self {
        Self {
            speech,
            segment_key,
            finish_due: None,
            started: false,
            owed: None,
            pushed: 0,
            pushed_len: Duration::ZERO,
            played: Duration::ZERO,
            completed: Duration::ZERO,
            reported: 0,
            batch: 0,
            first_segment: None,
            cut_below: false,
        }
    }

    /// A capture accepted by the sink; refused or cancelled captures count for nothing.
    pub fn captured(&mut self, key: Option<PlaybackSegment>, duration: Duration) {
        if self.first_segment.is_none() {
            self.first_segment = key;
            self.segment_key = self.first_segment;
        }
        self.pushed += 1;
        self.pushed_len += duration;
        self.batch += 1;
    }

    /// Whether accepted audio owes a flush, taken before the physical operation.
    pub fn take_flush(&mut self) -> bool {
        std::mem::take(&mut self.batch) > 0
    }

    /// The physical flush finished at the supplied instant.
    pub fn flushed(&mut self, now: Ts) -> PlayoutStep {
        let remaining = self.pushed_len.saturating_sub(self.played);
        let at = now.saturating_add(remaining).saturating_add(LOST_FINISH_MARGIN);
        self.finish_due = Some(at);
        PlayoutStep::FinishDue { speech: self.speech, at }
    }

    /// Settle the first-frame instant once the source is free of an outstanding read.
    pub fn take_started(&mut self) -> Option<f64> {
        self.owed.take()
    }

    /// Reports for an old playback key never release or credit the current forwarding.
    pub fn handle(&mut self, report: SinkReport, now: Now) -> Vec<PlayoutEffect> {
        let speech = self.speech;
        let mut effects = Vec::new();
        match report {
            SinkReport::Shown(word) => {
                effects.push(PlayoutEffect::Trace(PlayoutStep::TranscriptShown {
                    speech,
                    text: word.text,
                    start: word.start_time.unwrap_or_default(),
                    end: word.end_time.unwrap_or_default(),
                    played: self.played,
                }))
            }
            SinkReport::Reset => {
                effects.push(PlayoutEffect::Trace(PlayoutStep::SinkReset { speech }));
            }
            SinkReport::Playback(PlaybackEvent::Started(s)) => {
                if self.segment_key.is_some_and(|key| key != s.segment) {
                    return effects;
                }
                let at = lk_agents_abi::unix_secs(now.wall)
                    - now.at.saturating_duration_since(s.created_at).as_secs_f64();
                self.start(at, &mut effects);
            }
            SinkReport::Playback(PlaybackEvent::Progressed(p)) => {
                if self.segment_key.is_some_and(|key| key != p.segment) {
                    return effects;
                }
                let (offset, duration) = (p.offset, p.duration);
                effects.push(PlayoutEffect::Trace(PlayoutStep::PlaybackProgressed {
                    speech,
                    offset,
                    duration,
                }));
                self.played += p.duration;
                self.report_played(&mut effects);
            }
            // Started/progressed only: a tracker reports finishes with their segment.
            SinkReport::Playback(PlaybackEvent::Finished { .. }) => {}
            SinkReport::Finished { segment, ev: f } => {
                if self.segment_key.is_some_and(|key| key != segment) {
                    return effects;
                }
                if f.interrupted && self.first_segment.is_some_and(|first| segment == first) {
                    self.cut_below = true;
                }
                // A sink with no `Started` (a bare recorder) still played it, unless cut at zero.
                if !(f.interrupted && f.playback_position.is_zero()) {
                    self.start(lk_agents_abi::unix_secs(now.wall), &mut effects);
                }
                self.played = self.played.max(f.playback_position);
                if !f.interrupted {
                    self.completed += f.playback_position;
                }
                self.report_played(&mut effects);
            }
        }
        effects
    }

    fn start(&mut self, at: f64, effects: &mut Vec<PlayoutEffect>) {
        if !std::mem::replace(&mut self.started, true) {
            effects.push(PlayoutEffect::Started);
            self.owed = Some(at);
        }
    }

    fn report_played(&mut self, effects: &mut Vec<PlayoutEffect>) {
        let len =
            if self.pushed == 0 { DEFAULT_FRAME_LEN } else { self.pushed_len / self.pushed as u32 };
        let len = len.as_nanos().max(1);
        let total = ((self.played.as_nanos() + len / 2) / len) as u64;
        if total > self.reported {
            let frames = total - self.reported;
            effects.push(PlayoutEffect::Trace(PlayoutStep::AudioPlayed {
                speech: self.speech,
                frames,
            }));
            self.reported = total;
        }
    }
}
/// Scheduling turns given to a cut sink before awaiting its transport receipt.
const CUT_REPORT_HOPS: u8 = 16;
/// A host loop may deliver the cut receipt after those scheduling turns.
const CUT_REPORT_WAIT: Duration = Duration::from_secs(1);

/// The answer to one physical probe or wait for a cut segment's receipt.
pub enum CutEvent {
    Probed(Option<PlaybackFinished>),
    Yielded { closing: bool },
    Waited(Option<PlaybackFinished>),
}

/// One physical operation requested by the cut receipt policy.
pub enum CutEffect {
    Probe,
    Yield,
    WaitUntil(Ts),
    Complete(Option<PlaybackFinished>),
    Unheard,
}

#[derive(Clone, Copy)]
enum CutPhase {
    Probing,
    Yielding,
    Waiting,
    Done,
}

/// A cut's bounded opportunity for an asynchronously delivered playback receipt.
pub struct CutReceipt {
    phase: CutPhase,
    probes: u8,
}

impl CutReceipt {
    pub fn begin(frames: u64) -> (Self, CutEffect) {
        if frames == 0 {
            return (Self { phase: CutPhase::Done, probes: 0 }, CutEffect::Unheard);
        }
        (Self { phase: CutPhase::Probing, probes: 0 }, CutEffect::Probe)
    }

    /// A closing session gives receipts scheduling turns, without advancing its stopped clock.
    pub fn handle(&mut self, event: CutEvent, now: Ts) -> CutEffect {
        match (self.phase, event) {
            (CutPhase::Probing, CutEvent::Probed(Some(report)))
            | (CutPhase::Waiting, CutEvent::Waited(Some(report))) => {
                self.phase = CutPhase::Done;
                CutEffect::Complete(Some(report))
            }
            (CutPhase::Probing, CutEvent::Probed(None)) => {
                self.probes += 1;
                self.phase = CutPhase::Yielding;
                CutEffect::Yield
            }
            (CutPhase::Yielding, CutEvent::Yielded { closing }) => {
                if self.probes < CUT_REPORT_HOPS {
                    self.phase = CutPhase::Probing;
                    CutEffect::Probe
                } else if closing {
                    self.phase = CutPhase::Done;
                    CutEffect::Complete(None)
                } else {
                    self.phase = CutPhase::Waiting;
                    CutEffect::WaitUntil(now.saturating_add(CUT_REPORT_WAIT))
                }
            }
            (CutPhase::Waiting, CutEvent::Waited(None)) => {
                self.phase = CutPhase::Done;
                CutEffect::Complete(None)
            }
            _ => panic!("cut receipt outcome does not match its requested operation"),
        }
    }
}
/// The available receipt at a forwarding's completion boundary.
pub enum Completion {
    Played(PlaybackFinished),
    Cut(PlaybackFinished),
    CutUnreported { position: Duration },
    Unconfirmed,
}

impl Playout {
    /// Project only confirmed audio, preserving a synchronizer's transcript when it supplied one.
    pub fn heard(&self, text: String, completion: Completion) -> crate::transcription::Heard {
        use crate::transcription::{Heard, Visible};
        match completion {
            Completion::Played(ev) => Heard {
                text: ev.synchronized_transcript.map(|t| Visible::of(&t).text).unwrap_or(text),
                position: ev.playback_position,
                complete: !ev.interrupted && !self.cut_below,
                frames: self.pushed,
                played: self.completed.max(ev.playback_position),
            },
            Completion::Cut(ev) => {
                let cut = if ev.interrupted { ev.playback_position } else { Duration::ZERO };
                Heard {
                    text: ev.synchronized_transcript.map(|t| Visible::of(&t).text).unwrap_or(text),
                    position: ev.playback_position,
                    complete: false,
                    frames: self.pushed,
                    played: self.completed + cut,
                }
            }
            Completion::CutUnreported { position } => Heard {
                text,
                position,
                complete: false,
                frames: self.pushed,
                played: self.completed + position,
            },
            Completion::Unconfirmed => Heard {
                text: String::new(),
                position: self.played.saturating_sub(self.completed),
                complete: false,
                frames: self.pushed,
                played: self.played,
            },
        }
    }
}
