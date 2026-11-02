//! Playback accounting and acceptance of a speech forwarding's sink reports.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::Now;
use crate::generation::effects::{
    AudioPlaybackAccountingEffect, AudioPlaybackCutEffect, AudioPlaybackFinishEffect,
    AudioPlaybackFlowEffect,
};
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
    Finished {
        segment: PlaybackSegment,
        ev: PlaybackFinished,
    },
    /// A word a synchroniser released, and the playback that released it.
    Shown(Option<PlaybackSegment>, lk_agents_abi::TranscriptChunk),
    Reset,
}

/// Accepted audio and playback receipts, without a sink, clock or task.
pub(in crate::generation) struct Playout {
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
    /// Every segment this forwarding's audio went to: the first, and one per sink reset.
    segments: Vec<PlaybackSegment>,
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
            segments: segment_key.into_iter().collect(),
            cut_below: false,
        }
    }

    /// A capture accepted by the sink; refused or cancelled captures count for nothing.
    pub fn captured(&mut self, key: Option<PlaybackSegment>, duration: Duration) {
        if self.first_segment.is_none() {
            self.first_segment = key;
        }
        // The segment the audio went to: after a sink reset the rest of this forwarding's audio
        // plays as a segment of its own. Every one's reports are this forwarding's.
        if let Some(key) = key {
            self.segment_key = Some(key);
            if !self.segments.contains(&key) {
                self.segments.push(key);
            }
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
    pub fn handle(&mut self, report: SinkReport, now: Now) -> Vec<AudioPlaybackAccountingEffect> {
        let speech = self.speech;
        let mut effects = Vec::new();
        match report {
            // Words a playback released are its forwarding's: never one that has no playback yet,
            // which a late report of another's would otherwise reach.
            SinkReport::Shown(segment, word) => {
                if segment.is_some_and(|segment| !self.segments.contains(&segment)) {
                    return effects;
                }
                effects.push(AudioPlaybackAccountingEffect::Trace(PlayoutStep::TranscriptShown {
                    speech,
                    text: word.text,
                    start: word.start_time.unwrap_or_default(),
                    end: word.end_time.unwrap_or_default(),
                    played: self.played,
                }))
            }
            // What played so far is heard; the rest plays from the segment's start again.
            SinkReport::Reset => {
                self.completed = self.played;
                effects
                    .push(AudioPlaybackAccountingEffect::Trace(PlayoutStep::SinkReset { speech }));
            }
            SinkReport::Playback(PlaybackEvent::Started(s)) => {
                if !self.owns(s.segment) {
                    return effects;
                }
                let at = lk_agents_abi::unix_secs(now.wall)
                    - now.at.saturating_duration_since(s.created_at).as_secs_f64();
                self.start(at, &mut effects);
            }
            SinkReport::Playback(PlaybackEvent::Progressed(p)) => {
                if !self.owns(p.segment) {
                    return effects;
                }
                let (offset, duration) = (p.offset, p.duration);
                effects.push(AudioPlaybackAccountingEffect::Trace(
                    PlayoutStep::PlaybackProgressed { speech, offset, duration },
                ));
                self.played += p.duration;
                self.report_played(&mut effects);
            }
            // Started/progressed only: a tracker reports finishes with their segment.
            SinkReport::Playback(PlaybackEvent::Finished { .. }) => {}
            SinkReport::Finished { segment, ev: f } => {
                if !self.owns(segment) {
                    return effects;
                }
                if f.interrupted && self.first_segment.is_some_and(|first| segment == first) {
                    self.cut_below = true;
                }
                // A sink with no `Started` (a bare recorder) still played it, unless cut at zero.
                if !(f.interrupted && f.playback_position.is_zero()) {
                    self.start(lk_agents_abi::unix_secs(now.wall), &mut effects);
                }
                self.played = self.played.max(self.completed + f.playback_position);
                if !f.interrupted {
                    self.completed += f.playback_position;
                }
                self.report_played(&mut effects);
            }
        }
        effects
    }

    /// Whether `segment` is one of this forwarding's playbacks; any is, before it has one.
    fn owns(&self, segment: PlaybackSegment) -> bool {
        self.segments.is_empty() || self.segments.contains(&segment)
    }

    fn start(&mut self, at: f64, effects: &mut Vec<AudioPlaybackAccountingEffect>) {
        if !std::mem::replace(&mut self.started, true) {
            effects.push(AudioPlaybackAccountingEffect::Started);
            self.owed = Some(at);
        }
    }

    fn report_played(&mut self, effects: &mut Vec<AudioPlaybackAccountingEffect>) {
        let len =
            if self.pushed == 0 { DEFAULT_FRAME_LEN } else { self.pushed_len / self.pushed as u32 };
        let len = len.as_nanos().max(1);
        let total = ((self.played.as_nanos() + len / 2) / len) as u64;
        if total > self.reported {
            let frames = total - self.reported;
            effects.push(AudioPlaybackAccountingEffect::Trace(PlayoutStep::AudioPlayed {
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
pub(in crate::generation) enum AudioPlaybackCutEvent {
    Probed(Option<PlaybackFinished>),
    Yielded { closing: bool },
    Waited(Option<PlaybackFinished>),
}

#[derive(Clone, Copy)]
enum CutPhase {
    Probing,
    Yielding,
    Waiting,
    Done,
}

/// A cut's bounded opportunity for an asynchronously delivered playback receipt.
pub(in crate::generation) struct CutReceipt {
    phase: CutPhase,
    probes: u8,
}

impl CutReceipt {
    pub fn begin(frames: u64) -> (Self, AudioPlaybackCutEffect) {
        if frames == 0 {
            return (Self { phase: CutPhase::Done, probes: 0 }, AudioPlaybackCutEffect::Unheard);
        }
        (Self { phase: CutPhase::Probing, probes: 0 }, AudioPlaybackCutEffect::Probe)
    }

    /// A closing session gives receipts scheduling turns, without advancing its stopped clock.
    pub fn handle(&mut self, event: AudioPlaybackCutEvent, now: Ts) -> AudioPlaybackCutEffect {
        match (self.phase, event) {
            (CutPhase::Probing, AudioPlaybackCutEvent::Probed(Some(report)))
            | (CutPhase::Waiting, AudioPlaybackCutEvent::Waited(Some(report))) => {
                self.phase = CutPhase::Done;
                AudioPlaybackCutEffect::Complete(Some(report))
            }
            (CutPhase::Probing, AudioPlaybackCutEvent::Probed(None)) => {
                self.probes += 1;
                self.phase = CutPhase::Yielding;
                AudioPlaybackCutEffect::Yield
            }
            (CutPhase::Yielding, AudioPlaybackCutEvent::Yielded { closing }) => {
                if self.probes < CUT_REPORT_HOPS {
                    self.phase = CutPhase::Probing;
                    AudioPlaybackCutEffect::Probe
                } else if closing {
                    self.phase = CutPhase::Done;
                    AudioPlaybackCutEffect::Complete(None)
                } else {
                    self.phase = CutPhase::Waiting;
                    AudioPlaybackCutEffect::WaitUntil(now.saturating_add(CUT_REPORT_WAIT))
                }
            }
            (CutPhase::Waiting, AudioPlaybackCutEvent::Waited(None)) => {
                self.phase = CutPhase::Done;
                AudioPlaybackCutEffect::Complete(None)
            }
            _ => panic!("cut receipt outcome does not match its requested operation"),
        }
    }
}
/// The available receipt at a forwarding's completion boundary.
pub(in crate::generation) enum Completion {
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
/// How the physical source or capture pump ended.
pub enum SegmentEnd {
    Audio,
    Cut,
    Provider(crate::agent_session::DriverError),
}

/// A source failure with only the sink's confirmed lower bound of heard audio.
pub struct SourceFailed {
    pub error: crate::agent_session::DriverError,
    pub synchronized: Option<String>,
    pub heard: Option<crate::transcription::Heard>,
}

impl From<crate::agent_session::DriverError> for SourceFailed {
    fn from(error: crate::agent_session::DriverError) -> Self {
        Self { error, synchronized: None, heard: None }
    }
}

/// What the requested drain, receipt application or clear actually completed with.
pub(in crate::generation) enum AudioPlaybackFinishEvent {
    Receipt(Result<Option<PlaybackFinished>, crate::agent_session::DriverError>),
    Accepted(PlaybackFinished),
    Cut(crate::transcription::Heard),
    Confirmed,
}

pub(in crate::generation) enum SegmentFinish {
    Waiting { text: String, source_error: Option<crate::agent_session::DriverError> },
    Accepting { text: String, source_error: Option<crate::agent_session::DriverError> },
    Cutting { source_error: Option<crate::agent_session::DriverError> },
    Confirming { error: crate::agent_session::DriverError, synchronized: Option<String> },
    Done,
}

impl SegmentFinish {
    pub fn begin(
        end: SegmentEnd,
        text: String,
        playout: &Playout,
    ) -> (Self, AudioPlaybackFinishEffect) {
        let source_error = match end {
            SegmentEnd::Cut => {
                return (
                    Self::Cutting { source_error: None },
                    AudioPlaybackFinishEffect::Cut(text),
                );
            }
            SegmentEnd::Provider(error) => Some(error),
            SegmentEnd::Audio => None,
        };
        if playout.pushed > 0 {
            return (Self::Waiting { text, source_error }, AudioPlaybackFinishEffect::Drain);
        }
        match source_error {
            Some(error) => {
                (Self::Confirming { error, synchronized: None }, AudioPlaybackFinishEffect::Confirm)
            }
            None => (
                Self::Done,
                AudioPlaybackFinishEffect::Complete(Ok(crate::transcription::Heard {
                    text,
                    complete: true,
                    ..crate::transcription::Heard::default()
                })),
            ),
        }
    }

    pub fn handle(
        &mut self,
        event: AudioPlaybackFinishEvent,
        playout: &Playout,
    ) -> AudioPlaybackFinishEffect {
        let state = std::mem::replace(self, Self::Done);
        match (state, event) {
            (
                Self::Waiting { text, source_error },
                AudioPlaybackFinishEvent::Receipt(Ok(Some(receipt))),
            ) => {
                *self = Self::Accepting { text, source_error };
                AudioPlaybackFinishEffect::Accept(receipt)
            }
            (
                Self::Waiting { text, source_error: Some(error) },
                AudioPlaybackFinishEvent::Receipt(Ok(None) | Err(_)),
            ) => {
                *self = Self::Cutting { source_error: Some(error) };
                AudioPlaybackFinishEffect::Cut(text)
            }
            (
                Self::Waiting { text, source_error: None },
                AudioPlaybackFinishEvent::Receipt(Ok(None)),
            ) => {
                *self = Self::Cutting { source_error: None };
                AudioPlaybackFinishEffect::Cut(text)
            }
            (
                Self::Waiting { source_error: None, .. },
                AudioPlaybackFinishEvent::Receipt(Err(error)),
            ) => self.confirm(error, None),
            (
                Self::Accepting { text, source_error },
                AudioPlaybackFinishEvent::Accepted(receipt),
            ) => match source_error {
                Some(error) => self.confirm(
                    error,
                    receipt
                        .synchronized_transcript
                        .map(|text| crate::transcription::Visible::of(&text).text),
                ),
                None => AudioPlaybackFinishEffect::Complete(Ok(
                    playout.heard(text, Completion::Played(receipt))
                )),
            },
            (Self::Cutting { source_error }, AudioPlaybackFinishEvent::Cut(heard)) => {
                match source_error {
                    Some(error) => self.confirm(error, None),
                    None => AudioPlaybackFinishEffect::Complete(Ok(heard)),
                }
            }
            (Self::Confirming { error, synchronized }, AudioPlaybackFinishEvent::Confirmed) => {
                AudioPlaybackFinishEffect::Complete(Err(Box::new(SourceFailed {
                    error,
                    synchronized,
                    heard: Some(playout.heard(String::new(), Completion::Unconfirmed)),
                })))
            }
            _ => panic!("playout completion does not match its requested operation"),
        }
    }
    fn confirm(
        &mut self,
        error: crate::agent_session::DriverError,
        synchronized: Option<String>,
    ) -> AudioPlaybackFinishEffect {
        *self = Self::Confirming { error, synchronized };
        AudioPlaybackFinishEffect::Confirm
    }
}

/// The next physical source item, before conversion for the destination's audio rate.
#[derive(Debug)]
pub enum Next {
    Frame(lk_agents_abi::AudioFrame),
    Boundary,
    End,
}

pub(in crate::generation) enum AudioPlaybackFlowEvent {
    Ready { interrupted: bool },
    Source { next: Option<Result<Next, crate::agent_session::DriverError>>, interrupted: bool },
    Launched(bool),
    Forwarded(Result<(), SegmentEnd>),
    TailPushed(Result<(), SegmentEnd>),
    BoundaryFlushed,
}

enum FlowPhase {
    Ready,
    Reading,
    Launching(lk_agents_abi::AudioFrame),
    Forwarding,
    Boundary,
    Flushing,
    Ending,
    Done,
}

/// Chooses how source boundaries, interruption and physical failures end a forwarding.
pub(in crate::generation) struct SegmentFlow {
    has_sink: bool,
    stop_at_boundary: bool,
    phase: FlowPhase,
}

impl SegmentFlow {
    pub fn new(has_sink: bool, stop_at_boundary: bool) -> Self {
        Self { has_sink, stop_at_boundary, phase: FlowPhase::Ready }
    }

    pub fn handle(
        &mut self,
        event: AudioPlaybackFlowEvent,
        pushed: u64,
    ) -> AudioPlaybackFlowEffect {
        let phase = std::mem::replace(&mut self.phase, FlowPhase::Done);
        match (phase, event) {
            (FlowPhase::Ready, AudioPlaybackFlowEvent::Ready { interrupted: true })
            | (FlowPhase::Reading, AudioPlaybackFlowEvent::Source { next: None, .. }) => {
                AudioPlaybackFlowEffect::Finish(SegmentEnd::Cut)
            }
            (FlowPhase::Ready, AudioPlaybackFlowEvent::Ready { interrupted: false }) => {
                self.phase = FlowPhase::Reading;
                AudioPlaybackFlowEffect::Read
            }
            (FlowPhase::Reading, AudioPlaybackFlowEvent::Source { next: Some(Err(error)), .. }) => {
                AudioPlaybackFlowEffect::Finish(SegmentEnd::Provider(error))
            }
            (
                FlowPhase::Reading,
                AudioPlaybackFlowEvent::Source { next: Some(Ok(next)), interrupted },
            ) => match next {
                Next::Frame(_) if !self.has_sink => self.ready(),
                Next::Frame(_) if interrupted => AudioPlaybackFlowEffect::Finish(SegmentEnd::Cut),
                Next::Frame(frame) if pushed == 0 => {
                    self.phase = FlowPhase::Launching(frame);
                    AudioPlaybackFlowEffect::WaitLaunch
                }
                Next::Frame(frame) => {
                    self.phase = FlowPhase::Forwarding;
                    AudioPlaybackFlowEffect::Frame(frame)
                }
                Next::Boundary if !self.stop_at_boundary && !self.has_sink => self.ready(),
                Next::Boundary if !self.stop_at_boundary && interrupted => {
                    AudioPlaybackFlowEffect::Finish(SegmentEnd::Cut)
                }
                Next::Boundary | Next::End if !self.has_sink => {
                    AudioPlaybackFlowEffect::Finish(SegmentEnd::Audio)
                }
                Next::Boundary if !self.stop_at_boundary => {
                    self.phase = FlowPhase::Boundary;
                    AudioPlaybackFlowEffect::Tail
                }
                Next::Boundary | Next::End => {
                    self.phase = FlowPhase::Ending;
                    AudioPlaybackFlowEffect::Tail
                }
            },
            (FlowPhase::Launching(_), AudioPlaybackFlowEvent::Launched(false)) => {
                AudioPlaybackFlowEffect::Finish(SegmentEnd::Cut)
            }
            (FlowPhase::Launching(frame), AudioPlaybackFlowEvent::Launched(true)) => {
                self.phase = FlowPhase::Forwarding;
                AudioPlaybackFlowEffect::Frame(frame)
            }
            (FlowPhase::Forwarding, AudioPlaybackFlowEvent::Forwarded(Ok(()))) => self.ready(),
            (FlowPhase::Forwarding, AudioPlaybackFlowEvent::Forwarded(Err(end)))
            | (
                FlowPhase::Boundary | FlowPhase::Ending,
                AudioPlaybackFlowEvent::TailPushed(Err(end)),
            ) => AudioPlaybackFlowEffect::Finish(end),
            (FlowPhase::Boundary, AudioPlaybackFlowEvent::TailPushed(Ok(()))) => {
                self.phase = FlowPhase::Flushing;
                AudioPlaybackFlowEffect::FlushBoundary
            }
            (FlowPhase::Ending, AudioPlaybackFlowEvent::TailPushed(Ok(()))) => {
                AudioPlaybackFlowEffect::Finish(SegmentEnd::Audio)
            }
            (FlowPhase::Flushing, AudioPlaybackFlowEvent::BoundaryFlushed) => self.ready(),
            _ => panic!("source outcome does not match the requested forwarding operation"),
        }
    }

    fn ready(&mut self) -> AudioPlaybackFlowEffect {
        self.phase = FlowPhase::Ready;
        AudioPlaybackFlowEffect::CheckInterruption
    }
}

/// A source with no destination still drains until its boundary, end, failure or interruption.
pub(in crate::generation) fn drained(
    end: SegmentEnd,
    text: String,
) -> Result<crate::transcription::Heard, Box<SourceFailed>> {
    match end {
        SegmentEnd::Provider(error) => Err(Box::new(error.into())),
        SegmentEnd::Audio | SegmentEnd::Cut => Ok(crate::transcription::Heard {
            text,
            complete: matches!(end, SegmentEnd::Audio),
            ..crate::transcription::Heard::default()
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(segment: u64) -> PlaybackSegment {
        PlaybackSegment { attachment: 0, segment }
    }

    fn now() -> Now {
        Now { at: Ts::ZERO, wall: std::time::UNIX_EPOCH }
    }

    // Catches: after a sink reset the playout kept only the rest's key, so the first segment's
    // cut, reported after the rest's first capture, was dropped and the forwarding read as
    // played whole.
    #[test]
    fn the_first_segments_late_cut_counts_after_a_reset() {
        let frame = Duration::from_millis(20);
        let mut playout = Playout::new(SpeechId(1), None);
        playout.captured(Some(segment(1)), frame);
        playout.handle(SinkReport::Reset, now());
        playout.captured(Some(segment(2)), frame);
        let cut = PlaybackFinished { interrupted: true, ..PlaybackFinished::default() };
        playout.handle(SinkReport::Finished { segment: segment(1), ev: cut }, now());
        assert!(playout.cut_below, "the first segment's cut was dropped");
    }

    // Catches: a playout with no playback yet took the words a late report of another
    // forwarding's playback released, as no key filtered them.
    #[test]
    fn a_playout_with_no_playback_takes_no_released_words() {
        let mut playout = Playout::new(SpeechId(1), None);
        let word = lk_agents_abi::TranscriptChunk::new("late".to_owned());
        let effects = playout.handle(SinkReport::Shown(Some(segment(7)), word), now());
        assert!(effects.is_empty(), "a word another playback released was taken");
    }
}
