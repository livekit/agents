//! The transcript synchronizer: one machine for the transcript gate and its segments' pacing.
//!
//! Both legs start detached; the gate opens only with the audio and the text output both
//! attached, so words with no synchronised audio pass straight through. With it open, each leg is
//! an ordered stream of segments delimited by its flushes: a segment's text is the next text not
//! yet ended, its audio the next audio, so a segment the producer ended with no audio is one by
//! construction even when its words arrive later (through a `transcription_node`). The sink's
//! playback moves the oldest segment it owes; each segment's [`TranscriptPacer`] decides when its
//! words show.

use super::pacer::{TranscriptPacer, TranscriptPacerEvent};
pub use super::pacer::{TranscriptPacerOptions, TranscriptPacerWarning};
use crate::transcription::TranscriptSegment;
use crate::transcription::effects::TranscriptSyncEffect;
use lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
use lk_agents_abi::voice::io::PlaybackSegment;
use lk_agents_abi::{TranscriptChunk, Ts};
use std::collections::VecDeque;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptOutput {
    Word(TranscriptChunk),
    Flush,
    Detached,
}

pub enum TranscriptSyncEvent {
    /// Count the frame before capture: the sink may report playback inside its capture call.
    AudioCaptureStarted {
        secs: f64,
        /// The playback the frame goes to, as the audio node's tracker keyed it.
        segment: Option<PlaybackSegment>,
    },
    /// A silence in the audio captured so far, in seconds of the segment's audio.
    AudioPause {
        start: f64,
        end: f64,
    },
    AudioEnded,
    /// The sink's audio was thrown away at `at`.
    AudioCleared {
        at: Ts,
    },
    /// The sink's playback started at `at`: from there its words move with elapsed time, as far
    /// as the audio sent to it.
    PlaybackStarted {
        at: Ts,
    },
    /// The sink's playhead, reported at `at`: it corrects the elapsed-time estimate.
    PlaybackProgressed {
        position: Duration,
        at: Ts,
    },
    PlaybackPaused {
        at: Ts,
    },
    PlaybackResumed {
        at: Ts,
    },
    /// A [`TranscriptSyncEffect::TickAt`] came due; `at` is now.
    Tick(Ts),
    PlaybackFinished {
        position: Duration,
        interrupted: bool,
    },
    PlaybackReset,
    Capture {
        text: TranscriptChunk,
        aligned: bool,
    },
    FlushText,
    AttachText,
    AttachmentChanged {
        audio: Option<bool>,
        text: Option<bool>,
    },
    DetachText,
    TextRetired {
        admitted: bool,
    },
    Output(TranscriptOutput),
    Rotate,
    Close,
    WaitStarted(Ts),
    WaitFinished,
}

#[derive(Clone, Copy)]
enum ClosePhase {
    Open,
    Closing,
    Waiting,
    Closed,
}

/// The time stamped on a word forwarded with no pace to keep.
const UNPACED_TIME: f64 = 0.0;

/// A tick lands this much after the playhead reaches a word, so rounding never leaves it short.
const TICK_MARGIN: Duration = Duration::from_micros(1);

/// One segment as the synchroniser sees it.
struct Segment {
    /// `None` once closed while its audio is still the sink's (a rotation, synchronisation
    /// switched off), or opened while synchronisation was off: its playback moves no pacer.
    pacer: Option<TranscriptPacer>,
    text_ended: bool,
    /// A frame reached the sink: the sink owes this segment's playback finish.
    audio: bool,
    /// The playback its latest frame went to, as the sink's tracker keyed it.
    key: Option<PlaybackSegment>,
    audio_ended: bool,
    /// The sink reported this segment's playback started.
    started: bool,
    /// Where its playout was last known to be, from its start or the sink's last report.
    playhead: Option<Playhead>,
    /// The sink cut this segment's playout while its audio was still coming. It owes no finish
    /// until a reset starts the rest of its audio as a playout of its own.
    cut_below: bool,
    /// A clear closed its words before any of its audio and before its audio leg ended: the
    /// leg's late end is its, not a new segment's. Gone at the next words or audio: a cut leg
    /// usually ends with no audio end at all, and the next speech's end is its own.
    cleared: bool,
}

/// A playout's position when it was last known: it moves on with elapsed time from there.
#[derive(Debug, Clone, Copy)]
struct Playhead {
    position: Duration,
    at: Ts,
}

pub struct TranscriptSync {
    opts: TranscriptPacerOptions,
    audio_attached: bool,
    text_attached: bool,
    text_attachments: usize,
    capturing_text: bool,
    /// Text reached the transcript output since its last flush.
    segment: TranscriptSegment,
    close: ClosePhase,
    /// Both legs attached: words are paced against the audio.
    enabled: bool,
    segments: VecDeque<Segment>,
    /// The sink was paused: no playhead moves with elapsed time until it resumes.
    paused: bool,
    /// The earliest tick asked for and not yet fed back.
    scheduled: Option<Ts>,
}

impl TranscriptSync {
    pub fn new(opts: TranscriptPacerOptions) -> Self {
        Self {
            opts,
            audio_attached: false,
            text_attached: false,
            text_attachments: 0,
            capturing_text: false,
            segment: TranscriptSegment::default(),
            close: ClosePhase::Open,
            enabled: false,
            segments: VecDeque::new(),
            paused: false,
            scheduled: None,
        }
    }

    pub fn handle(&mut self, event: TranscriptSyncEvent) -> Vec<TranscriptSyncEffect> {
        match event {
            TranscriptSyncEvent::AudioCaptureStarted { secs, segment } => {
                self.pace(|s, out| s.on_audio(secs, segment, out))
            }
            TranscriptSyncEvent::AudioPause { start, end } => self.pace(|s, out| {
                if let Some(i) = s.segments.iter().position(|s| !s.audio_ended) {
                    s.apply(i, TranscriptPacerEvent::Pause { start, end }, out);
                }
            }),
            TranscriptSyncEvent::AudioEnded => self.pace(Self::on_audio_end),
            TranscriptSyncEvent::AudioCleared { at } => self.pace(|s, out| s.on_clear(at, out)),
            TranscriptSyncEvent::PlaybackStarted { at } => {
                self.pace(|s, out| s.on_started(at, out))
            }
            TranscriptSyncEvent::PlaybackProgressed { position, at } => self.pace(|s, out| {
                // The sink's own report corrects the estimate.
                if let Some(i) = s.playing() {
                    s.segments[i].playhead = Some(Playhead { position, at });
                    s.apply(i, TranscriptPacerEvent::Position(position), out);
                }
            }),
            TranscriptSyncEvent::PlaybackPaused { at } => self.pace(|s, out| {
                if let Some(i) = s.playing() {
                    s.advance(i, at, out);
                    s.apply(i, TranscriptPacerEvent::Paused, out);
                }
                s.paused = true;
            }),
            TranscriptSyncEvent::PlaybackResumed { at } => self.pace(|s, _| {
                // The playhead stood still while paused: it moves on from where it stopped.
                if std::mem::take(&mut s.paused)
                    && let Some(i) = s.playing()
                    && let Some(head) = &mut s.segments[i].playhead
                {
                    head.at = at;
                }
            }),
            TranscriptSyncEvent::Tick(at) => self.pace(|s, out| {
                s.scheduled = None;
                if let Some(i) = s.playing() {
                    s.advance(i, at, out);
                }
            }),
            TranscriptSyncEvent::PlaybackFinished { position, interrupted } => {
                // A finished segment's `Heard` is its synchronised transcript: kept, unlike any
                // other input's.
                let out = self.on_finished(position, interrupted);
                self.publish(out)
            }
            TranscriptSyncEvent::PlaybackReset => self.pace(Self::on_reset),
            TranscriptSyncEvent::Capture { text, aligned } => {
                if !self.enabled {
                    self.segment.wrote(&text.text);
                    return vec![TranscriptSyncEffect::Capture { text, aligned }];
                }
                self.capturing_text = true;
                let (start_time, end_time) =
                    if aligned { (text.start_time, text.end_time) } else { (None, None) };
                self.pace(|s, out| s.on_text(text, start_time, end_time, out))
            }
            TranscriptSyncEvent::FlushText => {
                if !self.enabled {
                    self.segment
                        .flush()
                        .then_some(TranscriptSyncEffect::FlushText)
                        .into_iter()
                        .collect()
                } else if std::mem::take(&mut self.capturing_text) {
                    self.pace(Self::on_text_end)
                } else {
                    Vec::new()
                }
            }
            TranscriptSyncEvent::AttachText => {
                self.text_attachments += 1;
                vec![TranscriptSyncEffect::AttachText]
            }
            TranscriptSyncEvent::AttachmentChanged { audio, text } => {
                self.audio_attached = audio.unwrap_or(self.audio_attached);
                self.text_attached = text.unwrap_or(self.text_attached);
                let enabled = self.audio_attached && self.text_attached;
                self.pace(|s, out| s.enable(enabled, out))
            }
            TranscriptSyncEvent::DetachText => {
                let attachments = self.text_attachments;
                let mut effects = Vec::new();
                if attachments <= 1 {
                    self.text_attached = false;
                    effects.extend(self.pace(|s, out| s.enable(false, out)));
                }
                // The final rotation still belongs to its admitted physical attachment.
                effects.push(TranscriptSyncEffect::RetireText { admitted: attachments > 0 });
                effects
            }
            TranscriptSyncEvent::TextRetired { admitted } => {
                if admitted {
                    self.text_attachments -= 1;
                    vec![TranscriptSyncEffect::Queue(TranscriptOutput::Detached)]
                } else {
                    Vec::new()
                }
            }
            TranscriptSyncEvent::Output(output) => {
                if matches!(output, TranscriptOutput::Word(_) | TranscriptOutput::Flush)
                    && self.text_attachments == 0
                {
                    Vec::new()
                } else {
                    vec![TranscriptSyncEffect::Queue(output)]
                }
            }
            TranscriptSyncEvent::Rotate => match self.close {
                ClosePhase::Open => self.pace(Self::rotate),
                _ => Vec::new(),
            },
            TranscriptSyncEvent::Close => {
                if !matches!(self.close, ClosePhase::Open) {
                    return Vec::new();
                }
                self.close = ClosePhase::Closing;
                let mut effects =
                    if self.text_attachments > 0 { self.pace(Self::rotate) } else { Vec::new() };
                effects
                    .extend([TranscriptSyncEffect::CloseSender, TranscriptSyncEffect::BeginWait]);
                effects
            }
            TranscriptSyncEvent::WaitStarted(at) => {
                if !matches!(self.close, ClosePhase::Closing) {
                    return Vec::new();
                }
                self.close = ClosePhase::Waiting;
                vec![TranscriptSyncEffect::WaitUntil(at.saturating_add(HOST_TEARDOWN_TIMEOUT))]
            }
            TranscriptSyncEvent::WaitFinished => {
                self.close = ClosePhase::Closed;
                Vec::new()
            }
        }
    }

    /// Runs a step over the segments. A `Heard` it produces is dropped: only a finished segment
    /// reports what was heard.
    fn pace(
        &mut self,
        step: impl FnOnce(&mut Self, &mut Vec<TranscriptSyncEffect>),
    ) -> Vec<TranscriptSyncEffect> {
        let mut out = Vec::new();
        step(self, &mut out);
        out.retain(|e| !matches!(e, TranscriptSyncEffect::ReportHeard(_)));
        self.publish(out)
    }

    /// The segments' effects, a flush only when the output had text since its last one, and the
    /// next tick they need.
    fn publish(&mut self, out: Vec<TranscriptSyncEffect>) -> Vec<TranscriptSyncEffect> {
        let mut effects: Vec<_> = out
            .into_iter()
            .filter(|effect| match effect {
                TranscriptSyncEffect::PublishWord { chunk, .. } => {
                    self.segment.wrote(&chunk.text);
                    true
                }
                TranscriptSyncEffect::FlushTranscript => self.segment.flush(),
                _ => true,
            })
            .collect();
        effects.extend(self.wake().map(TranscriptSyncEffect::TickAt));
        effects
    }

    /// The gate opened or closed: the segments so far end where they stand.
    fn enable(&mut self, enabled: bool, out: &mut Vec<TranscriptSyncEffect>) {
        if self.enabled != enabled {
            self.enabled = enabled;
            self.rotate(out);
        }
    }

    fn on_text(
        &mut self,
        chunk: lk_agents_abi::TranscriptChunk,
        start_time: Option<f64>,
        end_time: Option<f64>,
        out: &mut Vec<TranscriptSyncEffect>,
    ) {
        self.segments.retain(|s| !s.cleared);
        let i = self.next(|s| !s.text_ended);
        if self.segments[i].pacer.is_none() {
            // A segment closed mid-write: its words have no pace to keep.
            let (start_time, end_time) = (UNPACED_TIME, UNPACED_TIME);
            out.push(TranscriptSyncEffect::PublishWord { chunk, start_time, end_time });
        } else {
            self.apply(i, TranscriptPacerEvent::Text { chunk, start_time, end_time }, out);
        }
    }

    fn on_text_end(&mut self, out: &mut Vec<TranscriptSyncEffect>) {
        if let Some(i) = self.segments.iter().position(|s| !s.text_ended) {
            self.segments[i].text_ended = true;
            self.apply(i, TranscriptPacerEvent::TextEnd, out);
            self.settle(i);
        }
    }

    fn on_audio(
        &mut self,
        secs: f64,
        key: Option<PlaybackSegment>,
        out: &mut Vec<TranscriptSyncEffect>,
    ) {
        self.segments.retain(|s| !s.cleared);
        let i = self.next(|s| !s.audio_ended);
        self.segments[i].audio = true;
        self.segments[i].key = key.or(self.segments[i].key);
        self.apply(i, TranscriptPacerEvent::Audio { secs }, out);
    }

    fn on_audio_end(&mut self, out: &mut Vec<TranscriptSyncEffect>) {
        // Ended before any of its words arrived: a segment with no audio, unless its
        // words will pass unpaced.
        if !self.enabled && self.segments.iter().all(|s| s.audio_ended) {
            return;
        }
        let i = self.next(|s| !s.audio_ended);
        if self.segments[i].cleared {
            // The cut speech's leg ended after its clear: a new segment here would pair the
            // next speech's words with no audio.
            self.segments.remove(i);
            return;
        }
        self.segments[i].audio_ended = true;
        self.apply(i, TranscriptPacerEvent::AudioEnd, out);
        if self.segments[i].cut_below {
            // Cut below and no reset came: the sink owes it no finish.
            self.apply(i, TranscriptPacerEvent::Close, out);
            self.segments.remove(i);
        } else {
            self.settle(i);
        }
    }

    fn on_clear(&mut self, at: Ts, out: &mut Vec<TranscriptSyncEffect>) {
        self.stop_playouts(at, out);
        if !self.segments.back().is_some_and(|s| !s.audio) {
            return;
        }
        let i = self.segments.len() - 1;
        self.apply(i, TranscriptPacerEvent::Close, out);
        let segment = &mut self.segments[i];
        if segment.audio_ended {
            self.segments.pop_back();
        } else {
            (segment.pacer, segment.text_ended, segment.cleared) = (None, true, true);
        }
    }

    /// The sink dropped the audio of every segment it owed: each one's words stand where its
    /// playout was at `at`, and what was heard of it by then is reported for its playback, which
    /// may never report its own finish.
    fn stop_playouts(&mut self, at: Ts, out: &mut Vec<TranscriptSyncEffect>) {
        let playing = self.playing();
        for i in 0..self.segments.len() {
            let segment = &self.segments[i];
            let Some(pacer) = segment.pacer.as_ref().filter(|p| live(p) && !p.is_cleared()) else {
                continue;
            };
            if !segment.audio || segment.cut_below {
                continue;
            }
            let (key, had_text) = (segment.key, pacer.saw_text());
            if playing == Some(i) {
                self.advance(i, at, out);
            }
            let position = self.segments[i].pacer.as_ref().map(TranscriptPacer::position);
            let position = position.unwrap_or_default();
            let mut cleared = Vec::new();
            self.apply(i, TranscriptPacerEvent::Cleared { position }, &mut cleared);
            for effect in cleared {
                match (effect, key) {
                    // Audio with no words: no transcript, so the speech's own text stands.
                    (TranscriptSyncEffect::ReportHeard(heard), Some(segment)) if had_text => {
                        out.push(TranscriptSyncEffect::ReportCleared { segment, heard })
                    }
                    (TranscriptSyncEffect::ReportHeard(_), _) => {}
                    (effect, _) => out.push(effect),
                }
            }
        }
    }

    /// The tick that next moves the playing segment's words, unless one at or before it is
    /// already asked for. `None` while paused, with nothing waiting on the playhead, or with the
    /// next word past the audio sent: the next input asks again.
    fn wake(&mut self) -> Option<Ts> {
        let wake = self.next_wake()?;
        if self.scheduled.is_some_and(|armed| armed <= wake) {
            return None;
        }
        self.scheduled = Some(wake);
        Some(wake)
    }

    fn next_wake(&self) -> Option<Ts> {
        if self.paused {
            return None;
        }
        let segment = &self.segments[self.playing()?];
        let head = segment.playhead?;
        let pacer = segment.pacer.as_ref().filter(|p| !p.is_over())?;
        let due = pacer.next_due()?;
        if due > pacer.audio_sent().max(head.position) {
            return None;
        }
        Some(head.at.saturating_add(due.saturating_sub(head.position) + TICK_MARGIN))
    }

    /// Moves segment `i` to where its playout is estimated to be at `at`: its last known
    /// position plus the time since, unless paused, and never past the audio sent.
    fn advance(&mut self, i: usize, at: Ts, out: &mut Vec<TranscriptSyncEffect>) {
        let segment = &mut self.segments[i];
        let (Some(head), Some(pacer)) = (&mut segment.playhead, &segment.pacer) else { return };
        if !self.paused {
            let moved = head.position + at.saturating_duration_since(head.at);
            let position = moved.min(pacer.audio_sent()).max(head.position);
            *head = Playhead { position, at };
        }
        let position = head.position;
        self.apply(i, TranscriptPacerEvent::Position(position), out);
    }

    fn on_started(&mut self, at: Ts, out: &mut Vec<TranscriptSyncEffect>) {
        // A sink plays its segments in order: one starting while an older one it had
        // started is still owed means that one's finish was lost. It ends where it stood.
        if let Some(i) = self.playing()
            && self.segments[i].started
            && self.segments.iter().skip(i + 1).any(|s| s.audio)
        {
            self.apply(i, TranscriptPacerEvent::Close, out);
            self.segments.remove(i);
        }
        // The first word is taken as the playback starts.
        let Some(i) = self.playing() else { return };
        self.segments[i].started = true;
        let position = self.segments[i].pacer.as_ref().map(TranscriptPacer::position);
        self.segments[i].playhead = Some(Playhead { position: position.unwrap_or_default(), at });
        if let Some(position) = position {
            self.apply(i, TranscriptPacerEvent::Position(position), out);
        }
    }

    /// The sink finished the playing segment. Its `Heard` is kept only when it had text.
    fn on_finished(&mut self, position: Duration, interrupted: bool) -> Vec<TranscriptSyncEffect> {
        let heard = |e: &TranscriptSyncEffect| matches!(e, TranscriptSyncEffect::ReportHeard(_));
        let mut out = Vec::new();
        let Some(i) = self.playing() else { return out };
        let segment = &self.segments[i];
        let splits = |p: &TranscriptPacer| live(p) && !p.is_cleared();
        if interrupted && !segment.audio_ended && segment.pacer.as_ref().is_some_and(splits) {
            // Still capturing: a sink that cuts and resets its count plays the rest anew.
            let had_text = segment.pacer.as_ref().is_some_and(TranscriptPacer::saw_text);
            self.segments[i].cut_below = true;
            self.apply(i, TranscriptPacerEvent::PlaybackSplit { position }, &mut out);
            out.retain(|e| had_text || !heard(e));
            return out;
        }
        let segment = self.segments.remove(i).expect("playing is an index");
        let Some(pacer) = segment.pacer.filter(live) else { return out };
        // Audio with no words: no transcript, so the speech's own text stands.
        let had_text = pacer.saw_text();
        let (_, effects) =
            pacer.step(TranscriptPacerEvent::PlaybackFinished { position, interrupted });
        out.extend(effects.into_iter().filter(|e| had_text || !heard(e)));
        out
    }

    /// The sink threw its playback count away: a segment whose audio goes on starts the rest
    /// as a playout of its own, and the others it owed end where they stood.
    fn on_reset(&mut self, out: &mut Vec<TranscriptSyncEffect>) {
        for i in 0..self.segments.len() {
            let segment = &self.segments[i];
            let splits = |p: &TranscriptPacer| live(p) && !p.is_cleared();
            if segment.audio && !segment.audio_ended && segment.pacer.as_ref().is_some_and(splits) {
                self.split_playout(i, out);
            } else if segment.audio {
                self.apply(i, TranscriptPacerEvent::Close, out);
            }
        }
        self.segments.retain(|s| !s.audio);
    }

    /// Its audio goes on: the rest is a playout of its own, the words with it.
    fn split_playout(&mut self, i: usize, out: &mut Vec<TranscriptSyncEffect>) {
        let segment = &mut self.segments[i];
        if !std::mem::take(&mut segment.cut_below) {
            let position = segment.pacer.as_ref().map(TranscriptPacer::position);
            let position = position.unwrap_or_default();
            self.apply(i, TranscriptPacerEvent::PlaybackSplit { position }, out);
        }
        let segment = &mut self.segments[i];
        (segment.audio, segment.started, segment.playhead) = (false, false, None);
    }

    /// The oldest segment `open` holds for, else a new one.
    fn next(&mut self, open: impl Fn(&Segment) -> bool) -> usize {
        if let Some(i) = self.segments.iter().position(open) {
            return i;
        }
        let pacer = self.enabled.then(|| TranscriptPacer::new(self.opts.clone()));
        let segment = Segment {
            pacer,
            text_ended: false,
            audio: false,
            key: None,
            audio_ended: false,
            started: false,
            playhead: None,
            cut_below: false,
            cleared: false,
        };
        self.segments.push_back(segment);
        self.segments.len() - 1
    }

    /// A segment with no audio is gone once its pacer and its text are: nothing will report its
    /// playout.
    fn settle(&mut self, i: usize) {
        let segment = &self.segments[i];
        if !segment.audio
            && segment.text_ended
            && segment.pacer.as_ref().is_none_or(TranscriptPacer::is_over)
        {
            self.segments.remove(i);
        }
    }

    /// The oldest segment the sink owes a playout.
    fn playing(&self) -> Option<usize> {
        self.segments.iter().position(|s| s.audio && !s.cut_below)
    }

    /// Closing ends the text output's segment even with none open.
    fn rotate(&mut self, out: &mut Vec<TranscriptSyncEffect>) {
        if self.segments.is_empty() {
            out.push(TranscriptSyncEffect::FlushTranscript);
        }
        for i in 0..self.segments.len() {
            self.apply(i, TranscriptPacerEvent::Close, out);
            let segment = &mut self.segments[i];
            segment.pacer = None;
            segment.text_ended = true;
        }
        self.segments.retain(|s| s.audio && !s.cut_below);
    }

    fn apply(&mut self, i: usize, ev: TranscriptPacerEvent, out: &mut Vec<TranscriptSyncEffect>) {
        let slot = &mut self.segments[i].pacer;
        // An ended segment takes nothing more; its slot still accounts for its playout.
        let Some(pacer) = slot.take_if(|p| !p.is_over()) else { return };
        let (pacer, effects) = pacer.step(ev);
        *slot = Some(pacer);
        out.extend(effects);
    }
}

/// A pacer still owed its playout's end: not over, and with audio or text to account for.
fn live(pacer: &TranscriptPacer) -> bool {
    !pacer.is_over() && !pacer.is_blank()
}

#[cfg(test)]
#[path = "synchronizer_tests.rs"]
mod tests;
