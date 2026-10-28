use crate::transcription::Heard;
use crate::transcription::pacer::{PaceEffect, PaceEvent, Pacer, PacerOptions};
use lk_agents_abi::voice::io::{
    AudioOutput, AudioOutputExt, PlaybackEvent, PlaybackFinished, PlaybackObserver,
    PlaybackTracker, Result, TextOutput,
};
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::{AudioFrame, Clock, Monotone, Spawner, SpawnerExt, TimedString};
use lk_agents_voice::io::AudioSinkProxy;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::{HashMap, VecDeque};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;
use tokio::sync::mpsc;

const PATTERNS: &str = include_str!("liang/patterns.txt");

const EXCEPTIONS: &str = include_str!("liang/exceptions.txt");

/// How long `aclose` waits for the forwarder to hand the last words to the text sink.
pub const SEGMENT_ACLOSE_TIMEOUT: Duration = Duration::from_secs(5);

/// Counts syllable-like pieces used to pace transcript words: [`Liang`] by default.
pub trait Hyphenator: Send + Sync {
    /// `len(hyphenate_word(word))`. Never zero, including for the empty string.
    fn hyphens(&self, word: &str) -> usize;
}

#[derive(Default)]
struct Node {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    children: HashMap<char, Node>,
    /// The pattern's points when a pattern ends at this node (`t[None]`).
    points: Option<Vec<u8>>,
}

struct Table {
    tree: Node,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    exceptions: HashMap<String, Vec<u8>>,
}

impl Table {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn build() -> Table {
        let mut tree = Node::default();
        for pattern in PATTERNS.split_whitespace() {
            // One point before every character and one after the last, as the reference parses.
            let mut points = Vec::new();
            let mut digits = String::new();
            let mut node = &mut tree;
            for c in pattern.chars() {
                if c.is_ascii_digit() {
                    digits.push(c);
                } else {
                    points.push(digits.parse::<u8>().unwrap_or(0));
                    digits.clear();
                    node = node.children.entry(c).or_default();
                }
            }
            points.push(digits.parse::<u8>().unwrap_or(0));
            node.points = Some(points);
        }
        let mut exceptions = HashMap::new();
        for ex in EXCEPTIONS.split_whitespace() {
            // `[0] + [int(h == "-") for h in re.split(r"[a-z]", ex)]`.
            let mut points = vec![0u8];
            let mut between = String::new();
            for c in ex.chars() {
                if c.is_ascii_lowercase() {
                    points.push(u8::from(between == "-"));
                    between.clear();
                } else {
                    between.push(c);
                }
            }
            points.push(u8::from(between == "-"));
            exceptions.insert(ex.replace('-', ""), points);
        }
        Table { tree, exceptions }
    }
}

fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(Table::build)
}

/// `hyphenate_word(word)`: the pieces, broken at the possible hyphenation points.
#[lk_agents_abi::host(export(pid = false))]
#[python(forward(module = "tokenize_basic"))]
pub fn hyphenate_word(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    // "Short words aren't hyphenated." Characters, not bytes.
    if chars.len() <= 4 {
        return vec![word.to_owned()];
    }
    let table = table();
    let lower = word.to_lowercase();
    let points: Vec<u8> = match table.exceptions.get(&lower) {
        Some(p) => p.clone(),
        None => {
            let work: Vec<char> =
                std::iter::once('.').chain(lower.chars()).chain(std::iter::once('.')).collect();
            let mut points = vec![0u8; work.len() + 1];
            for i in 0..work.len() {
                let mut node = &table.tree;
                for &c in &work[i..] {
                    let Some(next) = node.children.get(&c) else { break };
                    node = next;
                    if let Some(p) = &node.points {
                        for (j, pj) in p.iter().enumerate() {
                            if let Some(slot) = points.get_mut(i + j) {
                                *slot = (*slot).max(*pj);
                            }
                        }
                    }
                }
            }
            // "No hyphens in the first two chars or the last two."
            let n = points.len();
            points[1] = 0;
            points[2] = 0;
            points[n - 2] = 0;
            points[n - 3] = 0;
            points
        }
    };
    // `for c, p in zip(word, points[2:])`: the original characters, the folded word's points.
    let mut pieces = vec![String::new()];
    for (c, p) in chars.iter().zip(points.iter().skip(2)) {
        pieces.last_mut().expect("one piece at least").push(*c);
        if p % 2 == 1 {
            pieces.push(String::new());
        }
    }
    pieces
}

/// The pacer's hyphenator: Liang's syllable count, `len(hyphenate_word(word))`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Liang;

impl Hyphenator for Liang {
    fn hyphens(&self, word: &str) -> usize {
        hyphenate_word(word).len().max(1)
    }
}

/// What the two legs and the sink below tell the synchroniser.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncInput {
    Text {
        raw: String,
        start_time: Option<f64>,
        end_time: Option<f64>,
    },
    /// The text leg's flush: the segment's text is complete.
    TextEnd,
    /// A frame for the sink below.
    Audio {
        secs: f64,
    },
    /// The audio leg's flush: the segment's audio is complete, possibly with none.
    AudioEnd,
    /// `clear_buffer`: the newest segment, if none of its audio reached the sink, is cut.
    Clear,
    Started,
    Progressed(Duration),
    Paused,
    Finished {
        position: Duration,
        interrupted: bool,
    },
    /// The sink threw its playback count away: the segments it owed are gone where they stood.
    Reset,
    Enabled(bool),
    /// End every segment where it stands; the next input starts a fresh one.
    Rotate,
}

/// One segment as the synchroniser sees it.
struct Segment {
    /// `None` once closed while its audio is still the sink's (a rotation, synchronisation
    /// switched off), or opened while synchronisation was off: its playback moves no pacer.
    pacer: Option<Pacer>,
    text_ended: bool,
    /// A frame reached the sink: the sink owes this segment's playback finish.
    audio: bool,
    audio_ended: bool,
    /// The sink reported this segment's playback started.
    started: bool,
    /// The sink cut this segment's playout while its audio was still coming. It owes no finish
    /// until a reset starts the rest of its audio as a playout of its own.
    cut_below: bool,
}

/// `_SegmentSynchronizerImpl`s in order. Each leg is an ordered stream of segments delimited by
/// its flushes: a segment's text is the next text not yet ended, its audio the next audio, so a
/// segment the producer ended with no audio is one by construction even when its words arrive
/// later (through a `transcription_node`). The sink's playback moves the oldest segment it owes.
pub struct SyncSegments {
    opts: PacerOptions,
    enabled: bool,
    segments: VecDeque<Segment>,
}

impl SyncSegments {
    pub fn new(opts: PacerOptions) -> SyncSegments {
        SyncSegments { opts, enabled: true, segments: VecDeque::new() }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// One input in, the words to show out. A `Heard` is kept only for a `Finished` segment that
    /// had text: its synchronised transcript.
    pub fn step(&mut self, input: SyncInput) -> Vec<PaceEffect> {
        let mut out = Vec::new();
        match input {
            SyncInput::Text { raw, start_time, end_time } => {
                let i = self.next(|s| !s.text_ended);
                if self.segments[i].pacer.is_none() {
                    // A segment closed mid-write: its words have no pace to keep.
                    out.push(PaceEffect::Forward { word: raw, start_time: 0.0, end_time: 0.0 });
                } else {
                    self.apply(i, PaceEvent::Text { raw, start_time, end_time }, &mut out);
                }
            }
            SyncInput::TextEnd => {
                if let Some(i) = self.segments.iter().position(|s| !s.text_ended) {
                    self.segments[i].text_ended = true;
                    self.apply(i, PaceEvent::TextEnd, &mut out);
                    self.settle(i);
                }
            }
            SyncInput::Audio { secs } => {
                let i = self.next(|s| !s.audio_ended);
                self.segments[i].audio = true;
                self.apply(i, PaceEvent::Audio { secs }, &mut out);
            }
            SyncInput::AudioEnd => {
                // Ended before any of its words arrived: a segment with no audio, unless its
                // words will pass unpaced.
                if !self.enabled && self.segments.iter().all(|s| s.audio_ended) {
                    return out;
                }
                let i = self.next(|s| !s.audio_ended);
                self.segments[i].audio_ended = true;
                self.apply(i, PaceEvent::AudioEnd, &mut out);
                if self.segments[i].cut_below {
                    // Cut below and no reset came: the sink owes it no finish.
                    self.apply(i, PaceEvent::Close, &mut out);
                    self.segments.remove(i);
                } else {
                    self.settle(i);
                }
            }
            SyncInput::Clear => {
                if self.segments.back().is_some_and(|s| !s.audio) {
                    let i = self.segments.len() - 1;
                    self.apply(i, PaceEvent::Close, &mut out);
                    self.segments.pop_back();
                }
            }
            SyncInput::Started => {
                // A sink plays its segments in order: one starting while an older one it had
                // started is still owed means that one's finish was lost. It ends where it stood.
                if let Some(i) = self.playing()
                    && self.segments[i].started
                    && self.segments.iter().skip(i + 1).any(|s| s.audio)
                {
                    self.apply(i, PaceEvent::Close, &mut out);
                    self.segments.remove(i);
                }
                // The reference's pacing loop takes its first word as the playback starts.
                if let Some(i) = self.playing() {
                    self.segments[i].started = true;
                    if let Some(at) = self.segments[i].pacer.as_ref().map(Pacer::position) {
                        self.apply(i, PaceEvent::Position(at), &mut out);
                    }
                }
            }
            SyncInput::Progressed(at) => {
                if let Some(i) = self.playing() {
                    self.apply(i, PaceEvent::Position(at), &mut out);
                }
            }
            SyncInput::Paused => {
                if let Some(i) = self.playing() {
                    self.apply(i, PaceEvent::Paused, &mut out);
                }
            }
            SyncInput::Finished { position, interrupted } => {
                let Some(i) = self.playing() else { return out };
                let segment = &self.segments[i];
                if interrupted && !segment.audio_ended && segment.pacer.as_ref().is_some_and(live) {
                    // Still capturing: a sink that cuts and resets its count plays the rest anew.
                    let had_text = segment.pacer.as_ref().is_some_and(Pacer::saw_text);
                    self.segments[i].cut_below = true;
                    self.apply(i, PaceEvent::PlaybackSplit { position }, &mut out);
                    out.retain(|e| had_text || !matches!(e, PaceEffect::Heard(_)));
                    return out;
                }
                let segment = self.segments.remove(i).expect("playing is an index");
                if let Some(pacer) = segment.pacer.filter(live) {
                    // Audio with no words: no transcript, so the speech's own text stands.
                    let had_text = pacer.saw_text();
                    let (_, effects) =
                        pacer.step(PaceEvent::PlaybackFinished { position, interrupted });
                    let heard = |e: &PaceEffect| matches!(e, PaceEffect::Heard(_));
                    out.extend(effects.into_iter().filter(|e| had_text || !heard(e)));
                    return out;
                }
            }
            SyncInput::Reset => {
                for i in 0..self.segments.len() {
                    let segment = &mut self.segments[i];
                    if segment.audio
                        && !segment.audio_ended
                        && segment.pacer.as_ref().is_some_and(live)
                    {
                        // Its audio goes on: the rest is a playout of its own, the words with it.
                        if !std::mem::take(&mut segment.cut_below) {
                            let position = segment.pacer.as_ref().map(Pacer::position);
                            let position = position.unwrap_or_default();
                            self.apply(i, PaceEvent::PlaybackSplit { position }, &mut out);
                        }
                        let segment = &mut self.segments[i];
                        (segment.audio, segment.started) = (false, false);
                    } else if segment.audio {
                        self.apply(i, PaceEvent::Close, &mut out);
                    }
                }
                self.segments.retain(|s| !s.audio);
            }
            SyncInput::Enabled(enabled) => {
                if self.enabled != enabled {
                    self.enabled = enabled;
                    self.rotate(&mut out);
                }
            }
            SyncInput::Rotate => self.rotate(&mut out),
        }
        out.retain(|e| !matches!(e, PaceEffect::Heard(_)));
        out
    }

    /// The oldest segment `open` holds for, else a new one.
    fn next(&mut self, open: impl Fn(&Segment) -> bool) -> usize {
        if let Some(i) = self.segments.iter().position(open) {
            return i;
        }
        let pacer = self.enabled.then(|| Pacer::new(self.opts.clone()));
        let segment = Segment {
            pacer,
            text_ended: false,
            audio: false,
            audio_ended: false,
            started: false,
            cut_below: false,
        };
        self.segments.push_back(segment);
        self.segments.len() - 1
    }

    /// A segment with no audio is gone once its pacer and its text are: nothing will report its
    /// playout.
    fn settle(&mut self, i: usize) {
        let segment = &self.segments[i];
        if !segment.audio && segment.text_ended && segment.pacer.as_ref().is_none_or(Pacer::is_over)
        {
            self.segments.remove(i);
        }
    }

    /// The oldest segment the sink owes a playout.
    fn playing(&self) -> Option<usize> {
        self.segments.iter().position(|s| s.audio && !s.cut_below)
    }

    /// Closing ends the text output's segment even with none open, as closing the reference's
    /// idle `_SegmentSynchronizerImpl` does.
    fn rotate(&mut self, out: &mut Vec<PaceEffect>) {
        if self.segments.is_empty() {
            out.push(PaceEffect::Flush);
        }
        for i in 0..self.segments.len() {
            self.apply(i, PaceEvent::Close, out);
            let segment = &mut self.segments[i];
            segment.pacer = None;
            segment.text_ended = true;
        }
        self.segments.retain(|s| s.audio && !s.cut_below);
    }

    fn apply(&mut self, i: usize, ev: PaceEvent, out: &mut Vec<PaceEffect>) {
        let slot = &mut self.segments[i].pacer;
        // An ended segment takes nothing more; its slot still accounts for its playout.
        let Some(pacer) = slot.take_if(|p| !p.is_over()) else { return };
        let (pacer, effects) = pacer.step(ev);
        *slot = Some(pacer);
        out.extend(effects);
    }
}

/// A pacer still owed its playout's end: not over, and with audio or text to account for.
fn live(pacer: &Pacer) -> bool {
    !pacer.is_over() && !pacer.is_blank()
}

enum Out {
    Word(TimedString),
    Flush,
}

/// The segments and the gate around them, under one lock.
struct SyncState {
    segments: SyncSegments,
    audio_attached: bool,
    text_attached: bool,
    closed: bool,
    /// `_SyncedTextOutput._capturing`: whether a flush has anything to end. With nothing
    /// captured, ending the text would pace later words against the next segment's audio.
    capturing_text: bool,
}

/// What both chain nodes and the facade share; nothing holds the facade, so no reference cycle.
struct SyncCore {
    clock: Arc<dyn Clock>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    state: Mutex<SyncState>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// `None` once closed: dropping the sender is what ends the forwarder.
    tx: Mutex<Option<mpsc::UnboundedSender<Out>>>,
    /// Shared with the forwarder task, which must not keep the core alive (or it never ends).
    forwarder_done: Arc<Monotone<bool>>,
    /// The audio node's tracker; released words are raised on it as `transcript_shown`.
    tracker: Arc<PlaybackTracker>,
}

impl SyncCore {
    fn enabled(&self) -> bool {
        self.state.lock().unwrap().segments.enabled()
    }

    /// Step the segments and perform what they say; the heard transcript of a finished segment.
    fn step(&self, input: SyncInput) -> Option<Heard> {
        let effects = self.state.lock().unwrap().segments.step(input);
        let mut heard = None;
        for effect in effects {
            match effect {
                PaceEffect::Forward { word, start_time, end_time } => {
                    let shown = TimedString {
                        start_time: Some(start_time),
                        end_time: Some(end_time),
                        ..TimedString::new(word)
                    };
                    self.tracker.transcript_shown(&shown);
                    // The text output gets where the word's audio ends, as the reference's does.
                    self.send(Out::Word(TimedString { start_time: None, ..shown }));
                }
                PaceEffect::Flush => self.send(Out::Flush),
                PaceEffect::Heard(h) => heard = Some(h),
                PaceEffect::Warn(w) => log::warn!(target: "livekit.agents", "{}", w.message()),
            }
        }
        heard
    }

    fn send(&self, out: Out) {
        if let Some(tx) = self.tx.lock().unwrap().as_ref() {
            let _ = tx.send(out);
        }
    }

    /// `_on_attachment_changed`. `None` leaves a leg's attachment where it is.
    fn on_attachment_changed(&self, audio: Option<bool>, text: Option<bool>) {
        let enabled = {
            let mut state = self.state.lock().unwrap();
            state.audio_attached = audio.unwrap_or(state.audio_attached);
            state.text_attached = text.unwrap_or(state.text_attached);
            state.audio_attached && state.text_attached
        };
        self.step(SyncInput::Enabled(enabled));
    }

    /// The forwarder: the only task and the only place the text sink is awaited.
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    async fn forward(next: Option<Arc<dyn TextOutput>>, mut rx: mpsc::UnboundedReceiver<Out>) {
        while let Some(out) = rx.recv().await {
            let Some(next) = &next else { continue };
            match out {
                // A failure is the sink's. Dropping it keeps the rest of the transcript flowing.
                Out::Word(t) => {
                    let _ = next.capture_timed(&t).await;
                }
                Out::Flush => next.flush(),
            }
        }
    }
}

/// `_SyncedAudioOutput`.
pub struct SyncedAudioOutput {
    core: Arc<SyncCore>,
    tracker: Arc<PlaybackTracker>,
    next: Arc<dyn AudioOutput>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    observer: Mutex<Option<Arc<dyn PlaybackObserver>>>,
}

/// [`Weak`] so the tracker doesn't keep the chain alive.
struct AudioEvents(Weak<SyncedAudioOutput>);

impl PlaybackObserver for AudioEvents {
    fn on_playback_event(&self, ev: &PlaybackEvent) {
        if let Some(node) = self.0.upgrade() {
            node.handle_playback(ev);
        }
    }

    fn on_playback_reset(&self) {
        if let Some(node) = self.0.upgrade() {
            node.core.step(SyncInput::Reset);
            node.tracker.on_playback_reset();
        }
    }
}

impl SyncedAudioOutput {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn new(core: Arc<SyncCore>, next: Arc<dyn AudioOutput>) -> Arc<Self> {
        let node = Arc::new(SyncedAudioOutput {
            tracker: Arc::clone(&core.tracker),
            core,
            next: next.clone(),
            observer: Mutex::new(None),
        });
        let observer: Arc<dyn PlaybackObserver> = Arc::new(AudioEvents(Arc::downgrade(&node)));
        next.tracker().observe(observer.clone());
        *node.observer.lock().unwrap() = Some(observer);
        node
    }

    /// Synchronous: a wrapper's counter must move before the leaf's emit returns, or
    /// `wait_for_playout` races the event meant to release it.
    fn handle_playback(&self, ev: &PlaybackEvent) {
        match ev {
            PlaybackEvent::Started(e) => {
                self.tracker.playback_started(e.created_at);
                self.core.step(SyncInput::Started);
            }
            PlaybackEvent::Progressed(e) => {
                self.tracker.playback_progressed(e.started_at, e.offset, e.duration);
                self.core.step(SyncInput::Progressed(e.offset + e.duration));
            }
            PlaybackEvent::Finished(e) => {
                let position = e.playback_position;
                let heard =
                    self.core.step(SyncInput::Finished { position, interrupted: e.interrupted });
                let synchronized_transcript =
                    heard.map(|h| h.text).or(e.synchronized_transcript.clone());
                self.tracker
                    .playback_finished(PlaybackFinished { synchronized_transcript, ..e.clone() });
            }
        }
    }
}

impl AudioOutput for SyncedAudioOutput {
    fn label(&self) -> &str {
        "TranscriptSynchronizer"
    }

    fn tracker(&self) -> &Arc<PlaybackTracker> {
        &self.tracker
    }

    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: true }
    }

    fn sample_rate(&self) -> Option<u32> {
        self.next.sample_rate()
    }

    fn next_in_chain(&self) -> Option<Arc<dyn AudioOutput>> {
        Some(self.next.clone())
    }

    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            // Counted before it passes: the sink below may report its playback from inside the
            // capture, and that playback is this segment's.
            let secs = lk_agents_rtc::frame_duration(frame).as_secs_f64();
            self.core.step(SyncInput::Audio { secs });
            self.next.capture_frame(frame).await
        })
    }

    fn flush_segment(&self) {
        self.next.flush();
        self.core.step(SyncInput::AudioEnd);
    }

    fn clear_buffer(&self) {
        self.next.clear_buffer();
        self.core.step(SyncInput::Clear);
    }

    fn on_attached(&self) {
        self.next.on_attached();
        self.core.on_attachment_changed(Some(true), None);
    }

    fn on_detached(&self) {
        self.next.on_detached();
        self.core.on_attachment_changed(Some(false), None);
    }

    fn pause(&self) {
        self.next.pause();
        self.core.step(SyncInput::Paused);
    }

    fn resume(&self) {
        self.next.resume();
    }
}

/// `_SyncedTextOutput`.
pub struct SyncedTextOutput {
    core: Arc<SyncCore>,
    next: Option<Arc<dyn TextOutput>>,
}

impl SyncedTextOutput {
    /// `aligned`: whether the caller used the timed API (see `Pacer::push_text`).
    async fn capture(&self, text: &TimedString, aligned: bool) -> Result<()> {
        if !self.core.enabled() {
            if let Some(next) = &self.next {
                if aligned {
                    next.capture_timed(text).await?;
                } else {
                    next.capture_text(&text.text).await?;
                }
            }
            return Ok(());
        }
        self.core.state.lock().unwrap().capturing_text = true;
        let (start_time, end_time) =
            if aligned { (text.start_time, text.end_time) } else { (None, None) };
        let raw = text.text.clone();
        self.core.step(SyncInput::Text { raw, start_time, end_time });
        Ok(())
    }
}

impl TextOutput for SyncedTextOutput {
    fn label(&self) -> &str {
        "TranscriptSynchronizer"
    }

    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        self.next.clone()
    }

    fn capture_timed<'a>(
        &'a self,
        text: &'a TimedString,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.capture(text, true).await })
    }

    fn capture_text<'a>(&'a self, text: &'a str) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move { self.capture(&TimedString::new(text), false).await })
    }

    fn flush(&self) {
        if !self.core.enabled() {
            if let Some(next) = &self.next {
                next.flush();
            }
            return;
        }
        if std::mem::take(&mut self.core.state.lock().unwrap().capturing_text) {
            self.core.step(SyncInput::TextEnd);
        }
    }

    fn on_attached(&self) {
        if let Some(next) = &self.next {
            next.on_attached();
        }
        self.core.on_attachment_changed(None, Some(true));
    }

    fn on_detached(&self) {
        if let Some(next) = &self.next {
            next.on_detached();
        }
        self.core.on_attachment_changed(None, Some(false));
    }
}

/// Everything `TranscriptSynchronizer.__init__` takes, plus the two ports Rust needs explicitly.
pub struct SyncConfig {
    /// The audio sink below; a bare leaf is wrapped in an [`AudioSinkProxy`].
    pub audio_next: Arc<dyn AudioOutput>,
    /// The transcript sink below this node. `None` for the console path, which prints locally.
    pub text_next: Option<Arc<dyn TextOutput>>,
    pub speed: f64,
    pub hyphenator: Arc<dyn crate::transcription::synchronizer::Hyphenator>,
    pub clock: Arc<dyn Clock>,
    pub spawner: Arc<dyn Spawner>,
}

pub struct TranscriptSynchronizer {
    core: Arc<SyncCore>,
    audio: Arc<SyncedAudioOutput>,
    text: Arc<SyncedTextOutput>,
    /// The clock's wall stamp, for a host listening to the audio node's playback.
    anchor: lk_agents_abi::WallAnchor,
}

/// `_core.TranscriptSynchronizer`: the node every native session builds, handed to a host that
/// assembles a synchroniser by hand (the console, a custom chain), so pacing is one implementation
/// on one clock.
#[lk_agents_abi::host(object(cell = "arc"))]
impl TranscriptSynchronizer {
    pub fn new(config: SyncConfig) -> Arc<Self> {
        Arc::new(Self::build(config))
    }

    /// `next_in_chain_audio` is the host `io.AudioOutput` the audio passes through to (its
    /// playback events pace the text); `next_in_chain_text` the host `io.TextOutput` the paced
    /// words go to, or `None`.
    #[host(new(kw_only = "speed", defaults(speed = 1.0)))]
    pub fn for_host(
        next_in_chain_audio: Arc<dyn AudioOutput>,
        next_in_chain_text: Option<Arc<dyn TextOutput>>,
        speed: f64,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Self {
        let tracker = Arc::clone(next_in_chain_audio.tracker());
        let sync = Self::build(SyncConfig {
            audio_next: next_in_chain_audio,
            text_next: next_in_chain_text,
            speed,
            hyphenator: Arc::new(Liang),
            clock: Arc::new(lk_agents_abi::RealClock::new()),
            spawner,
        });
        // The host sink's playback, stamped on the clock the pacer runs on.
        tracker.anchor_to(sync.anchor);
        sync
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn build(config: SyncConfig) -> Self {
        let SyncConfig { audio_next, text_next, speed, hyphenator, clock, spawner } = config;
        let anchor = lk_agents_abi::WallAnchor::of(&*clock);

        // `io.AudioOutput.__init__`'s auto-wrap, so an avatar hand-off can replace the leaf later.
        let audio_next = if audio_next.next_in_chain().is_none() {
            AudioSinkProxy::new(audio_next) as Arc<dyn AudioOutput>
        } else {
            audio_next
        };

        let opts = PacerOptions { speed, hyphenator };
        let (tx, rx) = mpsc::unbounded_channel();
        let core = Arc::new(SyncCore {
            state: Mutex::new(SyncState {
                segments: SyncSegments::new(opts),
                audio_attached: true,
                text_attached: true,
                closed: false,
                capturing_text: false,
            }),
            clock,
            tx: Mutex::new(Some(tx)),
            forwarder_done: Arc::new(Monotone::new(false)),
            tracker: Arc::new(PlaybackTracker::new()),
        });
        // The task holds only the done flag, so it never keeps the core alive.
        let forwarder_done = Arc::clone(&core.forwarder_done);
        let sink = text_next.clone();
        spawner.spawn_unboxed(async move {
            SyncCore::forward(sink, rx).await;
            forwarder_done.set(true);
        });

        let audio = SyncedAudioOutput::new(core.clone(), audio_next);
        let text = Arc::new(SyncedTextOutput { core: core.clone(), next: text_next });
        TranscriptSynchronizer { core, audio, text, anchor }
    }

    /// The node to install as `session.output.transcription`, as a host object.
    #[host(getter(name = "text_output"))]
    pub fn text_leg(&self) -> crate::io::CoreTextOutput {
        crate::io::CoreTextOutput::new(self.text_output())
    }

    /// The node to install as `session.output.audio`.
    pub fn audio_output(&self) -> Arc<dyn AudioOutput> {
        self.audio.clone()
    }

    /// The node to install as `session.output.transcription`.
    pub fn text_output(&self) -> Arc<dyn TextOutput> {
        self.text.clone()
    }

    #[host(getter)]
    pub fn enabled(&self) -> bool {
        self.core.enabled()
    }

    #[host(method)]
    pub fn set_enabled(&self, enabled: bool) {
        self.core.step(SyncInput::Enabled(enabled));
    }

    /// End the live segment now and start a fresh one.
    #[host(method)]
    pub fn rotate_segment(&self) {
        if !self.core.state.lock().unwrap().closed {
            self.core.step(SyncInput::Rotate);
        }
    }

    /// Close: end the live segment and give the forwarder [`SEGMENT_ACLOSE_TIMEOUT`] to drain.
    /// `RoomIO.aclose` must close this before the transcription outputs it feeds.
    #[host(method)]
    pub async fn aclose(&self) {
        if std::mem::replace(&mut self.core.state.lock().unwrap().closed, true) {
            return;
        }
        self.core.step(SyncInput::Rotate);
        // Dropping the sender is what ends the forwarder's `recv`, after the words already queued.
        self.core.tx.lock().unwrap().take();
        let _ = lk_agents_abi::timeout(
            &*self.core.clock,
            SEGMENT_ACLOSE_TIMEOUT,
            self.core.forwarder_done.wait_for(|set| *set),
        )
        .await;
    }
}

#[cfg(test)]
#[path = "synchronizer_tests.rs"]
mod tests;
