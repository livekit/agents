use crate::transcription::Heard;
use async_trait::async_trait;
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_abi::tts::markup::TranscriptMarkupStripper;
use lk_agents_abi::tokenize::{
    TokenBuffer, TokenBufferOptions, WordTokenizerOptions, Words, split_words,
};
use lk_agents_abi::{AudioFrame, Clock, Monotone, Spawner, SpawnerExt, TimedString};
use lk_agents_voice::io::{
    AudioOutput, AudioOutputExt, AudioSinkProxy, PlaybackEvent, PlaybackFinished, PlaybackObserver,
    PlaybackTracker, Result, TextOutput,
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;
use tokio::sync::mpsc;

/// The synchroniser's tokenizer settings, from `TranscriptSynchronizer.__init__`: tokens must
/// reassemble the input, punctuation is paced, and CJK/Thai split per character.
pub const SYNC_TOKENIZER: WordTokenizerOptions =
    WordTokenizerOptions { ignore_punctuation: false, split_character: true, retain_format: true };

/// Hyphens (syllables) per second at `speed = 1.0`, before a segment's own rate is known.
pub const STANDARD_SPEECH_RATE: f64 = 3.83;

const PATTERNS: &str = include_str!("liang/patterns.txt");

const EXCEPTIONS: &str = include_str!("liang/exceptions.txt");

/// How long `aclose` waits for the forwarder to hand the last words to the text sink.
pub const SEGMENT_ACLOSE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum PaceEvent {
    /// One text delta, raw. `start_time` is the TTS's alignment for it, when the TTS aligns.
    Text { raw: String, start_time: Option<f64> },
    TextEnd,
    Audio { secs: f64 },
    AudioEnd,
    Position(Duration),
    PlaybackFinished { position: Duration, interrupted: bool },
    Close,
}

#[must_use = "an effect not performed is a word the user never sees"]
#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum PaceEffect {
    /// One raw token for the text output, stamped with where its audio ends.
    Forward { word: String, end_time: f64 },
    Flush,
    /// Exactly once per segment.
    Heard(Heard),
    /// An event the segment was not in a state to take. Nothing changed.
    Warn(PaceWarning),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum PaceWarning {
    /// Text after `TextEnd`: a missing rotation upstream.
    TextAfterEnd,
    /// Audio after `AudioEnd`: a missing rotation upstream.
    AudioAfterEnd,
    /// `PlaybackFinished` before both inputs ended. The segment finishes anyway on what it has.
    PlaybackFinishedTooEarly,
    AfterFinished,
    AfterClose,
}

impl PaceWarning {
    /// The key of the facade's `_PACE_WARNINGS` table (checked by `test_transcript_warning_seam.py`).
    pub fn name(self) -> &'static str {
        match self {
            PaceWarning::TextAfterEnd => "text_after_end",
            PaceWarning::AudioAfterEnd => "audio_after_end",
            PaceWarning::PlaybackFinishedTooEarly => "playback_finished_too_early",
            PaceWarning::AfterFinished => "after_finished",
            PaceWarning::AfterClose => "after_close",
        }
    }
}

/// Where a segment is. The enum is the whole state; the accumulators beside it are data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaceState {
    Streaming,
    TextEnded,
    AudioEnded,
    /// Both inputs ended; waiting for the playout to finish.
    InputsEnded,
    /// `Heard` was produced. Only `Close` is left.
    Finished,
    Closed,
}

#[derive(Debug, Clone, PartialEq)]
struct Planned {
    word: String,
    /// The token's visible text, from the segment's one stateful stripper.
    visible: String,
    hyphens: usize,
    offset: usize,
    /// The TTS's `start_time` for the delta this word came from, or the segment start.
    base: Option<f64>,
}

impl Planned {
    fn due(&self, speed: f64) -> f64 {
        self.base.unwrap_or(0.0) + self.offset as f64 / speed
    }

    fn end(&self, speed: f64) -> f64 {
        self.due(speed) + self.hyphens as f64 / speed
    }
}

pub struct Pacer {
    state: PaceState,
    opts: PacerOptions,
    speed: f64,
    /// `basic.WordTokenizer(...).stream()`: a token is emitted once a following one proves it
    /// complete.
    words: TokenBuffer<Words>,
    stripper: TranscriptMarkupStripper,
    queue: VecDeque<Planned>,
    /// Hyphens planned since `base` moved (or since the segment start).
    offset: usize,
    base: Option<f64>,
    planned_hyphens: usize,
    forwarded: String,
    audio_secs: f64,
    frames: u64,
    position: Duration,
    saw_text: bool,
}

impl Pacer {
    pub fn new(opts: PacerOptions) -> Pacer {
        let speed = STANDARD_SPEECH_RATE * opts.speed;
        Pacer {
            state: PaceState::Streaming,
            opts,
            speed,
            words: TokenBuffer::new(Words(SYNC_TOKENIZER), TokenBufferOptions::WORDS),
            stripper: TranscriptMarkupStripper::new(),
            queue: VecDeque::new(),
            offset: 0,
            base: None,
            planned_hyphens: 0,
            forwarded: String::new(),
            audio_secs: 0.0,
            frames: 0,
            position: Duration::ZERO,
            saw_text: false,
        }
    }

    pub fn is_blank(&self) -> bool {
        self.frames == 0 && !self.saw_text
    }

    pub fn saw_text(&self) -> bool {
        self.saw_text
    }

    pub fn text_ended(&self) -> bool {
        matches!(self.state, PaceState::TextEnded | PaceState::InputsEnded)
    }

    pub fn audio_ended(&self) -> bool {
        matches!(self.state, PaceState::AudioEnded | PaceState::InputsEnded)
    }

    /// `Heard` was produced, or the segment was closed: it takes nothing more but `Close`.
    pub fn is_over(&self) -> bool {
        matches!(self.state, PaceState::Finished | PaceState::Closed)
    }

    /// One event in, its effects out. The state × event table:
    ///
    /// | state \ event | Text | TextEnd | Audio | AudioEnd | Position | PlaybackFinished | Close |
    /// |---|---|---|---|---|---|---|---|
    /// | Streaming | plan | → TextEnded | count | → AudioEnded¹ | release | finish (too early) | cut |
    /// | TextEnded | warn | warn | count | → InputsEnded¹ | release | finish (too early) | cut |
    /// | AudioEnded | plan | → InputsEnded¹ | warn | warn | release | finish (too early) | cut |
    /// | InputsEnded | warn | warn | warn | warn | release | finish | cut |
    /// | Finished | warn | warn | warn | warn | warn | warn | → Closed |
    /// | Closed | warn | warn | warn | warn | warn | warn | warn |
    ///
    /// ¹ re-estimates the speed; with no audio at all the segment is closed, nothing shown (as
    /// the `Streaming` row's `AudioEnd` with no audio is: see `Pacer::end_audio`).
    #[must_use = "an effect not performed is a word the user never sees"]
    pub fn step(mut self, ev: PaceEvent) -> (Pacer, Vec<PaceEffect>) {
        use PaceEvent as E;
        use PaceState as S;
        let mut out = Vec::new();
        let warn = |out: &mut Vec<PaceEffect>, w| out.push(PaceEffect::Warn(w));
        match (self.state, ev) {
            (S::Closed, _) => warn(&mut out, PaceWarning::AfterClose),
            (S::Finished, E::Close) => self.state = S::Closed,
            (S::Finished, _) => warn(&mut out, PaceWarning::AfterFinished),
            (_, E::Position(p)) => self.release(p, &mut out),
            (_, E::Close) => self.cut(&mut out),
            (state, E::PlaybackFinished { position, interrupted }) => {
                if state != S::InputsEnded {
                    warn(&mut out, PaceWarning::PlaybackFinishedTooEarly);
                }
                self.finish(position, interrupted, &mut out);
            }
            (S::Streaming | S::AudioEnded, E::Text { raw, start_time }) => {
                self.plan(&raw, start_time)
            }
            (S::Streaming, E::TextEnd) => {
                self.end_text();
                self.state = S::TextEnded;
            }
            (S::AudioEnded, E::TextEnd) => {
                self.end_text();
                self.inputs_ended(&mut out);
            }
            (_, E::Text { .. } | E::TextEnd) => warn(&mut out, PaceWarning::TextAfterEnd),
            (S::Streaming | S::TextEnded, E::Audio { secs }) => self.count_audio(secs),
            (S::Streaming, E::AudioEnd) => self.end_audio(&mut out),
            (S::TextEnded, E::AudioEnd) => self.inputs_ended(&mut out),
            (_, E::Audio { .. } | E::AudioEnd) => warn(&mut out, PaceWarning::AudioAfterEnd),
        }
        (self, out)
    }

    fn plan(&mut self, raw: &str, start_time: Option<f64>) {
        self.saw_text = true;
        if let Some(t) = start_time {
            // An aligned delta is a word boundary: the held word belongs to the previous onset.
            let held = self.words.flush();
            self.plan_tokens(held);
            self.base = Some(t);
            self.offset = 0;
        }
        let tokens = self.words.push(raw);
        self.plan_tokens(tokens);
    }

    fn plan_tokens(&mut self, tokens: Vec<String>) {
        for word in tokens {
            let visible = self.stripper.push(&word);
            self.plan_word(word, visible);
        }
    }

    fn plan_word(&mut self, word: String, visible: String) {
        let hyphens = self.hyphens(&visible);
        self.queue.push_back(Planned { word, visible, hyphens, offset: self.offset, base: self.base });
        self.offset += hyphens;
        self.planned_hyphens += hyphens;
    }

    /// Whitespace and markup cost nothing; everything else costs its hyphens.
    fn hyphens(&self, clean: &str) -> usize {
        if clean.trim().is_empty() {
            return 0;
        }
        split_words(clean, SYNC_TOKENIZER)
            .iter()
            .map(|w| self.opts.hyphenator.hyphens(&w.text))
            .sum()
    }

    /// The text ended: the held partial word is a whole word now, and so is a held tag.
    fn end_text(&mut self) {
        let tail = self.words.flush();
        self.plan_tokens(tail);
        let held = self.stripper.flush();
        if !held.is_empty() {
            // A fragment held as a possible tag that never closed: plain text after all.
            self.plan_word(String::new(), held);
        }
    }

    fn count_audio(&mut self, secs: f64) {
        if secs.is_finite() && secs > 0.0 {
            self.audio_secs += secs;
        }
        self.frames += 1;
    }

    /// With no audio at all the segment is closed and nothing shown (the reference's `flush()`
    /// rotates); only a cut before the first frame does this.
    fn end_audio(&mut self, out: &mut Vec<PaceEffect>) {
        if self.frames == 0 {
            self.cut(out);
            return;
        }
        self.state = PaceState::AudioEnded;
    }

    /// Both inputs ended: the segment's own rate is known.
    fn inputs_ended(&mut self, out: &mut Vec<PaceEffect>) {
        if self.frames == 0 {
            self.cut(out);
            return;
        }
        self.state = PaceState::InputsEnded;
        // A markup- or emoji-only segment would otherwise set the speed to zero.
        if self.planned_hyphens > 0 && self.audio_secs > 0.0 {
            self.speed = self.planned_hyphens as f64 / self.audio_secs;
        }
    }

    fn release(&mut self, position: Duration, out: &mut Vec<PaceEffect>) {
        self.position = position;
        let now = position.as_secs_f64();
        while let Some(front) = self.queue.front() {
            if front.due(self.speed) > now + 1e-9 {
                break;
            }
            let word = self.queue.pop_front().expect("peeked");
            let end_time = word.end(self.speed);
            self.forward(word, end_time, out);
        }
    }

    fn forward(&mut self, word: Planned, end_time: f64, out: &mut Vec<PaceEffect>) {
        self.forwarded.push_str(&word.visible);
        if !word.word.is_empty() {
            out.push(PaceEffect::Forward { word: word.word, end_time });
        }
    }

    /// The playout ended. Complete: everything left is heard, at once. Cut: what was forwarded.
    fn finish(&mut self, position: Duration, interrupted: bool, out: &mut Vec<PaceEffect>) {
        self.position = position;
        if interrupted {
            self.queue.clear();
        } else {
            self.end_text();
            let end = position.as_secs_f64();
            while let Some(word) = self.queue.pop_front() {
                // Stamped no later than the playout's own end: the audio is over.
                let end_time = word.end(self.speed).min(end.max(word.due(self.speed)));
                self.forward(word, end_time, out);
            }
        }
        out.push(PaceEffect::Flush);
        let heard = if interrupted {
            Heard::cut(self.forwarded.clone(), position, self.frames)
        } else {
            Heard::complete(self.forwarded.clone(), position, self.frames)
        };
        out.push(PaceEffect::Heard(heard));
        self.state = PaceState::Finished;
    }

    /// Closed before the playout finished: the words shown are the words heard.
    fn cut(&mut self, out: &mut Vec<PaceEffect>) {
        self.queue.clear();
        out.push(PaceEffect::Flush);
        out.push(PaceEffect::Heard(Heard::cut(self.forwarded.clone(), self.position, self.frames)));
        self.state = PaceState::Closed;
    }
}

/// Counts syllable-like pieces used to pace transcript words. The default is [`Liang`]; a caller's
/// own `hyphenate_word=` is this trait through `synchronizer.py`'s `_Hyphenator` (a GIL hop per
/// word).
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.transcription.synchronizer._Hyphenator"))]
pub trait Hyphenator: Send + Sync {
    /// `len(hyphenate_word(word))`. Never zero, including for the empty string.
    fn hyphens(&self, word: &str) -> usize;
}

/// `_TextSyncOptions`, minus the word tokenizer (pinned -- see [`SYNC_TOKENIZER`]).
#[derive(Clone)]
pub struct PacerOptions {
    /// The caller's `transcription_speed_factor`. Multiplies [`STANDARD_SPEECH_RATE`].
    pub speed: f64,
    pub hyphenator: Arc<dyn Hyphenator>,
}

impl std::fmt::Debug for PacerOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PacerOptions").field("speed", &self.speed).finish_non_exhaustive()
    }
}

impl Default for PacerOptions {
    fn default() -> Self {
        PacerOptions { speed: 1.0, hyphenator: Arc::new(Liang) }
    }
}

#[derive(Default)]
struct Node {
    children: HashMap<char, Node>,
    /// The pattern's points when a pattern ends at this node (`t[None]`).
    points: Option<Vec<u8>>,
}

struct Table {
    tree: Node,
    exceptions: HashMap<String, Vec<u8>>,
}

impl Table {
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

enum Out {
    Word(TimedString),
    Flush,
}

/// The one live segment and the gate around it, under one lock.
struct SyncState {
    pacer: Option<Pacer>,
    /// The `Heard` of the segment being finished, taken by the `PlaybackFinished` re-emit.
    heard: Option<Heard>,
    /// `audio_attached && text_attached`, unless `set_enabled` said otherwise since.
    enabled: bool,
    audio_attached: bool,
    text_attached: bool,
    closed: bool,
    /// `_SyncedTextOutput._capturing`: whether a flush has anything to end. With nothing
    /// captured, ending the text would pace later words against the next segment's audio.
    capturing_text: bool,
}

/// What both chain nodes and the facade share; nothing holds the facade, so no reference cycle.
struct SyncCore {
    opts: PacerOptions,
    clock: Arc<dyn Clock>,
    state: Mutex<SyncState>,
    /// `None` once closed: dropping the sender is what ends the forwarder.
    tx: Mutex<Option<mpsc::UnboundedSender<Out>>>,
    /// Shared with the forwarder task, which must not keep the core alive (or it never ends).
    forwarder_done: Arc<Monotone<bool>>,
    /// Every pacer warning, in order. A core crate has no logger; the host phrases them.
    warnings: Mutex<Vec<PaceWarning>>,
    /// The audio node's tracker; released words are raised on it as `transcript_shown`.
    tracker: Arc<PlaybackTracker>,
}

impl SyncCore {
    fn enabled(&self) -> bool {
        self.state.lock().unwrap().enabled
    }

    fn pacer(state: &SyncState) -> &Pacer {
        state.pacer.as_ref().expect("the pacer is only taken inside `apply`")
    }

    /// A finished segment takes nothing more: the next input starts the next one.
    fn fresh_if_over(&self, state: &mut SyncState) {
        if Self::pacer(state).is_over() {
            state.pacer = Some(Pacer::new(self.opts.clone()));
        }
    }

    fn rotate(&self, state: &mut SyncState) {
        if !Self::pacer(state).is_over() {
            self.apply(state, PaceEvent::Close);
        }
        state.pacer = Some(Pacer::new(self.opts.clone()));
    }

    fn apply(&self, state: &mut SyncState, ev: PaceEvent) {
        let pacer = state.pacer.take().expect("the pacer is only taken here");
        let (pacer, effects) = pacer.step(ev);
        state.pacer = Some(pacer);
        for effect in effects {
            match effect {
                PaceEffect::Forward { word, end_time } => {
                    self.tracker.transcript_shown(&word);
                    self.send(Out::Word(TimedString {
                        end_time: Some(end_time),
                        ..TimedString::new(word)
                    }));
                }
                PaceEffect::Flush => self.send(Out::Flush),
                PaceEffect::Heard(heard) => state.heard = Some(heard),
                PaceEffect::Warn(w) => self.warnings.lock().unwrap().push(w),
            }
        }
    }

    fn send(&self, out: Out) {
        if let Some(tx) = self.tx.lock().unwrap().as_ref() {
            let _ = tx.send(out);
        }
    }

    /// `_on_attachment_changed`. `None` leaves a leg's attachment where it is.
    fn on_attachment_changed(&self, audio: Option<bool>, text: Option<bool>) {
        let mut state = self.state.lock().unwrap();
        state.audio_attached = audio.unwrap_or(state.audio_attached);
        state.text_attached = text.unwrap_or(state.text_attached);
        let enabled = state.audio_attached && state.text_attached;
        self.set_enabled_locked(&mut state, enabled);
    }

    /// A change of `enabled` rotates the segment.
    fn set_enabled_locked(&self, state: &mut SyncState, enabled: bool) {
        if state.enabled != enabled {
            state.enabled = enabled;
            self.rotate(state);
        }
    }

    /// The forwarder: the only task and the only place the text sink is awaited.
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
}

impl SyncedAudioOutput {
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
            PlaybackEvent::Started(e) => self.tracker.playback_started(e.created_at),
            PlaybackEvent::Progressed(e) => {
                self.tracker.playback_progressed(e.started_at, e.offset, e.duration);
                let mut state = self.core.state.lock().unwrap();
                if state.enabled && !SyncCore::pacer(&state).is_over() {
                    self.core.apply(&mut state, PaceEvent::Position(e.offset + e.duration));
                }
            }
            PlaybackEvent::Finished(e) => {
                let heard = {
                    let mut state = self.core.state.lock().unwrap();
                    let pacer = SyncCore::pacer(&state);
                    if !state.enabled || pacer.is_over() || pacer.is_blank() {
                        None
                    } else {
                        // Audio with no words: finished, but `None` so the speech's own text stands.
                        let had_text = pacer.saw_text();
                        self.core.apply(
                            &mut state,
                            PaceEvent::PlaybackFinished {
                                position: e.playback_position,
                                interrupted: e.interrupted,
                            },
                        );
                        state.heard.take().filter(|_| had_text)
                    }
                };
                match heard {
                    Some(heard) => self.tracker.playback_finished(PlaybackFinished {
                        synchronized_transcript: Some(heard.text),
                        ..e.clone()
                    }),
                    None => self.tracker.playback_finished(e.clone()),
                };
            }
        }
    }
}

#[async_trait]
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

    async fn capture(&self, frame: &AudioFrame) -> Result<()> {
        // Passthrough first, unconditionally: the transcript must never hold up speech.
        self.next.capture_frame(frame).await?;
        if !self.core.enabled() {
            return Ok(());
        }
        let mut state = self.core.state.lock().unwrap();
        self.core.fresh_if_over(&mut state);
        if SyncCore::pacer(&state).audio_ended() {
            // A missing `playback_finished`: this frame is the next turn's.
            self.core.rotate(&mut state);
        }
        let secs = lk_agents_rtc::frame_duration(frame).as_secs_f64();
        self.core.apply(&mut state, PaceEvent::Audio { secs });
        Ok(())
    }

    fn flush_segment(&self) {
        self.next.flush();
        if !self.core.enabled() {
            return;
        }
        let mut state = self.core.state.lock().unwrap();
        let pacer = SyncCore::pacer(&state);
        // Ending a blank segment would end the next turn's audio early.
        if pacer.is_over() || pacer.is_blank() || pacer.audio_ended() {
            return;
        }
        self.core.apply(&mut state, PaceEvent::AudioEnd);
    }

    fn clear_buffer(&self) {
        self.next.clear_buffer();
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
        let mut state = self.core.state.lock().unwrap();
        state.capturing_text = true;
        self.core.fresh_if_over(&mut state);
        if SyncCore::pacer(&state).text_ended() {
            self.core.rotate(&mut state);
        }
        let start_time = if aligned { text.start_time } else { None };
        self.core.apply(&mut state, PaceEvent::Text { raw: text.text.clone(), start_time });
        Ok(())
    }
}

#[async_trait]
impl TextOutput for SyncedTextOutput {
    fn label(&self) -> &str {
        "TranscriptSynchronizer"
    }

    fn next_in_chain(&self) -> Option<Arc<dyn TextOutput>> {
        self.next.clone()
    }

    async fn capture_timed(&self, text: &TimedString) -> Result<()> {
        self.capture(text, true).await
    }

    async fn capture_text(&self, text: &str) -> Result<()> {
        self.capture(&TimedString::new(text), false).await
    }

    fn flush(&self) {
        if !self.core.enabled() {
            if let Some(next) = &self.next {
                next.flush();
            }
            return;
        }
        let mut state = self.core.state.lock().unwrap();
        if !std::mem::take(&mut state.capturing_text) {
            return;
        }
        let pacer = SyncCore::pacer(&state);
        if pacer.is_over() || pacer.text_ended() {
            return;
        }
        self.core.apply(&mut state, PaceEvent::TextEnd);
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
    /// words go to, or `None`. `hyphenate_word` is the caller's, or `None` for Liang.
    #[host(new(kw_only = "speed", defaults(speed = 1.0)))]
    pub fn for_host(
        next_in_chain_audio: Arc<dyn AudioOutput>,
        next_in_chain_text: Option<Arc<dyn TextOutput>>,
        speed: f64,
        hyphenate_word: Option<Arc<dyn Hyphenator>>,
        #[host(ambient)] spawner: Arc<dyn Spawner>,
    ) -> Self {
        let tracker = Arc::clone(next_in_chain_audio.tracker());
        let sync = Self::build(SyncConfig {
            audio_next: next_in_chain_audio,
            text_next: next_in_chain_text,
            speed,
            hyphenator: hyphenate_word.unwrap_or_else(|| Arc::new(Liang)),
            clock: Arc::new(lk_agents_abi::RealClock::new()),
            spawner,
        });
        // The host sink's playback, stamped on the clock the pacer runs on.
        tracker.anchor_to(sync.anchor);
        sync
    }

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
                pacer: Some(Pacer::new(opts.clone())),
                heard: None,
                enabled: true,
                audio_attached: true,
                text_attached: true,
                closed: false,
                capturing_text: false,
            }),
            opts,
            clock,
            tx: Mutex::new(Some(tx)),
            forwarder_done: Arc::new(Monotone::new(false)),
            warnings: Mutex::new(Vec::new()),
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

    /// The node to install as `session.output.audio`, as a host object.
    #[host(getter(name = "audio_output"))]
    pub fn audio_leg(&self) -> crate::room_io::_output::RoomAudioLeg {
        crate::room_io::_output::RoomAudioLeg::of(self.audio_output(), self.anchor)
    }

    /// The node to install as `session.output.transcription`, as a host object.
    #[host(getter(name = "text_output"))]
    pub fn text_leg(&self) -> crate::io::CoreTextOutput {
        crate::io::CoreTextOutput::new(self.text_output())
    }

    /// The pacer's warnings so far, by name (`PaceWarning::name`), in order.
    #[host(method(name = "warnings"))]
    pub fn warning_names(&self) -> Vec<&'static str> {
        self.warnings().into_iter().map(PaceWarning::name).collect()
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
        let mut state = self.core.state.lock().unwrap();
        self.core.set_enabled_locked(&mut state, enabled);
    }

    /// Every pacer warning so far, in order.
    pub fn warnings(&self) -> Vec<PaceWarning> {
        self.core.warnings.lock().unwrap().clone()
    }

    /// End the live segment now and start a fresh one.
    #[host(method)]
    pub fn rotate_segment(&self) {
        let mut state = self.core.state.lock().unwrap();
        if !state.closed {
            self.core.rotate(&mut state);
        }
    }

    /// Close: end the live segment and give the forwarder [`SEGMENT_ACLOSE_TIMEOUT`] to drain.
    /// `RoomIO.aclose` must close this before the transcription outputs it feeds.
    #[host(method)]
    pub async fn aclose(&self) {
        {
            let mut state = self.core.state.lock().unwrap();
            if state.closed {
                return;
            }
            state.closed = true;
            if !SyncCore::pacer(&state).is_over() {
                self.core.apply(&mut state, PaceEvent::Close);
            }
        }
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
