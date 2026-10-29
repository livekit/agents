//! A full-duplex speech model (`DuplexRealtimeAdapter`): the inference core's segmenter and gate
//! (`lk_agents_inference::duplex`) over a simulated model, driven the way the runtime's duplex
//! session drives them.
//!
//! The model speaks when asked: its audio arrives at playback pace, followed by its own silence
//! until the gate closes, and its transcript of itself comes as span-stamped fragments, the first
//! words just ahead of their sound. Its text runs behind its audio, as a duplex model's tokens do:
//! now and then the last words of an utterance arrive only after the utterance ended, and the
//! segmenter attributes them to the completed speech (`TranscriptAttribution`). What the person
//! says reaches the model as heard audio; a barge-in is the model's own business.
//!
//! The session is a resource (`Kind::Realtime`) the case's server events cannot reach, and every
//! utterance is one too (`Kind::Generation`), its audio tagged with it, until it ends.
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use fastrand::Rng;
use lk_agents_abi::{
    AbortHandle, AudioFrame, ChatContext, Clock, Realtime, RealtimeCommand, RealtimeSession,
    Spawner, SpawnerExt, Ts,
    errors::ProviderError,
    llm::realtime::{Event, RealtimeCapabilities},
};
use lk_agents_inference::duplex::{
    AUDIO_TIMEOUT, AudioGate, DuplexCapabilities, FixedGate, Fragment, Segmenter, Step,
};

use super::super::{Control, Registry, SAMPLE_RATE, Slot};
use crate::{
    input::AGENT_TURNS,
    record::{Detail, Kind, Tag},
};

/// One model frame: its own frames are this long, speech or silence.
const FRAME_MS: u64 = 100;
/// How long the model takes to start speaking once asked.
const FIRST_AUDIO_MS: std::ops::Range<u64> = 250..900;
/// Audio per character of what it says.
const MS_PER_CHAR: u64 = 55;
/// The silence the model streams after an utterance before it stops sending.
const TAIL_FRAMES: u64 = 2;
/// In a hundred utterances, how many have their last words' text arrive after the sound.
const LATE_TEXT: u64 = 35;

pub struct FakeDuplex {
    registry: Arc<Registry>,
    seed: u64,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
}

impl FakeDuplex {
    pub fn new(
        registry: Arc<Registry>,
        seed: u64,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Arc<Self> {
        Arc::new(FakeDuplex { registry, seed, clock, spawner })
    }
}

fn capabilities() -> RealtimeCapabilities {
    DuplexCapabilities {
        user_transcription: false,
        auto_tool_reply_generation: false,
        mutable_chat_context: true,
        mutable_instructions: true,
        mutable_tools: true,
    }
    .realtime()
}

impl Realtime for FakeDuplex {
    fn label(&self) -> String {
        "fake::duplex".into()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        capabilities()
    }

    /// Turn detection is the model's own: it is never asked to be off.
    fn session(
        &self,
        _: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
        Box::pin(async move {
            let model = Arc::new(Model {
                registry: self.registry.clone(),
                clock: self.clock.clone(),
                slot: OnceLock::new(),
                started: self.clock.now(),
                state: Mutex::new(State {
                    seg: Segmenter::new("fake::duplex", AUDIO_TIMEOUT, "duplex_".into()),
                    // Its own silence is digital zeros, speech stands far out of it, and it pauses
                    // inside no utterance: the first silent frame closes the gate, unplayed.
                    gate: FixedGate::new(0.0, 10.0, 2.0, 0.0),
                    rng: Rng::with_seed(self.seed),
                    commands: VecDeque::new(),
                    configured: false,
                    outputs: VecDeque::new(),
                    utterances: Vec::new(),
                    speaking: None,
                    events: VecDeque::new(),
                    over: false,
                }),
                wake: tokio::sync::Notify::new(),
                ready: tokio::sync::Notify::new(),
            });
            let slot = self.registry.register(Kind::Realtime, Detail::None, Control::Generation);
            let _ = model.slot.set(slot);
            let pump = model.clone();
            let abort = self.spawner.spawn_unboxed(async move { pump.run().await });
            Ok(Arc::new(Session { model, abort }) as Arc<dyn RealtimeSession>)
        })
    }
}

/// What the model sends, at its instant.
enum Output {
    Audio {
        frame: AudioFrame,
        start_ms: i64,
    },
    Text(Fragment),
    /// The utterance's last sound went out: its generation ends.
    Ended(Slot),
}

struct State {
    seg: Segmenter,
    gate: FixedGate,
    rng: Rng,
    commands: VecDeque<RealtimeCommand>,
    /// The startup configuration went out (the replay ends with the tools).
    configured: bool,
    /// What the model sends next, in time order.
    outputs: VecDeque<(Ts, Output)>,
    /// The utterances it is speaking or will speak, and when the last one's sound ends: an ask
    /// while it speaks is answered by the next utterance.
    utterances: Vec<Slot>,
    /// The last utterance planned: when its sound starts and when it ends.
    speaking: Option<(Ts, Ts)>,
    events: VecDeque<Event>,
    over: bool,
}

struct Model {
    registry: Arc<Registry>,
    clock: Arc<dyn Clock>,
    slot: OnceLock<Slot>,
    /// The model's audio clock starts with the session.
    started: Ts,
    state: Mutex<State>,
    wake: tokio::sync::Notify,
    ready: tokio::sync::Notify,
}

impl Model {
    fn audio_ms(&self, at: Ts) -> i64 {
        at.saturating_duration_since(self.started).as_millis() as i64
    }

    async fn until<T>(&self, mut read: impl FnMut(&mut State) -> Option<T>) -> T {
        loop {
            let ready = self.ready.notified();
            tokio::pin!(ready);
            ready.as_mut().enable();
            if let Some(value) = read(&mut self.state.lock().unwrap()) {
                return value;
            }
            ready.await;
        }
    }

    async fn run(self: Arc<Self>) {
        loop {
            let next = {
                let mut st = self.state.lock().unwrap();
                self.advance(&mut st)
            };
            self.ready.notify_waiters();
            match next {
                Some(at) => tokio::select! {
                    () = self.clock.wait_until(at) => {}
                    () = self.wake.notified() => {}
                },
                None => self.wake.notified().await,
            }
        }
    }

    /// Everything due now; then the next instant anything is.
    fn advance(&self, st: &mut State) -> Option<Ts> {
        loop {
            let now = self.clock.now();
            if st.over {
                return None;
            }
            if let Some(cmd) = st.commands.pop_front() {
                self.command(st, now, cmd);
                continue;
            }
            if st.outputs.front().is_some_and(|(at, _)| *at <= now) {
                let (_, output) = st.outputs.pop_front().expect("just seen");
                self.output(st, now, output);
                continue;
            }
            if st.seg.next_deadline().is_some_and(|at| at <= now) {
                let step = st.seg.tick(now);
                self.apply(st, step);
                continue;
            }
            let due = st.outputs.front().map(|(at, _)| *at);
            return due.into_iter().chain(st.seg.next_deadline()).min();
        }
    }

    fn apply(&self, st: &mut State, step: Step) {
        if step.deactivate_gate {
            st.gate.deactivate();
        }
        st.events.extend(step.events);
    }

    /// The runtime duplex session's handling of the driver's commands.
    fn command(&self, st: &mut State, now: Ts, cmd: RealtimeCommand) {
        match cmd {
            RealtimeCommand::PushAudio(p) => {
                if let Some(slot) = self.slot.get() {
                    let (pairs, _) = p.frame.data.as_chunks::<2>();
                    let tags = pairs.iter().map(|b| Tag::decode(i16::from_le_bytes(*b))).collect();
                    self.registry.record_forwarded(*slot, tags);
                }
            }
            RealtimeCommand::UpdateChatCtx(u) => {
                if st.configured {
                    st.seg.update_chat_ctx(&u.chat_ctx);
                } else {
                    st.seg.seed(&u.chat_ctx);
                }
            }
            RealtimeCommand::UpdateTools(_) => st.configured = true,
            RealtimeCommand::PushItem(p) => st.seg.push_item(*p.item),
            RealtimeCommand::GenerateReply(g) => {
                let events = st.seg.generate_reply(now, g.req_id);
                st.events.extend(events);
                // One that has not started yet answers this ask too.
                if !st.speaking.is_some_and(|(start, _)| start > now) {
                    let free = st.speaking.map_or(now, |(_, end)| end.max(now));
                    let delay = Duration::from_millis(st.rng.u64(FIRST_AUDIO_MS));
                    self.speak(st, free.saturating_add(delay));
                }
            }
            RealtimeCommand::CancelReply(c) => st.seg.forget_reply(&c.req_id),
            // Instructions it takes as they come; input is consumed continuously, barge-in and
            // the output timeline are the model's own.
            RealtimeCommand::UpdateInstructions(_)
            | RealtimeCommand::Say(_)
            | RealtimeCommand::CommitAudio(_)
            | RealtimeCommand::ClearAudio(_)
            | RealtimeCommand::Interrupt(_)
            | RealtimeCommand::Truncate(_) => {}
        }
    }

    /// One utterance from `at`: its frames at playback pace, its silence, its text.
    fn speak(&self, st: &mut State, at: Ts) {
        let spoken: Vec<&str> =
            AGENT_TURNS.iter().copied().filter(|t| !t.trim().is_empty()).collect();
        let text = spoken[st.rng.usize(0..spoken.len())];
        let frames = (text.chars().count() as u64 * MS_PER_CHAR).div_ceil(FRAME_MS).max(3);
        let start = self.audio_ms(at);
        let ms = |n: u64| Duration::from_millis(n * FRAME_MS);
        let slot = self.registry.register(Kind::Generation, Detail::None, Control::Generation);
        st.utterances.push(slot);
        let mut out: Vec<(Ts, Output)> = Vec::new();
        for n in 0..frames + TAIL_FRAMES {
            let speech = n < frames;
            let frame = frame(speech.then_some((slot.nth, n as u16)));
            let start_ms = start + (n * FRAME_MS) as i64;
            out.push((at.saturating_add(ms(n)), Output::Audio { frame, start_ms }));
        }
        // The model's span clock runs apart from its audio clock by a fixed offset.
        let span = st.rng.i64(0..10_000);
        let words: Vec<&str> = text.split_inclusive(' ').collect();
        let per_word = (frames * FRAME_MS) as i64 / words.len() as i64;
        let late = st.rng.u64(0..100) < LATE_TEXT && words.len() > 2;
        let trailing = if late { st.rng.usize(1..=2) } else { 0 };
        let end = at.saturating_add(ms(frames + TAIL_FRAMES));
        for (i, word) in words.iter().enumerate() {
            let begin = i as i64 * per_word;
            let fragment = Fragment {
                text: (*word).to_owned(),
                start_ms: Some(span + start + begin),
                end_ms: Some(span + start + begin + per_word),
            };
            // A little ahead of its sound, never before the utterance opened; the trailing
            // words of a late transcript well after the sound ended.
            let due = if i >= words.len() - trailing {
                end.saturating_add(Duration::from_millis(st.rng.u64(400..2_000)))
            } else {
                let lead = Duration::from_millis(begin.saturating_sub(200).max(1) as u64);
                at.saturating_add(lead)
            };
            out.push((due, Output::Text(fragment)));
        }
        out.push((end, Output::Ended(slot)));
        st.speaking = Some((at, end));
        st.outputs.extend(out);
        st.outputs.make_contiguous().sort_by_key(|(at, _)| *at);
    }

    fn output(&self, st: &mut State, now: Ts, output: Output) {
        let wall = self.clock.wall();
        match output {
            Output::Audio { frame, start_ms } => {
                if let Some(Tag::Generated { generation, chunk }) = frame
                    .data
                    .as_chunks::<2>()
                    .0
                    .first()
                    .map(|b| Tag::decode(i16::from_le_bytes(*b)))
                {
                    self.registry.record_generated(generation, chunk);
                }
                let open = st.gate.update(&frame);
                let step = st.seg.on_audio(now, wall, frame, Some(start_ms), open);
                self.apply(st, step);
            }
            Output::Text(fragment) => {
                let step = st.seg.on_transcript(now, wall, fragment);
                self.apply(st, step);
            }
            Output::Ended(slot) => {
                self.registry.mark_closed(slot);
                st.utterances.retain(|u| *u != slot);
            }
        }
    }

    fn close(&self, st: &mut State) {
        let step = st.seg.close();
        self.apply(st, step);
        st.outputs.clear();
        for slot in st.utterances.drain(..) {
            self.registry.mark_closed(slot);
        }
        st.over = true;
    }
}

/// The model's frame: speech, every sample naming its utterance and chunk, or its own silence.
fn frame(speech: Option<(u32, u16)>) -> AudioFrame {
    let samples = (FRAME_MS * u64::from(SAMPLE_RATE) / 1000) as usize;
    let pcm = match speech {
        Some((generation, chunk)) => {
            Tag::generated(generation, chunk.min(Tag::MAX_CHUNK)).to_le_bytes().repeat(samples)
        }
        None => vec![0u8; samples * 2],
    };
    AudioFrame::new(pcm, SAMPLE_RATE, 1, samples as u32)
}

struct Session {
    model: Arc<Model>,
    abort: AbortHandle,
}

impl RealtimeSession for Session {
    fn capabilities(&self) -> RealtimeCapabilities {
        capabilities()
    }

    fn send(&self, cmd: RealtimeCommand) {
        let mut st = self.model.state.lock().unwrap();
        if !st.over {
            st.commands.push_back(cmd);
            drop(st);
            self.model.wake.notify_one();
        }
    }

    fn next_event(&self) -> lk_agents_abi::BoxFuture<'_, Option<Event>> {
        Box::pin(async move {
            self.model
                .until(|st| match st.events.pop_front() {
                    Some(event) => Some(Some(event)),
                    None => st.over.then_some(None),
                })
                .await
        })
    }

    fn chat_ctx(&self) -> lk_agents_abi::BoxFuture<'_, ChatContext> {
        Box::pin(async move { self.model.state.lock().unwrap().seg.chat_ctx().clone() })
    }

    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            {
                let mut st = self.model.state.lock().unwrap();
                if !st.over {
                    self.model.close(&mut st);
                }
            }
            self.abort.abort();
            self.model.ready.notify_waiters();
            if let Some(slot) = self.model.slot.get() {
                self.model.registry.mark_closed(*slot);
            }
        })
    }
}
