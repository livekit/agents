//! A full-duplex speech model (`DuplexRealtimeAdapter`) at the `Realtime` boundary: the events the
//! adapter hands the driver, as its segmenter makes them of the model's stream.
//!
//! The model speaks when asked: an utterance opens a generation with its first sound and its
//! audio arrives at playback pace; each word reaches the driver once the audio nears it, stamped
//! with its place in the utterance; the first silent frame closes the generation. Its text runs
//! behind its audio, as a duplex model's tokens do: now and then the last words of an utterance
//! arrive only after it ended, and each becomes a generation of its own attributed to the
//! completed speech (`TranscriptAttribution`). What the person says reaches the model as heard
//! audio; a barge-in is the model's own business. The adapter's record of the conversation is
//! append-only: what the driver gives it, and what the model said.
//!
//! The session is a resource (`Kind::Realtime`) the case's server events cannot reach, and every
//! utterance is one too (`Kind::Generation`), its audio tagged with it, until it ends.
use lk_agents_core::llm::history::ChatContextExt;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, SystemTime},
};

use fastrand::Rng;
use lk_agents_abi::{
    AudioFrame, BoxFuture, ChatContext, ChatItem, ChatMessage, Clock, CopyOptions, OwnedTask,
    Realtime, RealtimeCommand, RealtimeSession, Role, Spawner, SpawnerExt, Ts,
    errors::ProviderError,
    llm::realtime::{
        AudioDelta, Event, GenerateReplyFailed, GenerationCreated, GenerationDone,
        GenerationStatus, MessageDone, MessageModalities, MessageStarted, Modalities,
        RealtimeCapabilities, RealtimeError, TextDelta, TranscriptAttribution,
    },
};

use super::super::{Control, Registry, SAMPLE_RATE, Slot};
use crate::{
    input::AGENT_TURNS,
    record::{Detail, Kind, Tag},
};

/// One model frame: its own frames are this long.
const FRAME_MS: u64 = 100;
/// How long the model takes to start speaking once asked.
const FIRST_AUDIO_MS: std::ops::Range<u64> = 250..900;
/// Audio per character of what it says.
const MS_PER_CHAR: u64 = 55;
/// In a hundred utterances, how many have their last words' text arrive after the sound.
const LATE_TEXT: u64 = 35;
/// How far ahead of its sound a word's text arrives.
const TEXT_LEAD_MS: u64 = 200;
/// An ask the model has not answered with speech by then fails (the adapter's reply timeout).
const REPLY_TIMEOUT: Duration = Duration::from_secs(10);
/// The adapter's generation ids.
const ID_PREFIX: &str = "duplex_";

pub struct FakeDuplex {
    registry: Arc<Registry>,
    seed: u64,
    late_words: Option<u64>,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
}

impl FakeDuplex {
    pub fn new(
        registry: Arc<Registry>,
        seed: u64,
        late_words: Option<u64>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Arc<Self> {
        Arc::new(FakeDuplex { registry, seed, late_words, clock, spawner })
    }
}

/// What a duplex adapter reports: it transcribes no user audio, replies to no tool output by
/// itself, and takes context, instruction and tool updates.
fn capabilities() -> RealtimeCapabilities {
    RealtimeCapabilities {
        message_truncation: false,
        turn_detection: true,
        user_transcription: false,
        auto_tool_reply_generation: false,
        audio_output: true,
        manual_function_calls: false,
        can_disable_turn_detection: false,
        mutable_chat_context: true,
        mutable_instructions: true,
        mutable_tools: true,
        per_response_tool_choice: false,
        supports_say: false,
        auto_turn_reply_generation: true,
    }
}

impl Realtime for FakeDuplex {
    fn label(&self) -> String {
        "fake::duplex".into()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        capabilities()
    }

    /// Turn detection is the model's own: it is never asked to be off.
    fn session(&self, _: bool) -> BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
        Box::pin(async move {
            let model = Arc::new(Model {
                registry: self.registry.clone(),
                late_words: self.late_words,
                clock: self.clock.clone(),
                slot: OnceLock::new(),
                state: Mutex::new(State {
                    rng: Rng::with_seed(self.seed),
                    commands: VecDeque::new(),
                    configured: false,
                    chat_ctx: ChatContext::default(),
                    outputs: VecDeque::new(),
                    pending_reply: None,
                    speaking: None,
                    next_id: 0,
                    open: None,
                    last_spoken: None,
                    events: VecDeque::new(),
                    over: false,
                }),
                wake: tokio::sync::Notify::new(),
                ready: tokio::sync::Notify::new(),
            });
            let slot = self.registry.register(Kind::Realtime, Detail::None, Control::Generation);
            let _ = model.slot.set(slot);
            let pump = model.clone();
            let abort = self.spawner.spawn(async move { pump.run().await });
            Ok(Arc::new(Session { model, abort }) as Arc<dyn RealtimeSession>)
        })
    }
}

/// What the model sends, at its instant.
enum Output {
    /// An utterance's first sound: its generation opens.
    Opened { utterance: Slot },
    /// One frame of an utterance's speech.
    Audio { utterance: Slot, chunk: u16 },
    /// A word of the utterance, `start` seconds into it.
    Word { utterance: Slot, word: String, start: f64 },
    /// The utterance's first silent frame: its generation ends.
    Silence { utterance: Slot },
    /// A word of an ended utterance, its text arriving after its sound.
    Late { utterance: Slot, word: String },
}

/// The generation an utterance opened.
struct Open {
    utterance: Slot,
    id: String,
    transcript: String,
    opened_at: SystemTime,
}

struct State {
    rng: Rng,
    commands: VecDeque<RealtimeCommand>,
    /// The startup configuration went out (the replay ends with the tools).
    configured: bool,
    /// The adapter's record of the conversation.
    chat_ctx: ChatContext,
    /// What the model sends next, in time order.
    outputs: VecDeque<(Ts, Output)>,
    /// The ask the next utterance answers, and when it fails unanswered.
    pending_reply: Option<(String, Ts)>,
    /// The last utterance planned: when its sound starts and when it ends.
    speaking: Option<(Ts, Ts)>,
    next_id: u32,
    open: Option<Open>,
    /// The last utterance that ended, and its generation: what its late words are attributed to.
    last_spoken: Option<(Slot, String)>,
    events: VecDeque<Event>,
    over: bool,
}

impl State {
    fn mint_id(&mut self) -> String {
        self.next_id += 1;
        format!("{ID_PREFIX}{}", self.next_id)
    }

    fn fail_reply(&mut self, reason: &str) {
        if let Some((req_id, _)) = self.pending_reply.take() {
            let error = RealtimeError::new(reason);
            self.events
                .push_back(Event::GenerateReplyFailed(GenerateReplyFailed { req_id, error }));
        }
    }

    /// A generation's message: started, its modalities settled, `text` said, done.
    fn opened(&mut self, id: &str, req_id: Option<String>) {
        let response_id = Some(id.to_owned());
        let message_id = id.to_owned();
        self.events.extend([
            Event::GenerationCreated(GenerationCreated { response_id: id.into(), req_id }),
            Event::MessageStarted(MessageStarted {
                message_id: message_id.clone(),
                response_id: response_id.clone(),
            }),
            Event::MessageModalities(MessageModalities {
                message_id,
                response_id,
                modalities: Modalities::AudioAndText,
            }),
        ]);
    }

    fn done(&mut self, id: &str, transcript: String, at: SystemTime) {
        let response_id = id.to_owned();
        self.events.extend([
            Event::MessageDone(MessageDone {
                message_id: id.into(),
                response_id: Some(response_id.clone()),
            }),
            Event::GenerationDone(GenerationDone {
                response_id,
                status: GenerationStatus::Completed,
                usage: Default::default(),
            }),
        ]);
        if !transcript.is_empty() {
            let said = ChatMessage::text(id.to_owned(), Role::Assistant, transcript, at);
            self.chat_ctx.insert(ChatItem::Message(said));
        }
    }
}

struct Model {
    registry: Arc<Registry>,
    late_words: Option<u64>,
    clock: Arc<dyn Clock>,
    slot: OnceLock<Slot>,
    state: Mutex<State>,
    wake: tokio::sync::Notify,
    ready: tokio::sync::Notify,
}

impl Model {
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
                    biased;
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
                self.output(st, output);
                continue;
            }
            if st.pending_reply.as_ref().is_some_and(|(_, at)| *at <= now) {
                st.fail_reply("the model did not start speaking when asked");
                continue;
            }
            let due = st.outputs.front().map(|(at, _)| *at);
            return due.into_iter().chain(st.pending_reply.as_ref().map(|(_, at)| *at)).min();
        }
    }

    /// The adapter's handling of the driver's commands.
    fn command(&self, st: &mut State, now: Ts, cmd: RealtimeCommand) {
        match cmd {
            RealtimeCommand::PushAudio(p) => {
                if let Some(slot) = self.slot.get() {
                    let (pairs, _) = p.frame.data.as_chunks::<2>();
                    let tags = pairs.iter().map(|b| Tag::decode(i16::from_le_bytes(*b))).collect();
                    self.registry.record_forwarded(*slot, tags);
                }
            }
            // Append-only: the update's items replace the record, whatever the model keeps.
            RealtimeCommand::UpdateChatCtx(u) => {
                let mut chat_ctx = u.chat_ctx.copy(&CopyOptions {
                    exclude_handoff: true,
                    exclude_config_update: true,
                    ..CopyOptions::default()
                });
                if st.configured {
                    chat_ctx.items.retain(|i| i.id() != "lk.agent_task.instructions");
                }
                st.chat_ctx = chat_ctx;
            }
            RealtimeCommand::UpdateTools(_) => st.configured = true,
            RealtimeCommand::PushItem(p) => st.chat_ctx.insert(*p.item),
            // The reply is the next speech to open; a newer ask supersedes the one waiting.
            RealtimeCommand::GenerateReply(g) => {
                st.fail_reply("a newer ask superseded this one");
                st.pending_reply = Some((g.req_id, now.saturating_add(REPLY_TIMEOUT)));
                // One that has not started yet answers this ask too.
                if !st.speaking.is_some_and(|(start, _)| start > now) {
                    let free = st.speaking.map_or(now, |(_, end)| end.max(now));
                    let delay = Duration::from_millis(st.rng.u64(FIRST_AUDIO_MS));
                    self.speak(st, free.saturating_add(delay));
                }
            }
            RealtimeCommand::CancelReply(c) => {
                if st.pending_reply.as_ref().is_some_and(|(id, _)| *id == c.req_id) {
                    st.pending_reply = None;
                }
            }
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

    /// One utterance from `at`: its frames at playback pace, then its first silent frame; its
    /// words a little ahead of their sound, the trailing ones of a late transcript well after it.
    fn speak(&self, st: &mut State, at: Ts) {
        let spoken: Vec<&str> =
            AGENT_TURNS.iter().copied().filter(|t| !t.trim().is_empty()).collect();
        let text = spoken[st.rng.usize(0..spoken.len())];
        let frames = (text.chars().count() as u64 * MS_PER_CHAR).div_ceil(FRAME_MS).max(3);
        let after = |ms: u64| at.saturating_add(Duration::from_millis(ms));
        let utterance = self.registry.register(Kind::Generation, Detail::None, Control::Generation);
        let mut out = vec![(at, Output::Opened { utterance })];
        for n in 0..frames {
            let chunk = (n as u16).min(Tag::MAX_CHUNK);
            out.push((after(n * FRAME_MS), Output::Audio { utterance, chunk }));
        }
        let end = after(frames * FRAME_MS);
        let words: Vec<&str> = text.split_inclusive(' ').collect();
        let per_word = frames * FRAME_MS / words.len() as u64;
        let late = words.len() > 2 && (self.late_words.is_some() || st.rng.u64(0..100) < LATE_TEXT);
        let trailing = if late { st.rng.usize(1..=2) } else { 0 };
        for (i, word) in words.iter().enumerate() {
            let word = (*word).to_owned();
            let begin = i as u64 * per_word;
            if i >= words.len() - trailing {
                let after = self.late_words.unwrap_or_else(|| st.rng.u64(400..2_000));
                let due = end.saturating_add(Duration::from_millis(after));
                out.push((due, Output::Late { utterance, word }));
            } else {
                // With the first frame its text has arrived for and its sound is near.
                let frame = begin.saturating_sub(TEXT_LEAD_MS).div_ceil(FRAME_MS).max(1);
                let start = begin as f64 / 1000.0;
                out.push((after(frame * FRAME_MS), Output::Word { utterance, word, start }));
            }
        }
        out.push((end, Output::Silence { utterance }));
        st.speaking = Some((at, end));
        st.outputs.extend(out);
        st.outputs.make_contiguous().sort_by_key(|(at, _)| *at);
    }

    fn output(&self, st: &mut State, output: Output) {
        let wall = self.clock.wall();
        match output {
            Output::Opened { utterance } => {
                let id = st.mint_id();
                let req_id = st.pending_reply.take().map(|(id, _)| id);
                st.opened(&id, req_id);
                st.open = Some(Open { utterance, id, transcript: String::new(), opened_at: wall });
            }
            Output::Audio { utterance, chunk } => {
                let Some(open) = st.open.as_ref().filter(|o| o.utterance == utterance) else {
                    return;
                };
                self.registry.record_generated(utterance.nth, chunk);
                let message_id = open.id.clone();
                st.events.push_back(Event::AudioDelta(AudioDelta {
                    message_id: message_id.clone(),
                    response_id: Some(message_id),
                    frame: frame(utterance.nth, chunk),
                }));
            }
            Output::Word { utterance, word, start } => {
                let Some(open) = st.open.as_mut().filter(|o| o.utterance == utterance) else {
                    return;
                };
                open.transcript.push_str(&word);
                let message_id = open.id.clone();
                st.events.push_back(Event::TextDelta(TextDelta {
                    message_id: message_id.clone(),
                    response_id: Some(message_id),
                    delta: word,
                    start_time: Some(start),
                }));
            }
            Output::Silence { utterance } => {
                if let Some(open) = st.open.take_if(|o| o.utterance == utterance) {
                    st.done(&open.id, open.transcript, open.opened_at);
                    st.last_spoken = Some((utterance, open.id));
                }
                self.registry.mark_closed(utterance);
            }
            Output::Late { utterance, word } => {
                let Some((_, source)) = st.last_spoken.clone().filter(|(u, _)| *u == utterance)
                else {
                    return;
                };
                let id = st.mint_id();
                st.events.push_back(Event::TranscriptAttribution(TranscriptAttribution {
                    response_id: id.clone(),
                    source_response_id: source,
                }));
                st.opened(&id, None);
                st.events.push_back(Event::TextDelta(TextDelta {
                    message_id: id.clone(),
                    response_id: Some(id.clone()),
                    delta: word.clone(),
                    start_time: None,
                }));
                st.done(&id, word, wall);
            }
        }
    }

    fn close(&self, st: &mut State) {
        if let Some(open) = st.open.take() {
            st.done(&open.id, open.transcript, open.opened_at);
        }
        st.fail_reply("the session closed before the model replied");
        for (_, output) in st.outputs.drain(..) {
            if let Output::Silence { utterance } = output {
                self.registry.mark_closed(utterance);
            }
        }
        st.over = true;
    }
}

/// One frame of an utterance's speech, every sample naming its utterance and chunk.
fn frame(generation: u32, chunk: u16) -> AudioFrame {
    let samples = (FRAME_MS * u64::from(SAMPLE_RATE) / 1000) as usize;
    let pcm = Tag::generated(generation, chunk).to_le_bytes().repeat(samples);
    AudioFrame::new(pcm, SAMPLE_RATE, 1, samples as u32)
}

struct Session {
    model: Arc<Model>,
    /// The model's pump; cancelled at the close or with the session.
    abort: OwnedTask,
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

    fn head(&self, frame: lk_agents_abi::AudioFrame) {
        self.send(RealtimeCommand::PushAudio(lk_agents_abi::PushAudio { frame }));
    }

    fn next_event(&self) -> BoxFuture<'_, Option<Event>> {
        Box::pin(self.model.until(|st| match st.events.pop_front() {
            Some(event) => Some(Some(event)),
            None => st.over.then_some(None),
        }))
    }

    fn chat_ctx(&self) -> BoxFuture<'_, ChatContext> {
        Box::pin(async move { self.model.state.lock().unwrap().chat_ctx.clone() })
    }

    fn aclose(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            {
                let mut st = self.model.state.lock().unwrap();
                if !st.over {
                    self.model.close(&mut st);
                }
            }
            self.abort.cancel();
            self.model.ready.notify_waiters();
            if let Some(slot) = self.model.slot.get() {
                self.model.registry.mark_closed(*slot);
            }
        })
    }
}
