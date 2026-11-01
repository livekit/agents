//! The OpenAI Realtime server, as the real one was measured answering the protocol and, where
//! that is silent, as the docs say: the peer the realtime core is tested against
//! ([`super::sim`]).
//!
//! Sans-IO, in the protocol's typed records: the realtime core's client frames
//! ([`Outgoing`]) go in with the instant they arrived ([`Server::client`]), what the server does
//! on its own accord goes in as an [`Exo`], and every server event comes out of [`Server::pop`]
//! once it is due, as the [`Incoming`] the client decodes from it. Each event's delay after its
//! cause is drawn from a hand-set range around what the real server took, so a reply is
//! created about 100 ms after its `response.create`, its first item 300 ms later, its audio in
//! 400 ms deltas faster than real time, and `response.done` while the client is still playing
//! most of it -- which is what made a server onset over a finished generation the common
//! barge-in, not a rare one. The server answers the client's frames in the order they came.
//!
//! It holds the conversation: items by id and kind, placed where the real server places them (a
//! reply's item when it is added, after any client item that arrived first), deleted and
//! truncated on the client's word (a truncation cuts what the item holds); a reply reads what the
//! conversation holds when it is asked. One reply at a time: a second `response.create` over an
//! active one is refused, a cancel frees the conversation at once (a reply asked in the same
//! breath is created before the cancelled one's tail and `response.done`), and a cancelled reply
//! keeps streaming what it had generated before the cancel landed. Under server VAD an onset needs
//! audio the client sent; it cancels the reply in progress when `interrupt_response`
//! (`cancelled`/`turn_detected`, its message `incomplete`), and the end of the turn commits the
//! buffer and, with `create_response` and no reply active, opens one. The transcription completes
//! asynchronously, before or after the reply's events.
//!
//! What the real server refuses it refuses, with the real code and the real choice of citing the
//! frame or not, and reports each refusal as a [`Breach`] with whether the client already knew
//! better when it sent the frame (it had been told the state that makes the frame invalid) or
//! merely raced an event on the wire. So are two answers the server gives from the wrong state: a
//! reply asked in the same breath before the tool output it should read, and a cancel landing on
//! the server's reply to the person's turn before anything prompted it.
//!
//! The audio a response carries is silence of the delta's nominal length: its length is what
//! truncation is judged against.
use std::{collections::BTreeMap, time::SystemTime};

use fastrand::Rng;
use lk_agents_abi::{
    AudioFrame, ChatItem, ChatMessage, FunctionCall, Role, ToolChoice, Ts,
    llm::realtime::GenerationStatus, usage::TokenUsage,
};
use lk_agents_inference::realtime::{
    Incoming, TurnDetection,
    protocol::{ResponseDone, TranscriptionCompleted, TranscriptionUsage},
};

#[path = "client_tests.rs"]
mod client;

/// The session's audio rate, both ways.
pub const SAMPLE_RATE: u32 = 24_000;

/// What the model says: sentence-length replies, unnormalised text among them.
const REPLIES: &[&str] = &[
    "Sure, I can help with that.",
    "Let me pull up your account -- one moment.",
    "I found three appointments: Tuesday at 9, Wednesday at 2:30, and Friday at 11 a.m. Which works best?",
    "Okay. I've cancelled the 2:30 and moved you to Friday. You'll get a confirmation by text.",
    "I'm sorry, I didn't catch that. Could you say the last four digits again?",
    "Your balance is $1,204.55 as of today, and the next payment of $87.20 is due on 03/14.",
    "That's outside what I can do here, so I'll pass you to someone who can.",
    "Dr. Alvarez is at 1600 Pennsylvania Ave. NW, Ste. 200 -- call (202) 555-0147 ext. 12.",
    "Ref #A1B2-C3D4; ETA ~15 min; cost is EUR 12.50 (approx. USD 13.60).",
    "Yes... no, wait -- actually, let me check that again.",
    "# Your options\n\n1. **Upgrade** to _Premium_ \u{1F680}\n2. Keep `basic` -- see [the plan](https://example.com/plans).\n",
    "Done \u{2705} Your refund of ~~$40~~ **$45** is on its way! \u{1F389}\u{1F44D}\u{1F3FD}",
    "> Note: *the office is closed* on Monday.\n- Tuesday 9-5\n- Wednesday 9-1 \u{2615}",
];

/// What the caller's transcribed turns say: disfluent, as people talk.
const UTTERANCES: &[&str] = &[
    "yeah hi um I'm calling about my order",
    "can you hear me",
    "no no not that one, the other one",
    "uh let me think... okay so it was last tuesday I think? maybe wednesday",
    "WAIT stop stop stop",
    "my number is five five five, oh one four seven",
    "what",
    "okay thanks bye",
];

/// A client frame the server refused, or a request it answered from the wrong state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Breach {
    /// Refused with an `error` event. `knowing`: when the client sent it, it had already been told
    /// what makes it invalid (the response it cancels was done, the one it asks over had been
    /// created, the item it names never was); otherwise it raced an event still on the wire.
    Refused { event: String, code: &'static str, why: &'static str, knowing: bool },
    /// A response asked for while the conversation ended in a function call without its output,
    /// whose output the client sent afterwards: the reply read the conversation without it.
    AskedBeforeOutput { response: String, request: Option<String>, call_id: String },
    /// A client `response.cancel` stopped the server's answer to the person's turn (its VAD ended
    /// the turn and replied), before any of its audio and with no onset told over it.
    /// `generation` is the hooks' id of the reply, once the client was told of it.
    CancelledUnprompted { response: String, generation: Option<u32> },
}

/// The nominal length of a full audio delta, as the real server sends nearly all of them.
const DELTA_MS: u64 = 400;

/// What a delta's PCM is long: the client plays all of it, as long as the server counts it.
const PCM_MS: u64 = DELTA_MS;

/// The samples a delta carries.
const DELTA_SAMPLES: usize = (DELTA_MS * SAMPLE_RATE as u64 / 1000) as usize;

/// What the server is told to do of its own accord.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exo {
    /// The oldest response still generating finishes now, all of it at once.
    FinishNow,
}

/// Server-side VAD, as the session configured it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vad {
    Semantic,
    Server,
}

/// An event the client reads no further than its type.
fn unhandled(event_type: &str) -> Incoming {
    Incoming::Unhandled { event_type: event_type.to_owned() }
}

#[derive(Clone, Debug, Default)]
struct Config {
    vad: Option<Vad>,
    instructions: Option<String>,
    voice: Option<String>,
    create_response: bool,
    interrupt_response: bool,
    transcription: bool,
    audio: bool,
    tools: Vec<String>,
    /// The session's `tool_choice`: `none` forbids calls, `required` makes every response one.
    tool_choice: Option<ToolChoice>,
}

impl Config {
    fn turn_detection(&mut self, td: Option<&TurnDetection>) {
        let (vad, create, interrupt) = match td {
            Some(TurnDetection::ServerVad { create_response, interrupt_response, .. }) => {
                (Some(Vad::Server), *create_response, *interrupt_response)
            }
            Some(TurnDetection::SemanticVad { create_response, interrupt_response, .. }) => {
                (Some(Vad::Semantic), *create_response, *interrupt_response)
            }
            Some(TurnDetection::Raw(v)) => {
                let vad = match v["type"].as_str() {
                    Some("semantic_vad") => Some(Vad::Semantic),
                    Some("server_vad") => Some(Vad::Server),
                    _ => None,
                };
                let flag = |k: &str| v[k].as_bool().unwrap_or(true);
                (vad, flag("create_response"), flag("interrupt_response"))
            }
            None => (None, true, true),
        };
        self.vad = vad;
        self.create_response = create;
        self.interrupt_response = interrupt;
    }
}

/// A server event and what delivering it means to the server's own bookkeeping.
#[derive(Clone, Debug)]
enum Body {
    Event(Box<Incoming>),
    /// An audio delta, built when it is delivered: its PCM names the generation that carries it.
    Audio {
        item: String,
        chunk: u16,
    },
}

#[derive(Clone, Debug)]
struct Pending {
    resp: Option<u32>,
    body: Body,
}

/// One conversation item the server holds.
#[derive(Clone, Debug)]
struct Item {
    id: String,
    /// `message`, `function_call` or `function_call_output`.
    kind: &'static str,
    role: Option<Role>,
    call_id: Option<String>,
    /// Audio delivered for an assistant message, nominal ms: what truncation is judged against.
    audio_ms: u64,
    /// The same audio as the client plays it (a delta's PCM is shorter than its nominal length):
    /// what the item keeps of it, a truncation cutting it where the client says.
    pcm_ms: u64,
    /// Where a truncation cut it: audio the server had generated and not yet delivered is cut too.
    cut_ms: Option<u64>,
    /// The generation that made it (`Hooks::created`'s id), for the server's own messages.
    source: Option<u32>,
}

impl Item {
    fn of(item: &ChatItem) -> Self {
        let (kind, role, call_id) = match item {
            ChatItem::Message(m) => ("message", Some(m.role), None),
            ChatItem::FunctionCall(f) => ("function_call", None, Some(f.call_id.clone())),
            ChatItem::FunctionCallOutput(f) => {
                ("function_call_output", None, Some(f.call_id.clone()))
            }
            ChatItem::AgentHandoff(_) | ChatItem::AgentConfigUpdate(_) => ("other", None, None),
        };
        Item {
            id: item.id().to_owned(),
            kind,
            role,
            call_id,
            audio_ms: 0,
            pcm_ms: 0,
            cut_ms: None,
            source: None,
        }
    }

    fn assistant(&self) -> bool {
        self.role == Some(Role::Assistant)
    }
}

/// A message as the server sends it, its content left out as the decoder reads it; stamped with
/// its arrival when delivered.
fn message(id: &str, role: Role) -> ChatItem {
    ChatItem::Message(ChatMessage::new(id, role, Vec::new(), SystemTime::UNIX_EPOCH))
}

fn function_call(id: &str, name: &str, arguments: &str) -> ChatItem {
    ChatItem::FunctionCall(FunctionCall {
        id: id.to_owned(),
        call_id: format!("call_{id}"),
        arguments: arguments.to_owned(),
        name: name.to_owned(),
        created_at: SystemTime::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    })
}

/// An item's `created_at` is the client's clock when the event arrived, as the decoder reads it.
fn stamp(event: &mut Incoming, wall: SystemTime) {
    if let Incoming::ItemAdded { item, .. } | Incoming::OutputItemDone { item } = event {
        match item {
            ChatItem::Message(m) => m.created_at = wall,
            ChatItem::FunctionCall(f) => f.created_at = wall,
            ChatItem::FunctionCallOutput(f) => f.created_at = wall,
            ChatItem::AgentHandoff(_) | ChatItem::AgentConfigUpdate(_) => {}
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Response {
    id: String,
    /// The `client_event_id` the client tagged it with.
    request: Option<String>,
    /// The server's answer to the person's turn its VAD ended (`create_response`).
    turn: bool,
    /// What it reads: the session's instructions (or the request's own) and how many items.
    instructions: Option<String>,
    read_items: usize,
    /// The message item, once minted (it is announced later).
    item: Option<String>,
    /// `response.output_item.added` for the message was delivered.
    announced: bool,
    /// `response.output_item.done` for the message was delivered: it is complete.
    closed: bool,
    /// When its `response.created` and its `response.done` were delivered. The client knows of
    /// either only once it had a moment to read it: a frame it sends in that same instant crossed
    /// the event on the wire.
    told_created: Option<Ts>,
    finished: Option<Ts>,
    /// The client was told the person started speaking while it was generating: a reason to
    /// cancel it.
    prompted: bool,
    /// `response.created` delivered: the generation exists for the client.
    created: bool,
    /// Delivered audio, nominal ms.
    audio_ms: u64,
    /// The generation's source id, from the hook at creation.
    source: Option<u32>,
    /// The tool call it makes, `(item, name)`.
    function_call: Option<(String, String)>,
    /// The call's `response.output_item.added` was delivered.
    call_announced: bool,
    /// The call's `response.output_item.done` was delivered: it is complete.
    call_closed: bool,
}

/// What the glue does when the server delivers an event: give the response its source id at
/// creation, hear the person's audio, close a generation.
pub trait Hooks {
    /// A response the client now knows of: the id its audio is tagged with.
    fn created(&mut self, response: &str) -> u32;
    /// A response finished, whatever its status.
    fn done(&mut self, source: u32);
    /// One audio delta delivered, as the chunk it is of its generation.
    fn produced(&mut self, source: u32, chunk: u16);
    /// The person's audio the client appended, as PCM.
    fn heard(&mut self, pcm: &[u8]);
    /// A client frame the real server would refuse or answer from the wrong state.
    fn breach(&mut self, breach: Breach);
}

/// One connection's server: its conversation, input buffer and responses.
pub struct Server {
    rng: Rng,
    config: Config,
    /// Events due, by instant then by the order they were scheduled.
    pending: BTreeMap<(Ts, u64), Pending>,
    seq: u64,
    items: Vec<Item>,
    next_item: u32,
    next_resp: u32,
    next_event: u32,
    responses: Vec<Response>,
    /// The response the server is generating, by index: at most one.
    active: Option<u32>,
    /// The user item the VAD opened with `speech_started`.
    speaking: Option<String>,
    /// Audio appended since the last commit or clear, and since the connection opened, in ms.
    buffer_ms: u64,
    heard_ms: u64,
    /// The response the client was last told of and not yet told is done: what it knows is
    /// active, from the events delivered so far.
    told_active: Option<u32>,
    /// The buffer was last emptied by the client's own commit or clear (or never), so the client
    /// can count what it holds; a server VAD commit empties it behind the client's back.
    own_empty: bool,
    /// Commits and clears whose event the client has not been delivered yet.
    untold_empties: u32,
    /// Assistant audio delivered on this connection: the voice is fixed from then on.
    voiced: bool,
    /// Replies asked while the conversation ended in a call without its output: `(response,
    /// call_id, when)`. The client asking for the tools' follow-up sends the outputs and the
    /// `response.create` in one breath; an output arriving in that breath after the reply, with
    /// no reply asked after it, was meant to be read by a reply that was asked too early.
    unanswered: Vec<(String, String, Ts)>,
    /// Outputs that arrived after such a reply in its breath, until the breath ends without
    /// another reply asked: then each is an [`Breach::AskedBeforeOutput`].
    early: Vec<(String, String, Ts)>,
    /// What the client did wrong, for the glue to record.
    breaches: Vec<Breach>,
    /// When the server answers the last client frame it took: the next answer comes after.
    answered: Ts,
    /// The last user item, for a failed transcription.
    last_user: Option<String>,
    /// How often a response calls a tool, in percent, when the session offers tools: the
    /// developer's prompt decides it. `None` is the measured rate (8 in 94).
    tool_calls: Option<u8>,
    /// What this connection's ids are minted under: a real server's ids are unique across
    /// sessions, so a reconnect never names a new item as an old one.
    ids: String,
}

impl Server {
    pub fn new(seed: u64) -> Self {
        Server {
            rng: Rng::with_seed(seed),
            config: Config::default(),
            pending: BTreeMap::new(),
            seq: 0,
            items: Vec::new(),
            next_item: 0,
            next_resp: 0,
            next_event: 0,
            responses: Vec::new(),
            active: None,
            speaking: None,
            buffer_ms: 0,
            heard_ms: 0,
            told_active: None,
            own_empty: true,
            untold_empties: 0,
            voiced: false,
            unanswered: Vec::new(),
            early: Vec::new(),
            breaches: Vec::new(),
            answered: Ts::ZERO,
            last_user: None,
            tool_calls: None,
            ids: String::new(),
        }
    }

    /// This connection's id namespace.
    pub fn with_ids(mut self, prefix: impl Into<String>) -> Self {
        self.ids = prefix.into();
        self
    }

    /// A model whose prompt has it call tools this often, in percent.
    pub fn with_tool_calls(mut self, percent: Option<u8>) -> Self {
        self.tool_calls = percent;
        self
    }

    /// The connection opened: `session.created`, before anything else.
    pub fn connect(&mut self, now: Ts) {
        self.put(now, None, Incoming::SessionCreated { model: None });
    }

    pub fn next_due(&self) -> Option<Ts> {
        self.pending.keys().next().map(|(t, _)| *t)
    }

    /// The next event due at `now`, delivered at wall time `wall`.
    pub fn pop(&mut self, now: Ts, wall: SystemTime, hooks: &mut dyn Hooks) -> Option<Incoming> {
        self.breathe(now);
        self.report(hooks);
        let (&key, _) = self.pending.iter().next().filter(|((t, _), _)| *t <= now)?;
        let Pending { resp, body } = self.pending.remove(&key).expect("just seen");
        let mut event = match body {
            Body::Event(event) => *event,
            Body::Audio { item, chunk } => {
                let r = resp.map(|r| &mut self.responses[r as usize]);
                let source = r.as_ref().and_then(|r| r.source).unwrap_or(0);
                if let Some(r) = r {
                    r.audio_ms += DELTA_MS;
                }
                if let Some(i) = self.items.iter_mut().find(|i| i.id == item) {
                    i.audio_ms += DELTA_MS;
                    i.pcm_ms = (i.pcm_ms + PCM_MS).min(i.cut_ms.unwrap_or(u64::MAX));
                }
                self.voiced = true;
                hooks.produced(source, chunk);
                let pcm = vec![0; DELTA_SAMPLES * 2];
                let frame = AudioFrame::new(pcm, SAMPLE_RATE, 1, DELTA_SAMPLES as u32);
                Incoming::AudioDelta { item_id: item, frame }
            }
        };
        stamp(&mut event, wall);
        if resp.is_some()
            && let Incoming::ItemAdded { previous_item_id, item } = &mut event
        {
            *previous_item_id = self.items.last().map(|i| i.id.clone());
            let mut held = Item::of(item);
            held.source = resp.and_then(|r| self.responses[r as usize].source);
            self.items.push(held);
        }
        match &event {
            Incoming::SpeechStarted { .. } => {
                for r in [self.active, self.told_active].into_iter().flatten() {
                    self.responses[r as usize].prompted = true;
                }
            }
            Incoming::Unhandled { event_type }
                if matches!(
                    event_type.as_str(),
                    "input_audio_buffer.committed" | "input_audio_buffer.cleared"
                ) =>
            {
                self.untold_empties = self.untold_empties.saturating_sub(1);
            }
            _ => {}
        }
        if let Some(r) = resp {
            let resp = &mut self.responses[r as usize];
            match &event {
                Incoming::ResponseCreated { .. } => {
                    resp.source = Some(hooks.created(&resp.id));
                    resp.created = true;
                    resp.told_created = Some(now);
                    self.told_active = Some(r);
                }
                Incoming::OutputItemAdded { item_type, .. } if item_type == "message" => {
                    resp.announced = true;
                }
                Incoming::OutputItemAdded { item_type, .. } if item_type == "function_call" => {
                    resp.call_announced = true;
                }
                Incoming::OutputItemDone { item: ChatItem::Message(_) } => resp.closed = true,
                Incoming::OutputItemDone { item: ChatItem::FunctionCall(_) } => {
                    resp.call_closed = true;
                }
                Incoming::ResponseDone(_) => {
                    resp.finished = Some(now);
                    if let Some(source) = resp.source {
                        hooks.done(source);
                    }
                    if self.active == Some(r) {
                        self.active = None;
                    }
                    if self.told_active == Some(r) {
                        self.told_active = None;
                    }
                }
                _ => {}
            }
        }
        Some(event)
    }

    // --- what the server does of its own accord ------------------------------------------------

    /// True when a real server could do this now.
    pub fn exo(&mut self, now: Ts, what: Exo) -> bool {
        match what {
            Exo::FinishNow => {
                let Some(r) = self.active else { return false };
                let keys: Vec<(Ts, u64)> = self
                    .pending
                    .iter()
                    .filter(|(_, p)| p.resp == Some(r))
                    .map(|(k, _)| *k)
                    .collect();
                // Everything still to come of it, at once and in order.
                let moved: Vec<Pending> =
                    keys.iter().filter_map(|k| self.pending.remove(k)).collect();
                for p in moved {
                    self.seq += 1;
                    self.pending.insert((now, self.seq), p);
                }
                true
            }
        }
    }

    // --- the pieces ----------------------------------------------------------------------------

    /// The buffer becomes a user item: `committed`, the item, and its transcription later.
    fn commit(&mut self, now: Ts, item: String) {
        self.buffer_ms = 0;
        self.untold_empties += 1;
        let previous = self.items.last().map(|i| i.id.clone());
        let user = message(&item, Role::User);
        self.items.push(Item::of(&user));
        self.last_user = Some(item.clone());
        self.put(later(now, 0), None, unhandled("input_audio_buffer.committed"));
        self.after(now, 1, Incoming::ItemAdded { previous_item_id: previous, item: user });
        self.put(later(now, 2), None, unhandled("conversation.item.done"));
        if !self.config.transcription {
            return;
        }
        let text = UTTERANCES[self.rng.usize(0..UTTERANCES.len())];
        let lag = self.rng.u64(340..=700);
        let words: Vec<&str> = text.split(' ').collect();
        // About one delta a word, streamed over the last fifth of the wait.
        for (i, word) in words.iter().enumerate() {
            let at = lag * 4 / 5 + (lag / 5) * i as u64 / words.len().max(1) as u64;
            let delta = if i == 0 { (*word).to_owned() } else { format!(" {word}") };
            let item_id = item.clone();
            self.after(now, at, Incoming::TranscriptionDelta { item_id, content_index: 0, delta });
        }
        let output = words.len() as u32;
        let completed = TranscriptionCompleted {
            item_id: item,
            content_index: 0,
            transcript: text.to_owned(),
            confidence: None,
            in_progress: false,
            event_id: self.event_id(),
            usage: Some(TranscriptionUsage::Tokens(TokenUsage {
                input_tokens: 20,
                output_tokens: output.into(),
                input_audio_tokens: Some(20),
                ..TokenUsage::default()
            })),
        };
        self.after(now, lag, Incoming::TranscriptionCompleted(completed));
    }

    /// A response, from its `response.created` at `at` to its `response.done`, reading the
    /// conversation as it stands now.
    fn respond(
        &mut self,
        at: Ts,
        client_event_id: Option<String>,
        tool_choice: Option<ToolChoice>,
        instructions: Option<String>,
    ) -> u32 {
        let r = self.responses.len() as u32;
        let id = format!("resp_{}{}", self.ids, self.next_resp);
        self.next_resp += 1;
        self.responses.push(Response {
            id: id.clone(),
            request: client_event_id.clone(),
            instructions: instructions.or_else(|| self.config.instructions.clone()),
            read_items: self.items.len(),
            ..Response::default()
        });
        self.active = Some(r);
        self.put(at, Some(r), Incoming::ResponseCreated { response_id: id, client_event_id });

        let text = REPLIES[self.rng.usize(0..REPLIES.len())];
        let item = self.mint_item("msg");
        self.responses[r as usize].item = Some(item.clone());
        // Its place in the conversation is taken when the item is added (`pop`): a client item
        // arriving before then, a tool output or a commit, comes before it, as on the real server.
        let mut t = later(at, self.rng.u64(210..=730));
        let item_type = "message".to_owned();
        self.put(t, Some(r), Incoming::OutputItemAdded { item_id: item.clone(), item_type });
        let assistant = message(&item, Role::Assistant);
        self.put(t, Some(r), Incoming::ItemAdded { previous_item_id: None, item: assistant });
        // The first audio, after the item as measured; the content part and the
        // transcript's first words before it.
        let item_at = t;
        let first_audio = later(item_at, self.rng.u64(210..=400));
        t = later(item_at, 15).min(first_audio);
        let audio = self.config.audio;
        let part_type = if audio { "audio" } else { "text" }.to_owned();
        self.put(t, Some(r), Incoming::ContentPartAdded { item_id: item.clone(), part_type });

        // The words, as transcript (or text) deltas, and the audio they become.
        let per_char = self.rng.u64(40..=61) as f64;
        let audio_ms = ((text.chars().count() as f64 * per_char) as u64).max(DELTA_MS);
        let deltas = audio_ms.div_ceil(DELTA_MS) as usize;
        let words: Vec<&str> = text.split_inclusive(' ').collect();
        let delta = |delta: String| {
            let item_id = item.clone();
            if audio {
                Incoming::AudioTranscriptDelta { item_id, delta, start_time: None }
            } else {
                Incoming::TextDelta { item_id, delta }
            }
        };
        // The transcript's first words come with the content part, before any audio, as the
        // real server sends them: a response cut there has words and no audio.
        let mut sent_words = words.len().min(2);
        if sent_words > 0 {
            self.put(t, Some(r), delta(words[..sent_words].concat()));
        }
        for chunk in 0..deltas.max(1) {
            t = if chunk == 0 { first_audio } else { later(t, self.rng.u64(20..=160)) };
            // The transcript leads the audio it describes.
            let upto = (words.len() * (chunk + 1)).div_ceil(deltas.max(1));
            if upto > sent_words {
                self.put(t, Some(r), delta(words[sent_words..upto].concat()));
                sent_words = upto;
            }
            if audio {
                let chunk = u16::try_from(chunk).unwrap_or(u16::MAX);
                self.put_body(t, Some(r), Body::Audio { item: item.clone(), chunk });
            }
        }
        // Now and then the server restates its rate limits, past the last audio and before the
        // message closes, as the real one was caught doing.
        let (limits, of) = (1, 94);
        if self.rng.u64(0..of) < limits {
            self.put(t, Some(r), unhandled("rate_limits.updated"));
        }
        // A tool call now and then, when the session offers tools: after the words, as the
        // real server sends them.
        let choice = tool_choice.or_else(|| self.config.tool_choice.clone());
        let (calls, of) = match (choice, self.tool_calls) {
            (Some(ToolChoice::Forbidden), _) => (0, 1),
            (Some(ToolChoice::Required), _) => (1, 1),
            (_, Some(percent)) => (u64::from(percent), 100),
            (_, None) => (8, 94),
        };
        if !self.config.tools.is_empty() && self.rng.u64(0..of) < calls {
            let name = self.config.tools[self.rng.usize(0..self.config.tools.len())].clone();
            let fc = self.mint_item("fc");
            self.responses[r as usize].function_call = Some((fc.clone(), name.clone()));
            t = later(t, 30);
            let item_type = "function_call".to_owned();
            self.put(t, Some(r), Incoming::OutputItemAdded { item_id: fc.clone(), item_type });
            let call = function_call(&fc, &name, "");
            self.put(t, Some(r), Incoming::ItemAdded { previous_item_id: Some(item), item: call });
            self.put(t, Some(r), unhandled("response.function_call_arguments.delta"));
            let t2 = later(t, 30);
            self.message_done(t2, r);
            self.call_done(t2, r);
            t = t2;
        } else {
            t = later(t, self.rng.u64(28..=190) / 2);
            self.message_done(t, r);
        }
        let done_at = later(t, self.rng.u64(28..=190) / 2);
        self.end_body(done_at, r, GenerationStatus::Completed);
        r
    }

    /// The message's closing events: audio, transcript, part, item.
    fn message_done(&mut self, t: Ts, r: u32) {
        let Some(item) = self.responses[r as usize].item.clone() else { return };
        let mut closing = Vec::new();
        if self.config.audio {
            closing.push("response.output_audio.done");
            closing.push("response.output_audio_transcript.done");
        } else {
            closing.push("response.output_text.done");
        }
        closing.extend(["response.content_part.done", "conversation.item.done"]);
        for event_type in closing {
            self.put(t, Some(r), unhandled(event_type));
        }
        self.put(t, Some(r), Incoming::OutputItemDone { item: message(&item, Role::Assistant) });
    }

    /// The tool call's closing events: its arguments, item and output item.
    fn call_done(&mut self, t: Ts, r: u32) {
        let Some((fc, name)) = self.responses[r as usize].function_call.clone() else { return };
        self.put(t, Some(r), unhandled("response.function_call_arguments.done"));
        self.put(t, Some(r), unhandled("conversation.item.done"));
        self.put(t, Some(r), Incoming::OutputItemDone { item: function_call(&fc, &name, "{}") });
    }

    fn end_body(&mut self, t: Ts, r: u32, status: GenerationStatus) {
        let resp = &self.responses[r as usize];
        let failed = status == GenerationStatus::Failed;
        // What it read: the instructions and the items the conversation held when it was asked.
        let text = resp.instructions.as_ref().map_or(0, |i| i.len() as u64 / 4) + 40;
        let audio = 20 * resp.read_items as u64;
        let input = text + audio;
        let done = ResponseDone {
            response_id: resp.id.clone(),
            status,
            error_type: failed.then(|| "server_error".into()),
            error_code: failed.then(|| "server_error".into()),
            usage: TokenUsage {
                input_tokens: input,
                output_tokens: 40,
                input_audio_tokens: Some(audio),
                input_text_tokens: Some(text),
                output_audio_tokens: Some(30),
                output_text_tokens: Some(10),
                ..TokenUsage::default()
            },
        };
        self.put_body(t, Some(r), Body::Event(Box::new(Incoming::ResponseDone(done))));
    }

    /// Stop a response at `t`: what was still to come is withdrawn, and it closes as `status`.
    fn end(&mut self, t: Ts, r: u32, status: GenerationStatus) {
        let keep_created = !self.responses[r as usize].created;
        // What the model generated before the stop still streams out (the probes show audio
        // deltas arriving after a cancel): the message's opening and its deltas due before `t`.
        let streams = |p: &Pending| match &p.body {
            Body::Audio { .. } => true,
            Body::Event(event) => match event.as_ref() {
                Incoming::AudioTranscriptDelta { .. }
                | Incoming::TextDelta { .. }
                | Incoming::ContentPartAdded { .. }
                | Incoming::OutputItemAdded { .. }
                | Incoming::ItemAdded { .. } => true,
                Incoming::Unhandled { event_type } => {
                    event_type == "response.function_call_arguments.delta"
                }
                _ => false,
            },
        };
        let (mut opened, mut called) = (false, false);
        let withdrawn: Vec<(Ts, u64)> = self
            .pending
            .iter()
            .filter(|((due, _), p)| {
                if p.resp != Some(r) {
                    return false;
                }
                let created = matches!(&p.body, Body::Event(event) if matches!(event.as_ref(), Incoming::ResponseCreated { .. }));
                let kept = (*due < t && streams(p)) || (keep_created && created);
                if kept && let Body::Event(event) = &p.body && let Incoming::OutputItemAdded { item_type, .. } = event.as_ref() {
                    opened |= item_type == "message";
                    called |= item_type == "function_call";
                }
                !kept
            })
            .map(|(k, _)| *k)
            .collect();
        for k in withdrawn {
            self.pending.remove(&k);
        }
        // A response stopped before the client heard of it still opens before it closes.
        let created_due =
            self.pending.iter().filter(|(_, p)| p.resp == Some(r)).map(|((due, _), _)| *due).max();
        let t = created_due.map_or(t, |due| t.max(due));
        // An item the client was told of closes `incomplete`; one it was not, never appears.
        let announced = self.responses[r as usize].announced || opened;
        // One already complete stays so.
        if announced && !self.responses[r as usize].closed {
            self.message_done(t, r);
        } else if !announced && let Some(item) = self.responses[r as usize].item.take() {
            self.items.retain(|i| i.id != item);
        }
        // A call already complete stays so: its item is never done twice.
        let resp = &self.responses[r as usize];
        if !resp.call_closed && (resp.call_announced || called) {
            self.call_done(t, r);
        } else if !resp.call_closed {
            self.responses[r as usize].function_call = None;
        }
        self.end_body(t, r, status);
    }

    fn cancel(&mut self, t: Ts, r: u32) {
        // The conversation is free at once: a reply asked right after the cancel is taken, and
        // created before the cancelled one's `response.done`, as the real server does.
        if self.active == Some(r) {
            self.active = None;
        }
        self.end(t, r, GenerationStatus::Cancelled);
    }

    /// How long the server takes to answer a client frame: drawn from what it was measured
    /// taking, and never
    /// before its answer to an earlier frame -- it processes the client's frames in order, and
    /// the client's mirror of the conversation is built on that order.
    fn answer_lag(&mut self, now: Ts) -> u64 {
        self.answer_after(now, 88..=103)
    }

    /// As [`Self::answer_lag`], drawn from another measured range, in ms.
    fn answer_after(&mut self, now: Ts, range: std::ops::RangeInclusive<u64>) -> u64 {
        let lag = self.rng.u64(range);
        let at = later(now, lag).max(self.answered);
        self.answered = at;
        at.saturating_duration_since(now).as_millis() as u64
    }

    fn mint_item(&mut self, kind: &str) -> String {
        self.next_item += 1;
        format!("item_{kind}_{}{}", self.ids, self.next_item)
    }

    fn event_id(&mut self) -> String {
        self.next_event += 1;
        format!("event_{}{}", self.ids, self.next_event)
    }

    /// An answer `delay_ms` after `now`, of no response.
    fn after(&mut self, now: Ts, delay_ms: u64, event: Incoming) {
        self.put(later(now, delay_ms), None, event);
    }

    fn put(&mut self, t: Ts, resp: Option<u32>, event: Incoming) {
        self.put_body(t, resp, Body::Event(Box::new(event)));
    }

    fn put_body(&mut self, t: Ts, resp: Option<u32>, body: Body) {
        self.seq += 1;
        self.pending.insert((t, self.seq), Pending { resp, body });
    }
}

fn later(t: Ts, ms: u64) -> Ts {
    t.saturating_add(std::time::Duration::from_millis(ms))
}
