//! The OpenAI Realtime server, as the real one answers the protocol's edge cases
//! (`corpus/probes/`, `corpus/probe_realtime.py`), the corpus shows it behaving, and, where both
//! are silent, the docs say.
//!
//! Sans-IO: client frames go in with the instant they arrived ([`Server::client`]), what the
//! server does on its own accord goes in as an [`Exo`], and every server event comes out of
//! [`Server::pop`] once it is due. Each event's delay after its cause is drawn from the corpus
//! (`corpus/calibration.json`, [`crate::corpus`]), so a reply is created about 100 ms after its
//! `response.create`, its first item 300 ms later, its audio in 400 ms deltas faster than real
//! time, and `response.done` while the client is still playing most of it -- which is what made
//! a server onset over a finished generation the common barge-in, not a rare one. The server
//! answers the client's frames in the order they came.
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
//! The audio a response carries is tagged per delta ([`Tag::generated`]) and compressed: a
//! delta's PCM is 10 ms whatever its nominal length, as the TTS fake's chunks are; its nominal
//! length is what truncation is judged against.
use std::collections::BTreeMap;

use base64::Engine as _;
use fastrand::Rng;
use lk_agents_abi::Ts;
use serde_json::{Value, json};

use crate::{
    corpus::{draw_ms, share},
    fakes::SAMPLES_PER_CHUNK,
    input::{AGENT_TURNS, USER_TURNS},
    record::{Breach, Tag},
};

/// The error codes the server refuses a client event with, as the real one does
/// (`corpus/probes/`), and the words a speech that failed on one carries: such a speech failed on
/// the server's word.
pub const REFUSALS: &[&str] = &[
    "conversation_already_has_active_response",
    "Conversation already has an active response",
    "response_cancel_not_active",
    "response_cancel_not_found",
    "input_audio_buffer_commit_empty",
    "item_create_duplicate_item_id",
    "item_create_invalid_previous_item_id",
    "invalid_tool_call_id",
    "missing_required_parameter",
    "item_delete_invalid_item_id",
    "item_retrieve_invalid_item_id",
    "item_truncate_invalid_item_id",
    "unsupported_content_type",
    "cannot_update_voice",
    "invalid_value",
];

/// The nominal length of a full audio delta, as the corpus has nearly all of them.
const DELTA_MS: u64 = 400;

/// What the server is told to do of its own accord: the person's speech as its VAD hears it, and
/// the ways a real server goes wrong. Mirrors `RtEvent`, which is how a case says it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exo {
    SpeechStarted,
    SpeechStopped,
    /// A delta for a response nobody created.
    StrayDelta,
    /// The server opens a response on its own (`input_audio_buffer.timeout_triggered` and the
    /// like): no `client_event_id`.
    ServerResponse,
    /// The oldest response still generating finishes now, all of it at once.
    FinishNow,
    /// A server-side `error` event, the session staying open.
    ServerError,
    /// The response in progress fails: `response.done` with `status: failed`.
    ResponseFailed,
    /// The last user item's transcription fails.
    TranscriptionFailed,
}

/// Server-side VAD, as the session configured it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vad {
    Semantic,
    Server,
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
    tool_choice: Option<String>,
}

/// A server event and what delivering it means to the server's own bookkeeping.
#[derive(Clone, Debug)]
enum Body {
    Json(Value),
    /// An audio delta, built when it is delivered: its PCM names the generation that carries it.
    Audio { item: String, chunk: u16 },
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
    kind: String,
    role: Option<String>,
    call_id: Option<String>,
    /// Audio delivered for an assistant message, nominal ms: what truncation is judged against.
    audio_ms: u64,
}

impl Item {
    fn new(id: &str, kind: &str, role: Option<&str>, call_id: Option<&str>) -> Self {
        Item {
            id: id.to_owned(),
            kind: kind.to_owned(),
            role: role.map(str::to_owned),
            call_id: call_id.map(str::to_owned),
            audio_ms: 0,
        }
    }

    fn of(id: &str, item: &Value) -> Self {
        let s = |k: &str| item.get(k).and_then(Value::as_str);
        Item::new(id, s("type").unwrap_or("message"), s("role"), s("call_id"))
    }
}

#[derive(Clone, Debug, Default)]
struct Response {
    id: String,
    /// Asked for by the client's `response.create`.
    asked: bool,
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
    /// `response.done` scheduled: nothing more may be added.
    ending: bool,
    /// Delivered audio, nominal ms.
    audio_ms: u64,
    /// The generation's source id, from the hook at creation.
    source: Option<u32>,
    /// The tool call it makes, `(item, name)`.
    function_call: Option<(String, String)>,
    /// The call's `response.output_item.added` was delivered.
    call_announced: bool,
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

/// Hooks that do nothing: what the conformance check runs the server with.
pub struct NoHooks(pub u32);

impl Hooks for NoHooks {
    fn created(&mut self, _: &str) -> u32 {
        self.0 += 1;
        self.0
    }
    fn done(&mut self, _: u32) {}
    fn produced(&mut self, _: u32, _: u16) {}
    fn heard(&mut self, _: &[u8]) {}
    fn breach(&mut self, _: Breach) {}
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
    /// Text of the response being created next, chosen when it is opened.
    pub said: Vec<String>,
    /// How often a response calls a tool, in percent, when the session offers tools: the
    /// developer's prompt decides it. `None` is the corpus's rate.
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
            said: Vec::new(),
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
        let ev = self.event_id();
        self.at(now, None, json!({"type": "session.created", "event_id": ev, "session": {"type": "realtime", "object": "realtime.session"}}));
    }

    pub fn next_due(&self) -> Option<Ts> {
        self.pending.keys().next().map(|(t, _)| *t)
    }

    /// The next event due at `now`, delivered: its JSON frame.
    pub fn pop(&mut self, now: Ts, hooks: &mut dyn Hooks) -> Option<Value> {
        self.breathe(now);
        self.report(hooks);
        let (&key, _) = self.pending.iter().next().filter(|((t, _), _)| *t <= now)?;
        let Pending { resp, body } = self.pending.remove(&key).expect("just seen");
        let mut frame = match body {
            Body::Json(v) => v,
            Body::Audio { item, chunk } => {
                let r = resp.map(|r| &mut self.responses[r as usize]);
                let source = r.as_ref().and_then(|r| r.source).unwrap_or(0);
                if let Some(r) = r {
                    r.audio_ms += DELTA_MS;
                }
                if let Some(i) = self.items.iter_mut().find(|i| i.id == item) {
                    i.audio_ms += DELTA_MS;
                }
                self.voiced = true;
                hooks.produced(source, chunk);
                let pcm = Tag::generated(source, chunk).to_le_bytes().repeat(SAMPLES_PER_CHUNK);
                let resp_id = resp.map(|r| self.responses[r as usize].id.clone());
                json!({
                    "type": "response.output_audio.delta",
                    "event_id": self.event_id(),
                    "response_id": resp_id,
                    "item_id": item,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": base64::engine::general_purpose::STANDARD.encode(pcm),
                })
            }
        };
        if resp.is_some() && frame["type"] == "conversation.item.added" {
            frame["previous_item_id"] = json!(self.items.last().map(|i| i.id.clone()));
            let id = frame["item"]["id"].as_str().unwrap_or_default().to_owned();
            self.items.push(Item::of(&id, &frame["item"]));
        }
        let ty = frame["type"].as_str().unwrap_or_default();
        if ty == "input_audio_buffer.speech_started" {
            for r in [self.active, self.told_active].into_iter().flatten() {
                self.responses[r as usize].prompted = true;
            }
        }
        if matches!(ty, "input_audio_buffer.committed" | "input_audio_buffer.cleared") {
            self.untold_empties = self.untold_empties.saturating_sub(1);
        }
        if let Some(r) = resp {
            let r = r as usize;
            match ty {
                "response.created" => {
                    let id = self.responses[r].id.clone();
                    self.responses[r].source = Some(hooks.created(&id));
                    self.responses[r].created = true;
                    self.responses[r].told_created = Some(now);
                    self.told_active = Some(r as u32);
                }
                "response.output_item.added"
                    if frame["item"]["type"] == "message" =>
                {
                    self.responses[r].announced = true;
                }
                "response.output_item.added" if frame["item"]["type"] == "function_call" => {
                    self.responses[r].call_announced = true;
                }
                "response.output_item.done" if frame["item"]["type"] == "message" => {
                    self.responses[r].closed = true;
                }
                "response.done" => {
                    self.responses[r].finished = Some(now);
                    if let Some(source) = self.responses[r].source {
                        hooks.done(source);
                    }
                    if self.active == Some(r as u32) {
                        self.active = None;
                    }
                    if self.told_active == Some(r as u32) {
                        self.told_active = None;
                    }
                }
                _ => {}
            }
        }
        Some(frame)
    }

    // --- the client's frames -------------------------------------------------------------------

    /// One client frame, arrived at `now`; what it breached goes to the hooks.
    pub fn client(&mut self, now: Ts, frame: &Value, hooks: &mut dyn Hooks) {
        self.breathe(now);
        self.receive(now, frame, hooks);
        self.report(hooks);
    }

    /// The client's breath before `now` is over: an output that arrived after a reply asked in
    /// it, and no reply was asked after it, was read by none.
    fn breathe(&mut self, now: Ts) {
        self.unanswered.retain(|(.., at)| *at == now);
        let (over, open): (Vec<_>, Vec<_>) = self.early.drain(..).partition(|(.., at)| *at < now);
        self.early = open;
        for (response, call_id, _) in over {
            let request = self.responses.iter().find(|r| r.id == response).and_then(|r| r.request.clone());
            self.breaches.push(Breach::AskedBeforeOutput { response, request, call_id });
        }
    }

    fn report(&mut self, hooks: &mut dyn Hooks) {
        for breach in self.breaches.drain(..) {
            hooks.breach(breach);
        }
    }

    fn receive(&mut self, now: Ts, frame: &Value, hooks: &mut dyn Hooks) {
        let ty = frame["type"].as_str().unwrap_or_default();
        match ty {
            "session.update" => {
                let session = &frame["session"];
                // The voice is fixed once assistant audio exists; the whole update is refused.
                let voice = session["audio"]["output"]["voice"].as_str();
                if self.voiced && voice.is_some_and(|v| Some(v) != self.config.voice.as_deref()) {
                    let why = "Cannot update a conversation's voice if assistant audio is present.";
                    return self.refuse(now, "cannot_update_voice", why, false, true, frame);
                }
                self.configure(session);
                let ev = self.event_id();
                self.after(now, 5, None, json!({"type": "session.updated", "event_id": ev, "session": session.clone()}));
            }
            "input_audio_buffer.append" => {
                let pcm = frame["audio"]
                    .as_str()
                    .and_then(|a| base64::engine::general_purpose::STANDARD.decode(a).ok())
                    .unwrap_or_default();
                let ms = (pcm.len() as u64 / 2) * 1000 / super::super::SAMPLE_RATE as u64;
                self.buffer_ms += ms;
                self.heard_ms += ms;
                hooks.heard(&pcm);
            }
            "input_audio_buffer.commit" => {
                // At least 100 ms, as the server says.
                if self.buffer_ms < 100 {
                    // The client could count what it appended unless the server emptied the
                    // buffer on its own, or an emptying is still on its way to it.
                    let knowing = self.own_empty && self.untold_empties == 0;
                    let why = "Error committing input audio buffer: buffer too small. Expected at least 100ms of audio.";
                    self.refuse(now, "input_audio_buffer_commit_empty", why, true, knowing, frame);
                } else {
                    let item = self.speaking.take().unwrap_or_else(|| self.mint_item("user"));
                    let lag = self.answer_lag(now);
                    self.own_empty = true;
                    self.commit(later(now, lag), item);
                }
            }
            "input_audio_buffer.clear" => {
                self.buffer_ms = 0;
                self.own_empty = true;
                self.untold_empties += 1;
                let ev = self.event_id();
                self.after(now, 90, None, json!({"type": "input_audio_buffer.cleared", "event_id": ev}));
            }
            "response.create" => {
                if self.active.is_some() {
                    let knowing = self
                        .told_active
                        .and_then(|r| self.responses[r as usize].told_created)
                        .is_some_and(|at| at < now);
                    let why = "Conversation already has an active response in progress. Wait until the response is finished before creating a new one.";
                    return self.refuse(now, "conversation_already_has_active_response", why, true, knowing, frame);
                }
                // Asked over a call whose output is not in yet: it reads the call without it.
                let tail = self.items.last().filter(|i| i.kind == "function_call");
                let unanswered = tail.and_then(|i| i.call_id.clone());
                let request = &frame["response"];
                let client_event_id = request["metadata"]["client_event_id"].as_str().map(str::to_owned);
                // Answered in the order asked: a reply cancelled at once still opens before the
                // one asked after it.
                let delay = self.answer_after(now, "created_after_create");
                // A response's own `tool_choice` and `instructions` override the session's.
                let choice = request["tool_choice"].as_str().map(str::to_owned);
                let instructions = request["instructions"].as_str().map(str::to_owned);
                let r = self.respond(later(now, delay), client_event_id, choice, instructions);
                self.responses[r as usize].asked = true;
                // A reply asked after the outputs reads them.
                self.early.retain(|(.., at)| *at != now);
                if let Some(call) = unanswered {
                    self.unanswered.push((self.responses[r as usize].id.clone(), call, now));
                }
            }
            "response.cancel" => {
                let named = frame["response_id"].as_str();
                if let Some(id) = named
                    && !self.responses.iter().any(|r| r.id == id)
                {
                    let why = "Response with id not found";
                    return self.refuse(now, "response_cancel_not_found", why, false, true, frame);
                }
                let target = self.active.filter(|r| named.is_none_or(|id| self.responses[*r as usize].id == id));
                match target {
                    Some(r) => {
                        // The server's answer to the person's turn, nobody has heard a word of, no
                        // onset told over it: the client cancels it on nothing the server saw.
                        let resp = &self.responses[r as usize];
                        if resp.turn && !resp.prompted && resp.audio_ms == 0 {
                            let (response, generation) = (resp.id.clone(), resp.source);
                            self.breaches.push(Breach::CancelledUnprompted { response, generation });
                        }
                        // What it had generated still streams out first: a reply asked in the same
                        // breath is created before this one's `response.done`
                        // (probe-cancel_then_create).
                        let lag = self.answer_after(now, "done_after_cancel");
                        self.cancel(later(now, lag), r, "client_cancelled");
                    }
                    None => {
                        // Knowing when the client had been told the end of the response it names
                        // (of every one, unnamed); a response it asked for and has not heard of,
                        // or one whose end is on its way, makes the cancel a race.
                        let knowing = match named {
                            Some(id) => self.responses.iter().any(|r| r.id == id && r.finished.is_some_and(|at| at < now)),
                            None => self.responses.iter().all(|r| r.finished.is_some_and(|at| at < now)),
                        };
                        let why = "Cancellation failed: no active response found";
                        self.refuse(now, "response_cancel_not_active", why, true, knowing, frame);
                    }
                }
            }
            "conversation.item.create" => {
                let mut item = frame["item"].clone();
                let id = item["id"].as_str().map(str::to_owned).unwrap_or_else(|| self.mint_item("item"));
                let given = frame.get("previous_item_id").and_then(Value::as_str);
                let previous = match given {
                    Some("root") => None,
                    Some(p) => Some(p.to_owned()),
                    None => self.items.last().map(|i| i.id.clone()),
                };
                let call_id = item["call_id"].as_str().map(str::to_owned);
                // An assistant message's content is text: an audio part has none.
                let audio_reply = item["role"] == "assistant"
                    && item["content"].as_array().is_some_and(|c| c.iter().any(|p| p["text"].is_null() && p["type"] != "output_text"));
                let refusal = if self.position(&id).is_some() {
                    Some(("item_create_duplicate_item_id", "Error adding item: an item with this id already exists.", true))
                } else if previous.as_deref().is_some_and(|p| self.position(p).is_none()) {
                    Some(("item_create_invalid_previous_item_id", "Error adding item: the previous item does not exist.", true))
                } else if item["type"] == "function_call_output"
                    && !self.items.iter().any(|i| i.kind == "function_call" && i.call_id == call_id)
                {
                    Some(("invalid_tool_call_id", "Tool call ID not found in conversation.", false))
                } else if audio_reply {
                    Some(("missing_required_parameter", "Missing required parameter: 'item.content[0].text'.", true))
                } else {
                    None
                };
                if let Some((code, why, cited)) = refusal {
                    return self.refuse(now, code, why, cited, true, frame);
                }
                let at = match &previous {
                    _ if given.is_none() => self.items.len(),
                    Some(p) => self.position(p).map_or(self.items.len(), |i| i + 1),
                    None => 0,
                };
                self.items.insert(at, Item::of(&id, &item));
                if item["type"] == "function_call_output" {
                    let call = call_id.unwrap_or_default();
                    for (response, c, at) in &self.unanswered {
                        if *c == call && *at == now {
                            self.early.push((response.clone(), call.clone(), now));
                        }
                    }
                } else {
                    item["status"] = json!("completed");
                }
                item["id"] = json!(id);
                let lag = self.answer_lag(now);
                let (e1, e2) = (self.event_id(), self.event_id());
                self.after(now, lag, None, json!({"type": "conversation.item.added", "event_id": e1, "previous_item_id": previous, "item": item.clone()}));
                self.after(now, lag, None, json!({"type": "conversation.item.done", "event_id": e2, "previous_item_id": previous, "item": item}));
            }
            "conversation.item.delete" => {
                let id = frame["item_id"].as_str().unwrap_or_default().to_owned();
                match self.position(&id) {
                    Some(at) => {
                        self.items.remove(at);
                        let ev = self.event_id();
                        let lag = self.answer_lag(now);
                        self.after(now, lag, None, json!({"type": "conversation.item.deleted", "event_id": ev, "item_id": id}));
                    }
                    None => {
                        let why = "Error deleting item: the item does not exist.";
                        self.refuse(now, "item_delete_invalid_item_id", why, true, true, frame);
                    }
                }
            }
            "conversation.item.retrieve" => {
                let id = frame["item_id"].as_str().unwrap_or_default().to_owned();
                match self.position(&id).map(|at| self.items[at].clone()) {
                    Some(item) => {
                        let ev = self.event_id();
                        let lag = self.answer_lag(now);
                        self.after(now, lag, None, json!({"type": "conversation.item.retrieved", "event_id": ev, "item": {"id": item.id, "type": item.kind, "role": item.role, "call_id": item.call_id, "status": "completed"}}));
                    }
                    None => {
                        let why = "Error retrieving item: the item does not exist.";
                        self.refuse(now, "item_retrieve_invalid_item_id", why, true, true, frame);
                    }
                }
            }
            "conversation.item.truncate" => {
                let id = frame["item_id"].as_str().unwrap_or_default().to_owned();
                let end = frame["audio_end_ms"].as_u64().unwrap_or(0);
                let Some(at) = self.position(&id) else {
                    let why = "Item with item_id not found";
                    return self.refuse(now, "item_truncate_invalid_item_id", why, true, true, frame);
                };
                if self.items[at].role.as_deref() != Some("assistant") {
                    let why = "Only model output audio messages can be truncated";
                    return self.refuse(now, "unsupported_content_type", why, false, true, frame);
                }
                // Within the audio it holds, a millisecond of rounding allowed; what it holds is
                // cut there, and a later truncation is judged against the cut.
                let audio = self.items[at].audio_ms;
                if end > audio + 1 {
                    let why = "Audio content is already shorter than audio_end_ms";
                    return self.refuse(now, "invalid_value", why, false, true, frame);
                }
                self.items[at].audio_ms = end.min(audio);
                let ev = self.event_id();
                let lag = self.answer_lag(now);
                self.after(now, lag, None, json!({"type": "conversation.item.truncated", "event_id": ev, "item_id": id, "content_index": 0, "audio_end_ms": end}));
            }
            "session.close" | "transcription_session.update" => {}
            // Any other type, `output_audio_buffer.clear` (WebRTC only) among them.
            _ => {
                let why = "Invalid value: not a supported client event.";
                self.refuse(now, "invalid_value", why, true, true, frame);
            }
        }
    }

    fn position(&self, id: &str) -> Option<usize> {
        self.items.iter().position(|i| i.id == id)
    }

    fn configure(&mut self, session: &Value) {
        let input = &session["audio"]["input"];
        if let Some(td) = input.get("turn_detection") {
            self.config.vad = match td["type"].as_str() {
                Some("semantic_vad") => Some(Vad::Semantic),
                Some("server_vad") => Some(Vad::Server),
                _ => None,
            };
            self.config.create_response = td["create_response"].as_bool().unwrap_or(true);
            self.config.interrupt_response = td["interrupt_response"].as_bool().unwrap_or(true);
        }
        if let Some(t) = input.get("transcription") {
            self.config.transcription = !t.is_null();
        }
        if let Some(m) = session.get("output_modalities").and_then(Value::as_array) {
            self.config.audio = m.iter().any(|v| v == "audio");
        }
        if let Some(choice) = session.get("tool_choice").and_then(Value::as_str) {
            self.config.tool_choice = Some(choice.to_owned());
        }
        if let Some(tools) = session.get("tools").and_then(Value::as_array) {
            self.config.tools =
                tools.iter().filter_map(|t| t["name"].as_str().map(str::to_owned)).collect();
        }
        if let Some(instructions) = session.get("instructions").and_then(Value::as_str) {
            self.config.instructions = Some(instructions.to_owned());
        }
        if let Some(voice) = session["audio"]["output"]["voice"].as_str() {
            self.config.voice = Some(voice.to_owned());
        }
    }

    // --- what the server does of its own accord ------------------------------------------------

    /// True when a real server could do this now.
    pub fn exo(&mut self, now: Ts, what: Exo) -> bool {
        match what {
            Exo::SpeechStarted => {
                // The VAD hears only audio the client appended since the buffer was last emptied.
                if self.config.vad.is_none() || self.speaking.is_some() || self.buffer_ms == 0 {
                    return false;
                }
                let item = self.mint_item("user");
                self.speaking = Some(item.clone());
                let ev = self.event_id();
                // The prefix padding, within what the buffer holds.
                let start = self.heard_ms.saturating_sub(self.buffer_ms.min(300));
                self.at(now, None, json!({"type": "input_audio_buffer.speech_started", "event_id": ev, "item_id": item, "audio_start_ms": start}));
                if self.config.interrupt_response
                    && let Some(r) = self.active
                {
                    let lag = draw_ms(&mut self.rng, &["realtime", "cancel_after_speech_started"]);
                    self.cancel(later(now, lag), r, "turn_detected");
                }
                true
            }
            Exo::SpeechStopped => {
                let Some(item) = self.speaking.take() else { return false };
                let ev = self.event_id();
                self.at(now, None, json!({"type": "input_audio_buffer.speech_stopped", "event_id": ev, "item_id": item, "audio_end_ms": self.heard_ms}));
                self.own_empty = false;
                self.commit(now, item);
                if self.config.create_response && self.active.is_none() {
                    let delay = draw_ms(&mut self.rng, &["realtime", "created_after_stopped"]);
                    let r = self.respond(later(now, delay), None, None, None);
                    self.responses[r as usize].turn = true;
                }
                true
            }
            Exo::StrayDelta => {
                let ev = self.event_id();
                self.at(now, None, json!({"type": "response.output_audio_transcript.delta", "event_id": ev, "response_id": "resp_stray", "item_id": "item_stray", "output_index": 0, "content_index": 0, "delta": "..."}));
                true
            }
            Exo::ServerResponse => {
                if self.active.is_some() {
                    return false;
                }
                self.respond(now, None, None, None);
                true
            }
            Exo::FinishNow => {
                let Some(r) = self.active else { return false };
                let keys: Vec<(Ts, u64)> = self
                    .pending
                    .iter()
                    .filter(|(_, p)| p.resp == Some(r))
                    .map(|(k, _)| *k)
                    .collect();
                // Everything still to come of it, at once and in order.
                let moved: Vec<Pending> = keys.iter().filter_map(|k| self.pending.remove(k)).collect();
                for p in moved {
                    self.seq += 1;
                    self.pending.insert((now, self.seq), p);
                }
                true
            }
            Exo::ServerError => {
                let ev = self.event_id();
                self.at(now, None, json!({"type": "error", "event_id": ev, "error": {"type": "server_error", "code": null, "message": "The server had an error while processing your request.", "event_id": null}}));
                true
            }
            Exo::ResponseFailed => {
                let Some(r) = self.active else { return false };
                self.end(now, r, "failed", None);
                true
            }
            Exo::TranscriptionFailed => {
                let Some(item) = self.last_user.clone().filter(|_| self.config.transcription) else {
                    return false;
                };
                let ev = self.event_id();
                self.at(now, None, json!({"type": "conversation.item.input_audio_transcription.failed", "event_id": ev, "item_id": item, "content_index": 0, "error": {"type": "transcription_error", "code": "audio_unintelligible", "message": "", "param": null}}));
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
        self.items.push(Item::new(&item, "message", Some("user"), None));
        self.last_user = Some(item.clone());
        let user = json!({"id": item, "type": "message", "role": "user", "status": "completed", "content": [{"type": "input_audio", "transcript": null}]});
        let (e1, e2, e3) = (self.event_id(), self.event_id(), self.event_id());
        self.after(now, 0, None, json!({"type": "input_audio_buffer.committed", "event_id": e1, "item_id": item, "previous_item_id": previous}));
        self.after(now, 1, None, json!({"type": "conversation.item.added", "event_id": e2, "previous_item_id": previous, "item": user.clone()}));
        self.after(now, 2, None, json!({"type": "conversation.item.done", "event_id": e3, "previous_item_id": previous, "item": user}));
        if self.config.transcription {
            let text = USER_TURNS[self.rng.usize(0..USER_TURNS.len() - 1)];
            let lag = draw_ms(&mut self.rng, &["realtime", "transcription_after_committed"]);
            let words: Vec<&str> = text.split(' ').collect();
            // About one delta a word, streamed over the last fifth of the wait.
            for (i, word) in words.iter().enumerate() {
                let at = lag * 4 / 5 + (lag / 5) * i as u64 / words.len().max(1) as u64;
                let delta = if i == 0 { (*word).to_owned() } else { format!(" {word}") };
                let ev = self.event_id();
                self.after(now, at, None, json!({"type": "conversation.item.input_audio_transcription.delta", "event_id": ev, "item_id": item, "content_index": 0, "delta": delta}));
            }
            let ev = self.event_id();
            self.after(now, lag, None, json!({"type": "conversation.item.input_audio_transcription.completed", "event_id": ev, "item_id": item, "content_index": 0, "transcript": text, "usage": {"type": "tokens", "input_tokens": 20, "output_tokens": words.len(), "total_tokens": 20 + words.len(), "input_token_details": {"audio_tokens": 20, "text_tokens": 0}}}));
        }
    }

    /// A response, from its `response.created` at `at` to its `response.done`, reading the
    /// conversation as it stands now.
    fn respond(
        &mut self,
        at: Ts,
        client_event_id: Option<String>,
        tool_choice: Option<String>,
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
        let metadata = client_event_id.map(|c| json!({"client_event_id": c}));
        let ev = self.event_id();
        self.put(at, Some(r), Body::Json(json!({"type": "response.created", "event_id": ev, "response": {"id": id, "object": "realtime.response", "status": "in_progress", "output": [], "metadata": metadata}})));

        let text = AGENT_TURNS[self.rng.usize(0..AGENT_TURNS.len() - 1)].to_owned();
        self.said.push(text.clone());
        let item = self.mint_item("msg");
        self.responses[r as usize].item = Some(item.clone());
        // Its place in the conversation is taken when the item is added (`pop`): a client item
        // arriving before then, a tool output or a commit, comes before it, as on the real server.
        let previous = Value::Null;
        let mut t = later(at, draw_ms(&mut self.rng, &["realtime", "item_after_created"]));
        let message = |status: &str| json!({"id": item, "type": "message", "role": "assistant", "status": status, "content": []});
        let (e1, e2, e3) = (self.event_id(), self.event_id(), self.event_id());
        self.put(t, Some(r), Body::Json(json!({"type": "response.output_item.added", "event_id": e1, "response_id": id, "output_index": 0, "item": message("in_progress")})));
        self.put(t, Some(r), Body::Json(json!({"type": "conversation.item.added", "event_id": e2, "previous_item_id": previous, "item": message("in_progress")})));
        // The first audio, after the item as the corpus measured it; the content part and the
        // transcript's first words before it.
        let item_at = t;
        let first_audio = later(item_at, draw_ms(&mut self.rng, &["realtime", "first_audio_after_item"]));
        t = later(item_at, 15).min(first_audio);
        let part = if self.config.audio { "audio" } else { "text" };
        self.put(t, Some(r), Body::Json(json!({"type": "response.content_part.added", "event_id": e3, "response_id": id, "item_id": item, "output_index": 0, "content_index": 0, "part": {"type": part}})));

        // The words, as transcript (or text) deltas, and the audio they become.
        let per_char = crate::corpus::draw(&mut self.rng, &["realtime", "audio_ms_per_char"]);
        let audio_ms = ((text.chars().count() as f64 * per_char) as u64).max(DELTA_MS);
        let deltas = audio_ms.div_ceil(DELTA_MS) as usize;
        let words: Vec<&str> = text.split_inclusive(' ').collect();
        let delta_kind = if self.config.audio { "response.output_audio_transcript.delta" } else { "response.output_text.delta" };
        let mut first = true;
        // The transcript's first words come with the content part, before any audio, as the
        // corpus has them: a response cut there has words and no audio.
        let mut sent_words = words.len().min(2);
        if sent_words > 0 {
            let piece: String = words[..sent_words].concat();
            let ev = self.event_id();
            self.put(t, Some(r), Body::Json(json!({"type": delta_kind, "event_id": ev, "response_id": id, "item_id": item, "output_index": 0, "content_index": 0, "delta": piece})));
        }
        for chunk in 0..deltas.max(1) {
            if first {
                t = first_audio;
                first = false;
            } else {
                t = later(t, draw_ms(&mut self.rng, &["realtime", "audio_gap"]));
            }
            // The transcript leads the audio it describes.
            let upto = (words.len() * (chunk + 1)).div_ceil(deltas.max(1));
            if upto > sent_words {
                let piece: String = words[sent_words..upto].concat();
                sent_words = upto;
                let ev = self.event_id();
                self.put(t, Some(r), Body::Json(json!({"type": delta_kind, "event_id": ev, "response_id": id, "item_id": item, "output_index": 0, "content_index": 0, "delta": piece})));
            }
            if self.config.audio {
                self.put(t, Some(r), Body::Audio { item: item.clone(), chunk: chunk.min(Tag::MAX_CHUNK as usize) as u16 });
            }
        }
        // Now and then the server restates its rate limits, past the last audio and before the
        // message closes, as the corpus caught it doing.
        let (limits, of) = share(&["realtime", "rate_limits_share"]);
        if self.rng.u64(0..of) < limits {
            let ev = self.event_id();
            self.put(t, Some(r), Body::Json(json!({"type": "rate_limits.updated", "event_id": ev, "rate_limits": [{"name": "tokens", "limit": 40000, "remaining": 39000, "reset_seconds": 1.5}]})));
        }
        // A tool call now and then, when the session offers tools: after the words, as the
        // corpus has them.
        let (calls, of) = match (tool_choice.or_else(|| self.config.tool_choice.clone()).as_deref(), self.tool_calls) {
            (Some("none"), _) => (0, 1),
            (Some("required"), _) => (1, 1),
            (_, Some(percent)) => (u64::from(percent), 100),
            (_, None) => share(&["realtime", "function_call_share"]),
        };
        if !self.config.tools.is_empty() && self.rng.u64(0..of) < calls {
            let name = self.config.tools[self.rng.usize(0..self.config.tools.len())].clone();
            let fc = self.mint_item("fc");
            self.responses[r as usize].function_call = Some((fc.clone(), name.clone()));
            t = later(t, 30);
            let call = json!({"id": fc, "type": "function_call", "status": "in_progress", "name": name, "call_id": format!("call_{fc}"), "arguments": ""});
            let (e1, e2, e3) = (self.event_id(), self.event_id(), self.event_id());
            self.put(t, Some(r), Body::Json(json!({"type": "response.output_item.added", "event_id": e1, "response_id": id, "output_index": 1, "item": call.clone()})));
            self.put(t, Some(r), Body::Json(json!({"type": "conversation.item.added", "event_id": e2, "previous_item_id": item, "item": call})));
            self.put(t, Some(r), Body::Json(json!({"type": "response.function_call_arguments.delta", "event_id": e3, "response_id": id, "item_id": fc, "output_index": 1, "call_id": format!("call_{fc}"), "delta": "{}"})));
            let t2 = later(t, 30);
            self.message_done(t2, r, "completed");
            self.call_done(t2, r, "completed");
            t = t2;
        } else {
            t = later(t, draw_ms(&mut self.rng, &["realtime", "done_after_last_audio"]) / 2);
            self.message_done(t, r, "completed");
        }
        let done_at = later(t, draw_ms(&mut self.rng, &["realtime", "done_after_last_audio"]) / 2);
        self.end_body(done_at, r, "completed", None);
        self.responses[r as usize].ending = true;
        r
    }

    /// The message's closing events: audio, transcript, part, item.
    fn message_done(&mut self, t: Ts, r: u32, status: &str) {
        let resp = &self.responses[r as usize];
        let (id, Some(item)) = (resp.id.clone(), resp.item.clone()) else { return };
        let previous = self.position(&item).and_then(|at| at.checked_sub(1)).map(|at| self.items[at].id.clone());
        let audio = self.config.audio;
        let mut events = Vec::new();
        if audio {
            events.push(json!({"type": "response.output_audio.done", "response_id": id, "item_id": item, "output_index": 0, "content_index": 0}));
            events.push(json!({"type": "response.output_audio_transcript.done", "response_id": id, "item_id": item, "output_index": 0, "content_index": 0}));
        } else {
            events.push(json!({"type": "response.output_text.done", "response_id": id, "item_id": item, "output_index": 0, "content_index": 0}));
        }
        let part = if audio { "audio" } else { "text" };
        events.push(json!({"type": "response.content_part.done", "response_id": id, "item_id": item, "output_index": 0, "content_index": 0, "part": {"type": part}}));
        let message = json!({"id": item, "type": "message", "role": "assistant", "status": status, "content": []});
        events.push(json!({"type": "conversation.item.done", "previous_item_id": previous, "item": message.clone()}));
        events.push(json!({"type": "response.output_item.done", "response_id": id, "output_index": 0, "item": message}));
        for mut ev in events {
            ev["event_id"] = json!(self.event_id());
            self.put(t, Some(r), Body::Json(ev));
        }
    }

    /// The tool call's closing events: its arguments, item and output item.
    fn call_done(&mut self, t: Ts, r: u32, status: &str) {
        let resp = &self.responses[r as usize];
        let (id, message) = (resp.id.clone(), resp.item.clone());
        let Some((fc, name)) = resp.function_call.clone() else { return };
        let call_id = format!("call_{fc}");
        let call = json!({"id": fc, "type": "function_call", "status": status, "name": name, "call_id": call_id, "arguments": "{}"});
        let (e1, e2, e3) = (self.event_id(), self.event_id(), self.event_id());
        self.put(t, Some(r), Body::Json(json!({"type": "response.function_call_arguments.done", "event_id": e1, "response_id": id, "item_id": fc, "output_index": 1, "call_id": call_id, "name": name, "arguments": "{}"})));
        self.put(t, Some(r), Body::Json(json!({"type": "conversation.item.done", "event_id": e2, "previous_item_id": message, "item": call.clone()})));
        self.put(t, Some(r), Body::Json(json!({"type": "response.output_item.done", "event_id": e3, "response_id": id, "output_index": 1, "item": call})));
    }

    fn end_body(&mut self, t: Ts, r: u32, status: &str, reason: Option<&str>) {
        let resp = &self.responses[r as usize];
        let mut outputs = Vec::new();
        if resp.announced || status == "completed" {
            outputs.push(json!({"type": "message", "id": resp.item}));
        }
        // A call cut short is still among the outputs, `incomplete` (probe-cancel_mid_audio).
        if let Some((fc, name)) = &resp.function_call {
            let status = if status == "completed" { "completed" } else { "incomplete" };
            outputs.push(json!({"type": "function_call", "id": fc, "status": status, "call_id": format!("call_{fc}"), "name": name, "arguments": "{}"}));
        }
        let details = match (status, reason) {
            ("failed", _) => json!({"type": "failed", "error": {"type": "server_error", "code": "server_error"}}),
            (_, Some(reason)) => json!({"type": status, "reason": reason}),
            _ => Value::Null,
        };
        let id = resp.id.clone();
        // What it read: the instructions and the items the conversation held when it was asked.
        let text = resp.instructions.as_ref().map_or(0, |i| i.len() as u64 / 4) + 40;
        let audio = 20 * resp.read_items as u64;
        let input = text + audio;
        let ev = self.event_id();
        self.put(t, Some(r), Body::Json(json!({"type": "response.done", "event_id": ev, "response": {"id": id, "object": "realtime.response", "status": status, "status_details": details, "output": outputs, "usage": {"total_tokens": input + 40, "input_tokens": input, "output_tokens": 40, "input_token_details": {"text_tokens": text, "audio_tokens": audio, "cached_tokens": 0}, "output_token_details": {"text_tokens": 10, "audio_tokens": 30}}}})));
    }

    /// Stop a response at `t`: what was still to come is withdrawn, and it closes as `status`.
    fn end(&mut self, t: Ts, r: u32, status: &str, reason: Option<&str>) {
        let keep_created = !self.responses[r as usize].created;
        // What the model generated before the stop still streams out (the probes show audio
        // deltas arriving after a cancel): the message's opening and its deltas due before `t`.
        let streams = |p: &Pending| match &p.body {
            Body::Audio { .. } => true,
            Body::Json(v) => {
                let ty = v["type"].as_str().unwrap_or_default();
                ty.ends_with("transcript.delta")
                    || ty == "response.output_text.delta"
                    || ty == "response.content_part.added"
                    || ty == "response.function_call_arguments.delta"
                    || matches!(ty, "response.output_item.added" | "conversation.item.added")
            }
        };
        let (mut opened, mut called) = (false, false);
        let withdrawn: Vec<(Ts, u64)> = self
            .pending
            .iter()
            .filter(|((due, _), p)| {
                if p.resp != Some(r) {
                    return false;
                }
                let kept = (*due < t && streams(p))
                    || (keep_created
                        && matches!(&p.body, Body::Json(v) if v["type"] == "response.created"));
                if kept && let Body::Json(v) = &p.body && v["type"] == "response.output_item.added" {
                    opened |= v["item"]["type"] == "message";
                    called |= v["item"]["type"] == "function_call";
                }
                !kept
            })
            .map(|(k, _)| *k)
            .collect();
        for k in withdrawn {
            self.pending.remove(&k);
        }
        // A response stopped before the client heard of it still opens before it closes.
        let created_due = self
            .pending
            .iter()
            .filter(|(_, p)| p.resp == Some(r))
            .map(|((due, _), _)| *due)
            .max();
        let t = created_due.map_or(t, |due| t.max(due));
        // An item the client was told of closes `incomplete`; one it was not, never appears.
        let announced = self.responses[r as usize].announced || opened;
        // One already complete stays so.
        if announced && !self.responses[r as usize].closed {
            self.message_done(t, r, "incomplete");
        } else if !announced && let Some(item) = self.responses[r as usize].item.take() {
            self.items.retain(|i| i.id != item);
        }
        if self.responses[r as usize].call_announced || called {
            self.call_done(t, r, "incomplete");
        } else {
            self.responses[r as usize].function_call = None;
        }
        self.end_body(t, r, status, reason);
        self.responses[r as usize].ending = true;
    }

    fn cancel(&mut self, t: Ts, r: u32, reason: &str) {
        // The conversation is free at once: a reply asked right after the cancel is taken, and
        // created before the cancelled one's `response.done` (probe-cancel_then_create).
        if self.active == Some(r) {
            self.active = None;
        }
        self.end(t, r, "cancelled", Some(reason));
    }

    /// The client's frame is refused with an `error` event, and the breach is noted.
    /// `cited`: the error names the client event it refuses (`error.event_id`), as the real
    /// server does for most refusals and not for some (`corpus/probes/probe-refusals.json`).
    fn refuse(
        &mut self,
        now: Ts,
        code: &'static str,
        why: &'static str,
        cited: bool,
        knowing: bool,
        frame: &Value,
    ) {
        let event = frame["type"].as_str().unwrap_or_default().to_owned();
        self.breaches.push(Breach::Refused { event, code, why, knowing });
        let lag = self.answer_lag(now);
        let ev = self.event_id();
        let cause = frame.get("event_id").cloned().filter(|_| cited).unwrap_or(Value::Null);
        self.after(now, lag, None, json!({"type": "error", "event_id": ev, "error": {"type": "invalid_request_error", "code": code, "message": why, "param": null, "event_id": cause}}));
    }

    /// How long the server takes to answer a client frame: drawn from the corpus, and never
    /// before its answer to an earlier frame -- it processes the client's frames in order, and
    /// the client's mirror of the conversation is built on that order.
    fn answer_lag(&mut self, now: Ts) -> u64 {
        self.answer_after(now, "error_after_client")
    }

    /// As [`Self::answer_lag`], drawn from another of the corpus's quantities.
    fn answer_after(&mut self, now: Ts, quantity: &str) -> u64 {
        let lag = draw_ms(&mut self.rng, &["realtime", quantity]);
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

    fn at(&mut self, t: Ts, resp: Option<u32>, v: Value) {
        self.put(t, resp, Body::Json(v));
    }

    fn after(&mut self, now: Ts, delay_ms: u64, resp: Option<u32>, v: Value) {
        self.put(later(now, delay_ms), resp, Body::Json(v));
    }

    fn put(&mut self, t: Ts, resp: Option<u32>, body: Body) {
        self.seq += 1;
        self.pending.insert((t, self.seq), Pending { resp, body });
    }

    /// Whether a response is being generated.
    pub fn generating(&self) -> bool {
        self.active.is_some()
    }
}

fn later(t: Ts, ms: u64) -> Ts {
    t.saturating_add(std::time::Duration::from_millis(ms))
}
