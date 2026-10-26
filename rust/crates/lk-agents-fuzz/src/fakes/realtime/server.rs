//! The OpenAI Realtime server, as the docs specify it and the corpus shows it behaving.
//!
//! Sans-IO: client frames go in with the instant they arrived ([`Server::client`]), what the
//! server does on its own accord goes in as an [`Exo`], and every server event comes out of
//! [`Server::pop`] once it is due. Each event's delay after its cause is drawn from the corpus
//! (`corpus/calibration.json`, [`crate::corpus`]), so a reply is created about 100 ms after its
//! `response.create`, its first item 300 ms later, its audio in 400 ms deltas faster than real
//! time, and `response.done` while the client is still playing most of it -- which is what made
//! a server onset over a finished generation the common barge-in, not a rare one.
//!
//! What it does, by the documented semantics (developers.openai.com, realtime server events):
//! `session.created` first; `session.updated` per `session.update`; under server VAD a
//! `speech_started` cancels an in-progress response when `interrupt_response` (its
//! `response.done` is `cancelled`/`turn_detected`, its message `incomplete`) and a
//! `speech_stopped` commits the buffer (`committed`, then the user item) and, with
//! `create_response`, opens a response; the transcription completes asynchronously, before or
//! after the response's events. `response.cancel` with nothing active and a truncation past the
//! audio are the errors the server really sends (`response_cancel_not_active`, `invalid_value`),
//! a second `response.create` over an active one is refused, and a commit of an empty buffer too.
//!
//! The audio a response carries is tagged per delta ([`Tag::generated`]) and compressed: a
//! delta's PCM is 10 ms whatever its nominal length, as the TTS fake's chunks are; its nominal
//! length is what truncation is judged against.
use std::collections::{BTreeMap, HashMap};

use base64::Engine as _;
use fastrand::Rng;
use lk_agents_abi::Ts;
use serde_json::{Value, json};

use crate::{
    corpus::{draw_ms, share},
    fakes::SAMPLES_PER_CHUNK,
    input::{AGENT_TURNS, USER_TURNS},
    record::Tag,
};

/// The error codes the server refuses a client event with, as the real one does (the corpus
/// and the docs): a speech that fails with one of them failed on the server's word.
pub const REFUSALS: &[&str] = &[
    "conversation_already_has_active_response",
    "response_cancel_not_active",
    "input_audio_buffer_commit_empty",
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

#[derive(Clone, Debug, Default)]
struct Response {
    id: String,
    /// The message item, once minted (it is announced later).
    item: Option<String>,
    /// `response.output_item.added` for the message was delivered.
    announced: bool,
    /// `response.created` delivered: the generation exists for the client.
    created: bool,
    /// `response.done` scheduled: nothing more may be added.
    ending: bool,
    /// Delivered audio, nominal ms.
    audio_ms: u64,
    /// The generation's source id, from the hook at creation.
    source: Option<u32>,
    function_call: Option<String>,
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
}

/// One connection's server: its conversation, input buffer and responses.
pub struct Server {
    rng: Rng,
    config: Config,
    /// Events due, by instant then by the order they were scheduled.
    pending: BTreeMap<(Ts, u64), Pending>,
    seq: u64,
    items: Vec<String>,
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
    /// Nominal audio of each assistant item, for truncation.
    item_audio: HashMap<String, u64>,
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
            item_audio: HashMap::new(),
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
        let (&key, _) = self.pending.iter().next().filter(|((t, _), _)| *t <= now)?;
        let Pending { resp, body } = self.pending.remove(&key).expect("just seen");
        let frame = match body {
            Body::Json(v) => v,
            Body::Audio { item, chunk } => {
                let r = resp.map(|r| &mut self.responses[r as usize]);
                let source = r.as_ref().and_then(|r| r.source).unwrap_or(0);
                if let Some(r) = r {
                    r.audio_ms += DELTA_MS;
                }
                *self.item_audio.entry(item.clone()).or_default() += DELTA_MS;
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
        let ty = frame["type"].as_str().unwrap_or_default();
        if let Some(r) = resp {
            let r = r as usize;
            match ty {
                "response.created" => {
                    let id = self.responses[r].id.clone();
                    self.responses[r].source = Some(hooks.created(&id));
                    self.responses[r].created = true;
                }
                "response.output_item.added"
                    if frame["item"]["type"] == "message" =>
                {
                    self.responses[r].announced = true;
                }
                "response.done" => {
                    if let Some(source) = self.responses[r].source {
                        hooks.done(source);
                    }
                    if self.active == Some(r as u32) {
                        self.active = None;
                    }
                }
                _ => {}
            }
        }
        Some(frame)
    }

    // --- the client's frames -------------------------------------------------------------------

    /// One client frame, arrived at `now`.
    pub fn client(&mut self, now: Ts, frame: &Value, hooks: &mut dyn Hooks) {
        let ty = frame["type"].as_str().unwrap_or_default();
        match ty {
            "session.update" => {
                self.configure(&frame["session"]);
                let ev = self.event_id();
                self.after(now, 5, None, json!({"type": "session.updated", "event_id": ev, "session": frame["session"].clone()}));
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
                if self.buffer_ms == 0 {
                    self.error(now, "invalid_request_error", "input_audio_buffer_commit_empty", frame);
                } else {
                    let item = self.speaking.take().unwrap_or_else(|| self.mint_item("user"));
                    let lag = draw_ms(&mut self.rng, &["realtime", "error_after_client"]);
                    self.commit(later(now, lag), item);
                }
            }
            "input_audio_buffer.clear" => {
                self.buffer_ms = 0;
                let ev = self.event_id();
                self.after(now, 95, None, json!({"type": "input_audio_buffer.cleared", "event_id": ev}));
            }
            "response.create" => {
                if self.active.is_some() {
                    self.error(now, "invalid_request_error", "conversation_already_has_active_response", frame);
                    return;
                }
                let client_event_id = frame["response"]["metadata"]["client_event_id"].as_str().map(str::to_owned);
                let delay = draw_ms(&mut self.rng, &["realtime", "created_after_create"]);
                // A response's own `tool_choice` overrides the session's.
                let choice = frame["response"]["tool_choice"].as_str().map(str::to_owned);
                self.respond(later(now, delay), client_event_id, choice);
            }
            "response.cancel" => match self.active {
                Some(r) => {
                    let lag = draw_ms(&mut self.rng, &["realtime", "error_after_client"]);
                    self.cancel(later(now, lag), r, "client_cancelled");
                }
                None => self.error(now, "invalid_request_error", "response_cancel_not_active", frame),
            },
            "conversation.item.create" => {
                let mut item = frame["item"].clone();
                let id = item["id"].as_str().map(str::to_owned).unwrap_or_else(|| self.mint_item("item"));
                item["id"] = json!(id);
                item["status"] = json!("completed");
                let previous = match frame.get("previous_item_id").and_then(Value::as_str) {
                    Some("root") => None,
                    Some(p) => Some(p.to_owned()),
                    None => self.items.last().cloned(),
                };
                match &previous {
                    Some(p) if !self.items.contains(p) => {
                        self.error(now, "invalid_request_error", "invalid_value", frame);
                        return;
                    }
                    Some(p) => {
                        let at = self.items.iter().position(|i| i == p).map_or(self.items.len(), |i| i + 1);
                        self.items.insert(at, id.clone());
                    }
                    None if frame.get("previous_item_id").and_then(Value::as_str) == Some("root") => {
                        self.items.insert(0, id.clone())
                    }
                    None => self.items.push(id.clone()),
                }
                let (e1, e2) = (self.event_id(), self.event_id());
                self.after(now, 5, None, json!({"type": "conversation.item.added", "event_id": e1, "previous_item_id": previous, "item": item.clone()}));
                self.after(now, 6, None, json!({"type": "conversation.item.done", "event_id": e2, "previous_item_id": previous, "item": item}));
            }
            "conversation.item.delete" => {
                let id = frame["item_id"].as_str().unwrap_or_default().to_owned();
                match self.items.iter().position(|i| *i == id) {
                    Some(at) => {
                        self.items.remove(at);
                        let ev = self.event_id();
                        self.after(now, 5, None, json!({"type": "conversation.item.deleted", "event_id": ev, "item_id": id}));
                    }
                    None => self.error(now, "invalid_request_error", "invalid_value", frame),
                }
            }
            "conversation.item.truncate" => {
                let id = frame["item_id"].as_str().unwrap_or_default().to_owned();
                let end = frame["audio_end_ms"].as_u64().unwrap_or(0);
                match self.item_audio.get(&id) {
                    Some(&audio) if end <= audio => {
                        let ev = self.event_id();
                        self.after(now, 250, None, json!({"type": "conversation.item.truncated", "event_id": ev, "item_id": id, "content_index": 0, "audio_end_ms": end}));
                    }
                    _ => self.error(now, "invalid_request_error", "invalid_value", frame),
                }
            }
            // `conversation.item.retrieve`, `output_audio_buffer.clear` (WebRTC only) and anything
            // else: an unknown event is an error on the real server.
            _ => self.error(now, "invalid_request_error", "invalid_value", frame),
        }
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
    }

    // --- what the server does of its own accord ------------------------------------------------

    /// True when a real server could do this now.
    pub fn exo(&mut self, now: Ts, what: Exo) -> bool {
        match what {
            Exo::SpeechStarted => {
                if self.config.vad.is_none() || self.speaking.is_some() {
                    return false;
                }
                let item = self.mint_item("user");
                self.speaking = Some(item.clone());
                let ev = self.event_id();
                let start = self.heard_ms.saturating_sub(300);
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
                self.commit(now, item);
                if self.config.create_response && self.active.is_none() {
                    let delay = draw_ms(&mut self.rng, &["realtime", "created_after_stopped"]);
                    self.respond(later(now, delay), None, None);
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
                self.respond(now, None, None);
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
        let previous = self.items.last().cloned();
        self.items.push(item.clone());
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

    /// A response, from its `response.created` at `at` to its `response.done`.
    fn respond(&mut self, at: Ts, client_event_id: Option<String>, tool_choice: Option<String>) {
        let r = self.responses.len() as u32;
        let id = format!("resp_{}{}", self.ids, self.next_resp);
        self.next_resp += 1;
        self.responses.push(Response { id: id.clone(), ..Response::default() });
        self.active = Some(r);
        let metadata = client_event_id.map(|c| json!({"client_event_id": c}));
        let ev = self.event_id();
        self.put(at, Some(r), Body::Json(json!({"type": "response.created", "event_id": ev, "response": {"id": id, "object": "realtime.response", "status": "in_progress", "output": [], "metadata": metadata}})));

        let text = AGENT_TURNS[self.rng.usize(0..AGENT_TURNS.len() - 1)].to_owned();
        self.said.push(text.clone());
        let item = self.mint_item("msg");
        self.responses[r as usize].item = Some(item.clone());
        let previous = self.items.last().cloned();
        self.items.push(item.clone());
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
            self.items.push(fc.clone());
            self.responses[r as usize].function_call = Some(fc.clone());
            t = later(t, 30);
            let call = |status: &str, args: &str| json!({"id": fc, "type": "function_call", "status": status, "name": name, "call_id": format!("call_{fc}"), "arguments": args});
            let (e1, e2, e3, e4, e5, e6) = (self.event_id(), self.event_id(), self.event_id(), self.event_id(), self.event_id(), self.event_id());
            self.put(t, Some(r), Body::Json(json!({"type": "response.output_item.added", "event_id": e1, "response_id": id, "output_index": 1, "item": call("in_progress", "")})));
            self.put(t, Some(r), Body::Json(json!({"type": "conversation.item.added", "event_id": e2, "previous_item_id": item, "item": call("in_progress", "")})));
            self.put(t, Some(r), Body::Json(json!({"type": "response.function_call_arguments.delta", "event_id": e3, "response_id": id, "item_id": fc, "output_index": 1, "call_id": format!("call_{fc}"), "delta": "{}"})));
            let t2 = later(t, 30);
            self.message_done(t2, r, "completed");
            self.put(t2, Some(r), Body::Json(json!({"type": "response.function_call_arguments.done", "event_id": e4, "response_id": id, "item_id": fc, "output_index": 1, "call_id": format!("call_{fc}"), "name": name, "arguments": "{}"})));
            self.put(t2, Some(r), Body::Json(json!({"type": "conversation.item.done", "event_id": e5, "previous_item_id": item, "item": call("completed", "{}")})));
            self.put(t2, Some(r), Body::Json(json!({"type": "response.output_item.done", "event_id": e6, "response_id": id, "output_index": 1, "item": call("completed", "{}")})));
            t = t2;
        } else {
            t = later(t, draw_ms(&mut self.rng, &["realtime", "done_after_last_audio"]) / 2);
            self.message_done(t, r, "completed");
        }
        let done_at = later(t, draw_ms(&mut self.rng, &["realtime", "done_after_last_audio"]) / 2);
        self.end_body(done_at, r, "completed", None);
        self.responses[r as usize].ending = true;
    }

    /// The message's closing events: audio, transcript, part, item.
    fn message_done(&mut self, t: Ts, r: u32, status: &str) {
        let resp = &self.responses[r as usize];
        let (id, Some(item)) = (resp.id.clone(), resp.item.clone()) else { return };
        let previous = self.items.iter().position(|i| *i == item).and_then(|at| at.checked_sub(1)).map(|at| self.items[at].clone());
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

    fn end_body(&mut self, t: Ts, r: u32, status: &str, reason: Option<&str>) {
        let resp = &self.responses[r as usize];
        let mut outputs = Vec::new();
        if resp.announced || status == "completed" {
            outputs.push(json!({"type": "message", "id": resp.item}));
        }
        if let Some(fc) = &resp.function_call
            && status == "completed"
        {
            outputs.push(json!({"type": "function_call", "id": fc, "call_id": format!("call_{fc}"), "name": "", "arguments": "{}"}));
        }
        let details = match (status, reason) {
            ("failed", _) => json!({"type": "failed", "error": {"type": "server_error", "code": "server_error"}}),
            (_, Some(reason)) => json!({"type": status, "reason": reason}),
            _ => Value::Null,
        };
        let id = resp.id.clone();
        let ev = self.event_id();
        self.put(t, Some(r), Body::Json(json!({"type": "response.done", "event_id": ev, "response": {"id": id, "object": "realtime.response", "status": status, "status_details": details, "output": outputs, "usage": {"total_tokens": 100, "input_tokens": 60, "output_tokens": 40, "input_token_details": {"text_tokens": 40, "audio_tokens": 20, "cached_tokens": 0}, "output_token_details": {"text_tokens": 10, "audio_tokens": 30}}}})));
    }

    /// Stop a response at `t`: what was still to come is withdrawn, and it closes as `status`.
    fn end(&mut self, t: Ts, r: u32, status: &str, reason: Option<&str>) {
        let keep_created = !self.responses[r as usize].created;
        let withdrawn: Vec<(Ts, u64)> = self
            .pending
            .iter()
            .filter(|(_, p)| {
                p.resp == Some(r)
                    && !(keep_created
                        && matches!(&p.body, Body::Json(v) if v["type"] == "response.created"))
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
        let announced = self.responses[r as usize].announced;
        if announced {
            self.message_done(t, r, "incomplete");
        } else if let Some(item) = self.responses[r as usize].item.take() {
            self.items.retain(|i| *i != item);
        }
        self.responses[r as usize].function_call = None;
        self.end_body(t, r, status, reason);
        self.responses[r as usize].ending = true;
    }

    fn cancel(&mut self, t: Ts, r: u32, reason: &str) {
        self.end(t, r, "cancelled", Some(reason));
    }

    fn error(&mut self, now: Ts, kind: &str, code: &str, frame: &Value) {
        let lag = draw_ms(&mut self.rng, &["realtime", "error_after_client"]);
        let ev = self.event_id();
        let cause = frame.get("event_id").cloned().unwrap_or(Value::Null);
        self.after(now, lag, None, json!({"type": "error", "event_id": ev, "error": {"type": kind, "code": code, "message": code, "param": null, "event_id": cause}}));
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
