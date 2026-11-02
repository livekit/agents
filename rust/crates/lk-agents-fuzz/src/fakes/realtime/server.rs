//! One connection's realtime server, at the level the session reports it: what a LiveKit
//! Inference realtime model (the OpenAI Realtime server behind it) does with the session's
//! requests, as the real one was measured doing it (`calibration.json`), and what it refuses.
//! This model reproduces the outcomes that reach the driver.
//!
//! Each happening's delay after its cause is drawn from those measurements (`fakes::delay`, which
//! also strays to deadlines and stalls on random runs), so a reply is created about
//! 100 ms after it is asked, its message 300 ms later, its audio in 400 ms deltas faster than real
//! time, and its end while the person is still hearing most of it -- which is what makes a server
//! onset over a finished generation the common barge-in, not a rare one. The server answers
//! requests in the order they came.
//!
//! It holds the conversation: items by id and kind, a reply's message placed when it is added
//! (after any item that arrived first), deleted and truncated on the session's word (a truncation
//! cuts what the message holds); a reply reads what the conversation holds when it is asked. One
//! reply at a time: a second request over an active one is refused, a cancel frees the
//! conversation at once (a reply asked in the same breath is created before the cancelled one's
//! end), and a cancelled reply keeps streaming what it had generated before the cancel landed.
//! Under server VAD an onset needs audio the session sent; it cancels the reply in progress, and
//! the end of the turn commits the buffer and, with no reply active, opens one. The transcription
//! completes asynchronously, before or after the reply's events.
//!
//! What the real server refuses it refuses, with the real code, and reports each refusal as a
//! [`Breach`] with whether the driver already knew better when it asked (it had been told the
//! state that makes the request invalid) or merely raced an event on its way. So are two answers
//! the server gives from the wrong state: a reply asked in the same breath before the tool output
//! it should read, and a cancel landing on the server's reply to the person's turn before
//! anything prompted it.
use std::{collections::BTreeMap, time::Duration};

use fastrand::Rng;
use lk_agents_abi::{
    ChatItem, ChatMessage, FunctionCall, Role, ToolChoice, Ts, llm::realtime::GenerationStatus,
};

use crate::{
    fakes::{Measured, RealtimeDelays, Timing, delay, measured},
    input::{AGENT_TURNS, USER_TURNS},
    record::{Breach, Tag},
};

/// The error codes the server refuses a request with, as the real one does,
/// and the words a speech that failed on one carries: such a speech failed on the server's word.
pub const REFUSALS: &[&str] = &[
    "conversation_already_has_active_response",
    "Conversation already has an active response",
    "response_cancel_not_active",
    "response_cancel_not_found",
    "input_audio_buffer_commit_empty",
    "item_truncate_invalid_item_id",
    "invalid_value",
];

/// The nominal length of a full audio delta, as the real server sends nearly all of them.
pub const DELTA_MS: u64 = 400;

/// What the server is told to do of its own accord: the person's speech as its VAD hears it, and
/// the ways a real server goes wrong. Mirrors `RtEvent`, which is how a case says it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exo {
    SpeechStarted,
    SpeechStopped,
    /// The server opens a response on its own (`input_audio_buffer.timeout_triggered` and the
    /// like).
    ServerResponse,
    /// The oldest response still generating finishes now, all of it at once.
    FinishNow,
    /// A server-side `error`, the session staying open.
    ServerError,
    /// An `error` the session cannot go on after (the account's quota).
    FatalError,
    /// The response in progress fails.
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

/// One thing the server tells the session, at its instant. `r` is the response it belongs to.
#[derive(Clone, Debug)]
pub enum Happening {
    Created {
        r: u32,
    },
    /// The reply's message item.
    MessageAdded {
        r: u32,
        previous: Option<String>,
    },
    /// Its content part: the modalities settle.
    Part {
        r: u32,
    },
    Text {
        r: u32,
        delta: String,
    },
    Audio {
        r: u32,
        chunk: u16,
    },
    CallAdded {
        r: u32,
    },
    MessageDone {
        r: u32,
    },
    CallDone {
        r: u32,
    },
    Done {
        r: u32,
        status: GenerationStatus,
    },
    /// An item the session created, or the person's committed turn, now in the conversation.
    ItemAdded {
        item: Box<ChatItem>,
        previous: Option<String>,
    },
    SpeechStarted {
        item: String,
    },
    SpeechStopped,
    /// The person's committed turn's transcript so far, or all of it.
    Transcript {
        item: String,
        text: String,
        last: bool,
    },
    /// The transcription failed: what had come of it stands.
    TranscriptFailed {
        item: String,
    },
    /// An `error` event; `request` when it refuses a reply request.
    Error {
        code: Option<&'static str>,
        message: &'static str,
        request: Option<String>,
    },
    /// An `error` the session cannot go on after.
    Fatal,
}

/// What the server holds of an item.
#[derive(Clone, Debug)]
struct Item {
    id: String,
    call_id: Option<String>,
    function_call: bool,
    assistant: bool,
    /// Audio delivered for an assistant message, ms: what truncation is judged against and what
    /// the message keeps, a truncation cutting it where the session says.
    audio_ms: u64,
    cut_ms: Option<u64>,
    /// The generation that made it, for the server's own messages.
    source: Option<u32>,
}

impl Item {
    fn of(item: &ChatItem) -> Self {
        let (call_id, function_call, assistant) = match item {
            ChatItem::Message(m) => (None, false, m.role == Role::Assistant),
            ChatItem::FunctionCall(f) => (Some(f.call_id.clone()), true, false),
            ChatItem::FunctionCallOutput(f) => (Some(f.call_id.clone()), false, false),
            ChatItem::AgentHandoff(_) | ChatItem::AgentConfigUpdate(_) => (None, false, false),
        };
        Item {
            id: item.id().to_owned(),
            call_id,
            function_call,
            assistant,
            audio_ms: 0,
            cut_ms: None,
            source: None,
        }
    }
}

/// A response, from its request to its end.
#[derive(Clone, Debug, Default)]
pub struct Response {
    pub id: String,
    /// The request it answers.
    pub request: Option<String>,
    /// The server's answer to the person's turn its VAD ended.
    turn: bool,
    instructions: usize,
    read_items: usize,
    /// Its message item, once minted, and the words it says.
    pub item: Option<String>,
    pub text: String,
    /// The tool call it makes, `(item, name)`.
    pub function_call: Option<(String, String)>,
    /// What the session was told: created, the message, its end.
    /// Created on the server, and when the driver was told so (a reply cancelled before it was
    /// created never is).
    created: bool,
    told_created: Option<Ts>,
    announced: bool,
    message_done: bool,
    call_announced: bool,
    call_done: bool,
    finished: Option<Ts>,
    /// The session was told the person started speaking while it generated: a reason to cancel.
    prompted: bool,
    /// Delivered audio, ms.
    audio_ms: u64,
    /// The run's generation, registered when the session was told it was created.
    pub source: Option<u32>,
}

// What the real server was measured doing, in ms: `calibration.json`'s `realtime/<name>` in
// lower case. These executable constants are maintained manually from that reference data;
// editing calibration.json alone does not change a simulation.
pub const SPEECH_STARTED_LAG_SEMANTIC: Measured =
    [238, 278, 299, 316, 329, 335, 345, 385, 404, 451, 1327];
pub const SPEECH_STARTED_LAG_SERVER: Measured =
    [262, 307, 319, 335, 365, 398, 1704, 1803, 2409, 2693, 2728];
pub const SPEECH_STOPPED_LAG_SEMANTIC: Measured =
    [671, 729, 737, 759, 787, 828, 1168, 1421, 1668, 1853, 3450];
pub const SPEECH_STOPPED_LAG_SERVER: Measured =
    [582, 614, 628, 632, 653, 686, 725, 1606, 3016, 3820, 5682];
const CANCEL_AFTER_SPEECH_STARTED: Measured = [3, 8, 9, 13, 17, 24, 27, 29, 35, 39, 246];
const CREATED_AFTER_STOPPED: Measured = [0, 4, 6, 6, 7, 7, 8, 9, 9, 10, 15];
const CREATED_AFTER_CREATE: Measured = [104, 105, 107, 111, 118, 120, 129, 134, 134, 148, 345];
const TRANSCRIPTION_AFTER_COMMITTED: Measured =
    [278, 340, 362, 383, 391, 408, 453, 513, 599, 683, 3248];
const ITEM_AFTER_CREATED: Measured = [164, 214, 290, 349, 386, 419, 475, 536, 572, 723, 1463];
const FIRST_AUDIO_AFTER_ITEM: Measured = [62, 209, 235, 248, 268, 305, 317, 329, 345, 401, 1482];
const AUDIO_GAP: Measured = [0, 23, 53, 60, 65, 70, 74, 81, 104, 159, 1108];
const DONE_AFTER_LAST_AUDIO: Measured = [0, 28, 30, 33, 36, 38, 42, 44, 73, 187, 568];
const ERROR_AFTER_CLIENT: Measured = [83, 88, 89, 90, 91, 93, 96, 99, 100, 103, 110];
const DONE_AFTER_CANCEL: Measured = [115, 120, 124, 128, 131, 133, 135, 138, 144, 150, 156];
/// The audio a character of the reply takes (`audio_ms_per_char`): a pace, never stray.
const AUDIO_MS_PER_CHAR: Measured = [27, 40, 48, 50, 52, 56, 58, 58, 60, 61, 75];
/// Responses that called a tool, of all measured (`function_call_share`).
const FUNCTION_CALL_SHARE: (u64, u64) = (8, 94);

/// One connection's server: its conversation, input buffer and responses.
pub struct Server {
    rng: Rng,
    timing: Timing,
    delays: RealtimeDelays,
    vad: Option<Vad>,
    instructions: usize,
    tools: Vec<String>,
    tool_choice: Option<ToolChoice>,
    audio: bool,
    tool_calls: Option<u8>,
    ids: String,
    /// What is due, by instant then by the order it was scheduled.
    pending: BTreeMap<(Ts, u64), Happening>,
    seq: u64,
    items: Vec<Item>,
    next_item: u32,
    pub responses: Vec<Response>,
    /// The response the server is generating: at most one.
    active: Option<u32>,
    /// The user item the VAD opened with `speech_started`.
    speaking: Option<String>,
    /// Audio sent since the last commit or clear, ms.
    buffer_ms: u64,
    /// The buffer was last emptied by the session's own commit or clear (or never), so it can
    /// count what the buffer holds; a VAD commit empties it behind the session's back.
    own_empty: bool,
    /// Replies asked while the conversation ended in a call without its output, `(response,
    /// call, when)`; outputs that arrived after one in its breath, until the breath ends.
    unanswered: Vec<(String, String, Ts)>,
    early: Vec<(String, String, Ts)>,
    breaches: Vec<Breach>,
    /// When the server answers the last request it took: the next answer comes after.
    answered: Ts,
    last_user: Option<String>,
}

impl Server {
    pub fn new(seed: u64, ids: String) -> Self {
        Server {
            rng: Rng::with_seed(seed),
            timing: Timing::default(),
            delays: RealtimeDelays::default(),
            vad: None,
            instructions: 0,
            tools: Vec::new(),
            tool_choice: None,
            audio: true,
            tool_calls: None,
            ids,
            pending: BTreeMap::new(),
            seq: 0,
            items: Vec::new(),
            next_item: 0,
            responses: Vec::new(),
            active: None,
            speaking: None,
            buffer_ms: 0,
            own_empty: true,
            unanswered: Vec::new(),
            early: Vec::new(),
            breaches: Vec::new(),
            answered: Ts::ZERO,
            last_user: None,
        }
    }

    /// The session's configuration, as its opening `session.update` gives it.
    pub fn configure(
        &mut self,
        vad: Option<Vad>,
        audio: bool,
        tool_calls: Option<u8>,
        timing: Timing,
        delays: RealtimeDelays,
    ) {
        self.vad = vad;
        self.audio = audio;
        self.tool_calls = tool_calls;
        self.timing = timing;
        self.delays = delays;
    }

    pub fn instructions(&mut self, instructions: &str) {
        self.instructions = instructions.len();
    }

    pub fn tools(&mut self, tools: Vec<String>) {
        self.tools = tools;
    }

    pub fn next_due(&self) -> Option<Ts> {
        self.pending.keys().next().map(|(t, _)| *t)
    }

    /// The next happening due at `now`, as the session is told it: what telling it means to the
    /// server's own bookkeeping is done here.
    pub fn pop(&mut self, now: Ts) -> Option<Happening> {
        self.breathe(now);
        let (&key, _) = self.pending.iter().next().filter(|((t, _), _)| *t <= now)?;
        let mut happening = self.pending.remove(&key).expect("just seen");
        match &mut happening {
            Happening::Created { r } => {
                self.responses[*r as usize].created = true;
            }
            Happening::MessageAdded { r, previous } => {
                *previous = self.items.last().map(|i| i.id.clone());
                let resp = &self.responses[*r as usize];
                let mut item =
                    Item::of(&message(resp.item.as_deref().unwrap_or(""), Role::Assistant));
                item.source = resp.source;
                self.items.push(item);
                self.responses[*r as usize].announced = true;
            }
            Happening::Audio { r, .. } => {
                let resp = &mut self.responses[*r as usize];
                resp.audio_ms += DELTA_MS;
                if let Some(item) = resp.item.clone()
                    && let Some(i) = self.items.iter_mut().find(|i| i.id == item)
                {
                    i.audio_ms = (i.audio_ms + DELTA_MS).min(i.cut_ms.unwrap_or(u64::MAX));
                }
            }
            Happening::CallAdded { r } => {
                let resp = &mut self.responses[*r as usize];
                resp.call_announced = true;
                if let Some((item, name)) = resp.function_call.clone() {
                    self.items.push(Item::of(&function_call(&item, &name)));
                }
            }
            Happening::MessageDone { r } => self.responses[*r as usize].message_done = true,
            Happening::CallDone { r } => self.responses[*r as usize].call_done = true,
            Happening::Done { r, .. } => {
                self.responses[*r as usize].finished = Some(now);
                if self.active == Some(*r) {
                    self.active = None;
                }
            }
            Happening::SpeechStarted { .. } => {
                for resp in self.responses.iter_mut().filter(|r| r.finished.is_none()) {
                    resp.prompted = true;
                }
            }
            _ => {}
        }
        Some(happening)
    }

    /// The generation the run registered for a response the server created.
    pub fn registered(&mut self, r: u32, source: u32) {
        self.responses[r as usize].source = Some(source);
    }

    /// The driver was told the response was created.
    pub fn told_created(&mut self, r: u32, now: Ts) {
        self.responses[r as usize].told_created = Some(now);
    }

    /// What it breached since asked last.
    pub fn take_breaches(&mut self) -> Vec<Breach> {
        std::mem::take(&mut self.breaches)
    }

    /// The assistant messages it generated and still holds: by generation, the audio each keeps.
    pub fn held(&self) -> Vec<(u32, u64)> {
        self.items
            .iter()
            .filter(|i| i.assistant)
            .filter_map(|i| Some((i.source?, i.audio_ms)))
            .collect()
    }

    /// The ids of what it holds, in order.
    pub fn ids(&self) -> Vec<String> {
        self.items.iter().map(|i| i.id.clone()).collect()
    }

    pub fn holds(&self, id: &str) -> bool {
        self.items.iter().any(|i| i.id == id)
    }

    pub fn holds_call(&self, call_id: &str) -> bool {
        self.items.iter().any(|i| i.function_call && i.call_id.as_deref() == Some(call_id))
    }

    // --- what the session asks ------------------------------------------------------------

    /// An item the session creates after `previous` (`None`: at the end).
    pub fn create(&mut self, now: Ts, item: &ChatItem, previous: Option<String>) {
        self.breathe(now);
        let at = match &previous {
            Some(p) => {
                self.items.iter().position(|i| i.id == *p).map_or(self.items.len(), |i| i + 1)
            }
            None => self.items.len(),
        };
        let held = Item::of(item);
        if let (false, Some(call)) = (held.function_call, held.call_id.clone()) {
            for (response, c, asked) in &self.unanswered {
                if *c == call && *asked == now {
                    self.early.push((response.clone(), call.clone(), now));
                }
            }
        }
        self.items.insert(at, held);
        let lag = self.answer_lag(now);
        let previous = at.checked_sub(1).map(|p| self.items[p].id.clone());
        self.put(later(now, lag), Happening::ItemAdded { item: Box::new(item.clone()), previous });
    }

    pub fn delete(&mut self, id: &str) {
        self.items.retain(|i| i.id != id);
    }

    pub fn append_audio(&mut self, ms: u64) {
        self.buffer_ms += ms;
    }

    pub fn buffered_ms(&self) -> u64 {
        self.buffer_ms
    }

    /// The VAD heard the person start and not yet stop.
    pub fn speaking(&self) -> bool {
        self.speaking.is_some()
    }

    /// The audio an item holds, if it holds the item.
    pub fn audio_ms(&self, id: &str) -> Option<u64> {
        self.items.iter().find(|i| i.id == id).map(|i| i.audio_ms)
    }

    /// The session's commit of what it sent.
    pub fn commit_audio(&mut self, now: Ts) {
        if self.buffer_ms < 100 {
            // The session could count what it sent unless the VAD emptied the buffer on its own.
            let knowing = self.own_empty;
            let why = "Error committing input audio buffer: buffer too small. Expected at least 100ms of audio.";
            // The session takes the refusal of a commit over its VAD's own as nothing.
            let told = self.vad.is_none();
            return self.refuse(
                now,
                "input_audio_buffer.commit",
                "input_audio_buffer_commit_empty",
                why,
                knowing,
                told,
            );
        }
        let item = self.speaking.take().unwrap_or_else(|| self.mint_item("user"));
        let lag = self.answer_lag(now);
        self.own_empty = true;
        self.commit(later(now, lag), item);
    }

    pub fn clear_audio(&mut self) {
        self.buffer_ms = 0;
        self.own_empty = true;
    }

    /// A reply asked for; refused while one is active.
    pub fn respond_to(
        &mut self,
        now: Ts,
        request: String,
        instructions: Option<&str>,
        tool_choice: Option<ToolChoice>,
    ) {
        self.breathe(now);
        if let Some(r) = self.active {
            // Knowing only if the session had been told of the response refusing it.
            let knowing = self.responses[r as usize].told_created.is_some_and(|at| at < now);
            let code = "conversation_already_has_active_response";
            let why = "Conversation already has an active response in progress. Wait until the response is finished before creating a new one.";
            self.breaches.push(Breach::Refused {
                event: "response.create".into(),
                code,
                why,
                knowing,
            });
            let lag = self.answer_lag(now);
            let error = Happening::Error { code: Some(code), message: why, request: Some(request) };
            return self.put(later(now, lag), error);
        }
        let unanswered =
            self.items.last().filter(|i| i.function_call).and_then(|i| i.call_id.clone());
        let lag = self.delays.created.unwrap_or_else(|| self.delay(&CREATED_AFTER_CREATE));
        let delay = self.answer_in(now, lag);
        let instructions = instructions.map(str::len);
        let r = self.respond(later(now, delay), Some(request), tool_choice, instructions);
        // A reply asked after the outputs reads them.
        self.early.retain(|(.., at)| *at != now);
        if let Some(call) = unanswered {
            self.unanswered.push((self.responses[r as usize].id.clone(), call, now));
        }
    }

    /// A `response.cancel`, of `named` or of whatever is in progress.
    pub fn cancel(&mut self, now: Ts, named: Option<&str>) {
        if let Some(id) = named
            && !self.responses.iter().any(|r| r.id == id)
        {
            let why = "Response with id not found";
            return self.refuse(
                now,
                "response.cancel",
                "response_cancel_not_found",
                why,
                true,
                false,
            );
        }
        let target =
            self.active.filter(|r| named.is_none_or(|id| self.responses[*r as usize].id == id));
        let Some(r) = target else {
            // Knowing when the session had been told the end of the response it names (of every
            // one, unnamed); a response it has not heard the end of makes the cancel a race.
            let told_done = |r: &Response| r.finished.is_some_and(|at| at < now);
            let knowing = match named {
                Some(id) => self.responses.iter().any(|r| r.id == id && told_done(r)),
                None => self.responses.iter().all(told_done),
            };
            // "Cancellation failed": the session takes it as nothing.
            let why = "Cancellation failed: no active response found";
            return self.refuse(
                now,
                "response.cancel",
                "response_cancel_not_active",
                why,
                knowing,
                false,
            );
        };
        // The server's answer to the person's turn, nobody has heard a word of, no onset told
        // over it: the session cancels it on nothing the server saw.
        let resp = &self.responses[r as usize];
        if resp.turn && !resp.prompted && resp.audio_ms == 0 {
            let (response, generation) = (resp.id.clone(), resp.source);
            self.breaches.push(Breach::CancelledUnprompted { response, generation });
        }
        // The conversation is free at once: a reply asked right after the cancel is taken, and
        // created before the cancelled one's end (probe-cancel_then_create). The cancel is taken
        // in order, but its wind-down holds back no later answer.
        self.active = None;
        let lag = self.delay(&DONE_AFTER_CANCEL);
        self.stop(later(now, lag).max(self.answered), r, GenerationStatus::Cancelled);
    }

    /// A truncation of a message the session played part of.
    pub fn truncate(&mut self, now: Ts, item: &str, end_ms: u64) {
        let Some(at) = self.items.iter().position(|i| i.id == item) else {
            let why = "Item with item_id not found";
            return self.refuse(
                now,
                "conversation.item.truncate",
                "item_truncate_invalid_item_id",
                why,
                true,
                true,
            );
        };
        // Within the audio it holds, a millisecond of rounding allowed.
        let held = &mut self.items[at];
        if end_ms > held.audio_ms + 1 {
            let why = "Audio content is already shorter than audio_end_ms";
            return self.refuse(
                now,
                "conversation.item.truncate",
                "invalid_value",
                why,
                true,
                true,
            );
        }
        held.audio_ms = end_ms;
        held.cut_ms = Some(end_ms);
    }

    /// A message the session put back as text: it holds no audio.
    pub fn as_text(&mut self, item: &str) {
        if let Some(held) = self.items.iter_mut().find(|i| i.id == item) {
            held.audio_ms = 0;
            held.cut_ms = Some(0);
        }
    }

    // --- what the server does of its own accord ------------------------------------------

    /// True when a real server could do this now.
    pub fn exo(&mut self, now: Ts, what: Exo) -> bool {
        match what {
            Exo::SpeechStarted => {
                // The VAD hears only audio the session sent since the buffer was last emptied.
                if self.vad.is_none() || self.speaking.is_some() || self.buffer_ms == 0 {
                    return false;
                }
                let item = self.mint_item("user");
                self.speaking = Some(item.clone());
                self.put(now, Happening::SpeechStarted { item });
                if let Some(r) = self.active {
                    let lag = self.delay(&CANCEL_AFTER_SPEECH_STARTED);
                    self.active = None;
                    self.stop(later(now, lag), r, GenerationStatus::Cancelled);
                }
                true
            }
            Exo::SpeechStopped => {
                let Some(item) = self.speaking.take() else { return false };
                self.put(now, Happening::SpeechStopped);
                self.own_empty = false;
                self.commit(now, item);
                if self.active.is_none() {
                    let delay = self.delay(&CREATED_AFTER_STOPPED);
                    let r = self.respond(later(now, delay), None, None, None);
                    self.responses[r as usize].turn = true;
                }
                true
            }
            Exo::ServerResponse => {
                // Only the server's own turn detection opens a response unasked.
                if self.active.is_some() || self.vad.is_none() {
                    return false;
                }
                self.respond(now, None, None, None);
                true
            }
            Exo::FinishNow => {
                let Some(r) = self.active else { return false };
                let mine: Vec<(Ts, u64)> = self
                    .pending
                    .iter()
                    .filter(|(_, h)| response_of(h) == Some(r))
                    .map(|(k, _)| *k)
                    .collect();
                // Everything still to come of it, at once and in order.
                for k in mine {
                    let h = self.pending.remove(&k).expect("just seen");
                    self.put(now, h);
                }
                true
            }
            Exo::ServerError => {
                let message = "The server had an error while processing your request.";
                self.put(now, Happening::Error { code: None, message, request: None });
                true
            }
            Exo::FatalError => {
                self.put(now, Happening::Fatal);
                true
            }
            Exo::ResponseFailed => {
                // It holds the conversation until its end is out.
                let Some(r) = self.active else { return false };
                self.stop(now, r, GenerationStatus::Failed);
                true
            }
            Exo::TranscriptionFailed => {
                let Some(item) = self.last_user.clone() else { return false };
                // What is still to come of its transcript never comes.
                let mine: Vec<(Ts, u64)> = self
                    .pending
                    .iter()
                    .filter(
                        |(_, h)| matches!(h, Happening::Transcript { item: i, .. } if *i == item),
                    )
                    .map(|(k, _)| *k)
                    .collect();
                if mine.is_empty() {
                    return false;
                }
                for k in mine {
                    self.pending.remove(&k);
                }
                self.put(now, Happening::TranscriptFailed { item });
                true
            }
        }
    }

    // --- the pieces ------------------------------------------------------------------------

    /// The buffer becomes a user item, and its transcription follows.
    fn commit(&mut self, now: Ts, item: String) {
        self.buffer_ms = 0;
        let previous = self.items.last().map(|i| i.id.clone());
        let user = message(&item, Role::User);
        self.items.push(Item::of(&user));
        self.last_user = Some(item.clone());
        self.put(later(now, 1), Happening::ItemAdded { item: Box::new(user), previous });
        let spoken: Vec<&str> = USER_TURNS.iter().copied().filter(|t| !t.is_empty()).collect();
        let text = spoken[self.rng.usize(0..spoken.len())];
        let lag = self.delay(&TRANSCRIPTION_AFTER_COMMITTED);
        let words: Vec<&str> = text.split(' ').collect();
        // About one delta a word, streamed over the last fifth of the wait.
        for i in 0..words.len() {
            let at = lag * 4 / 5 + (lag / 5) * i as u64 / words.len() as u64;
            let text = words[..=i].join(" ");
            let item = item.clone();
            self.put(later(now, at), Happening::Transcript { item, text, last: false });
        }
        let text = text.to_owned();
        self.put(later(now, lag), Happening::Transcript { item, text, last: true });
    }

    /// A response, from its creation at `at` to its end, reading the conversation as it stands.
    fn respond(
        &mut self,
        at: Ts,
        request: Option<String>,
        tool_choice: Option<ToolChoice>,
        instructions: Option<usize>,
    ) -> u32 {
        let r = self.responses.len() as u32;
        let id = format!("resp_{}{}", self.ids, r);
        let spoken: Vec<&str> = AGENT_TURNS.iter().copied().filter(|t| !t.is_empty()).collect();
        let text = spoken[self.rng.usize(0..spoken.len())].to_owned();
        let item = self.mint_item("msg");
        self.responses.push(Response {
            id,
            request,
            instructions: instructions.unwrap_or(self.instructions),
            read_items: self.items.len(),
            item: Some(item),
            text: text.clone(),
            ..Response::default()
        });
        self.active = Some(r);
        self.put(at, Happening::Created { r });
        // Its place in the conversation is taken when the item is added: an item arriving before
        // then comes before it.
        let mut t = later(at, self.delay(&ITEM_AFTER_CREATED));
        self.put(t, Happening::MessageAdded { r, previous: None });
        let item_at = t;
        let first_audio = later(item_at, self.delay(&FIRST_AUDIO_AFTER_ITEM));
        t = later(item_at, 15).min(first_audio);
        self.put(t, Happening::Part { r });
        let per_char = measured(&mut self.rng, &AUDIO_MS_PER_CHAR) as f64;
        let audio_ms = ((text.chars().count() as f64 * per_char) as u64).max(DELTA_MS);
        let deltas = audio_ms.div_ceil(DELTA_MS) as usize;
        let words: Vec<&str> = text.split_inclusive(' ').collect();
        // The transcript's first words come with the content part, before any audio, as the
        // real server sends them: a response cut there has words and no audio.
        let mut said = words.len().min(2);
        if said > 0 {
            self.put(t, Happening::Text { r, delta: words[..said].concat() });
        }
        for chunk in 0..deltas {
            t = if chunk == 0 { first_audio } else { later(t, self.delay(&AUDIO_GAP)) };
            // The transcript leads the audio it describes.
            let upto = (words.len() * (chunk + 1)).div_ceil(deltas);
            if upto > said {
                self.put(t, Happening::Text { r, delta: words[said..upto].concat() });
                said = upto;
            }
            if self.audio {
                let chunk = (chunk as u16).min(Tag::MAX_CHUNK);
                self.put(t, Happening::Audio { r, chunk });
            }
        }
        // A tool call now and then, when the session offers tools: after the words.
        let choice = tool_choice.or_else(|| self.tool_choice.clone());
        let (calls, of) = match (choice, self.tool_calls) {
            (Some(ToolChoice::Forbidden), _) => (0, 1),
            (Some(ToolChoice::Required), _) => (1, 1),
            (_, Some(percent)) => (u64::from(percent), 100),
            (_, None) => FUNCTION_CALL_SHARE,
        };
        if !self.tools.is_empty() && self.rng.u64(0..of) < calls {
            let name = self.tools[self.rng.usize(0..self.tools.len())].clone();
            let fc = self.mint_item("fc");
            self.responses[r as usize].function_call = Some((fc, name));
            t = later(t, 30);
            self.put(t, Happening::CallAdded { r });
            t = later(t, 30);
            self.put(t, Happening::MessageDone { r });
            self.put(t, Happening::CallDone { r });
        } else {
            t = later(t, self.done_after_audio() / 2);
            self.put(t, Happening::MessageDone { r });
        }
        let done_at = later(t, self.done_after_audio() / 2);
        self.put(done_at, Happening::Done { r, status: GenerationStatus::Completed });
        r
    }

    /// Stop a response at `t`: what it had generated before then still streams, what was still
    /// to come is withdrawn, and it ends as `status`.
    fn stop(&mut self, t: Ts, r: u32, status: GenerationStatus) {
        let streams = |h: &Happening| {
            matches!(
                h,
                Happening::Audio { .. }
                    | Happening::Text { .. }
                    | Happening::Part { .. }
                    | Happening::MessageAdded { .. }
                    | Happening::CallAdded { .. }
            )
        };
        let not_created = !self.responses[r as usize].created;
        let (mut opened, mut called) = (false, false);
        let withdrawn: Vec<(Ts, u64)> = self
            .pending
            .iter()
            .filter(|((due, _), h)| {
                if response_of(h) != Some(r) {
                    return false;
                }
                let created = matches!(h, Happening::Created { .. });
                let kept = (*due < t && streams(h)) || (not_created && created);
                opened |= kept && matches!(h, Happening::MessageAdded { .. });
                called |= kept && matches!(h, Happening::CallAdded { .. });
                !kept
            })
            .map(|(k, _)| *k)
            .collect();
        for k in withdrawn {
            self.pending.remove(&k);
        }
        // A response stopped before the session heard of it still opens before it ends.
        let last = self
            .pending
            .iter()
            .filter(|(_, h)| response_of(h) == Some(r))
            .map(|((due, _), _)| *due)
            .max();
        let t = last.map_or(t, |due| t.max(due));
        let resp = &self.responses[r as usize];
        if (resp.announced || opened) && !resp.message_done {
            self.put(t, Happening::MessageDone { r });
        } else if !(resp.announced || opened) {
            self.responses[r as usize].item = None;
        }
        let resp = &self.responses[r as usize];
        if (resp.call_announced || called) && !resp.call_done {
            self.put(t, Happening::CallDone { r });
        } else if !(resp.call_announced || called) {
            self.responses[r as usize].function_call = None;
        }
        self.put(t, Happening::Done { r, status });
    }

    /// The token usage a finished response reports: what it read and what it said.
    pub fn usage(&self, r: u32) -> (u64, u64) {
        let resp = &self.responses[r as usize];
        let input = resp.instructions as u64 / 4 + 40 + 20 * resp.read_items as u64;
        (input, 40)
    }

    /// The breath before `now` is over: an output that arrived after a reply asked in it, and no
    /// reply was asked after it, was read by none.
    fn breathe(&mut self, now: Ts) {
        self.unanswered.retain(|(.., at)| *at == now);
        let (over, open): (Vec<_>, Vec<_>) = self.early.drain(..).partition(|(.., at)| *at < now);
        self.early = open;
        for (response, call_id, _) in over {
            let request =
                self.responses.iter().find(|r| r.id == response).and_then(|r| r.request.clone());
            self.breaches.push(Breach::AskedBeforeOutput { response, request, call_id });
        }
    }

    /// A refused request: the breach noted, and the `error` the session hears when `told`.
    fn refuse(
        &mut self,
        now: Ts,
        event: &str,
        code: &'static str,
        why: &'static str,
        knowing: bool,
        told: bool,
    ) {
        self.breaches.push(Breach::Refused { event: event.into(), code, why, knowing });
        if told {
            let lag = self.answer_lag(now);
            self.put(
                later(now, lag),
                Happening::Error { code: Some(code), message: why, request: None },
            );
        }
    }

    /// How long the server takes to answer a request: drawn from what was measured, and never before its
    /// answer to an earlier one -- it takes the session's requests in order.
    fn answer_lag(&mut self, now: Ts) -> u64 {
        self.answer_after(now, &ERROR_AFTER_CLIENT)
    }

    fn answer_after(&mut self, now: Ts, measured: &Measured) -> u64 {
        let lag = self.delay(measured);
        self.answer_in(now, lag)
    }

    fn answer_in(&mut self, now: Ts, lag: u64) -> u64 {
        let at = later(now, lag).max(self.answered);
        self.answered = at;
        at.saturating_duration_since(now).as_millis() as u64
    }

    /// How long after its last audio a reply ends; its message ends halfway.
    fn done_after_audio(&mut self) -> u64 {
        self.delays.done.unwrap_or_else(|| self.delay(&DONE_AFTER_LAST_AUDIO))
    }

    fn delay(&mut self, measured: &Measured) -> u64 {
        delay(&mut self.rng, measured, &self.timing)
    }

    fn mint_item(&mut self, kind: &str) -> String {
        self.next_item += 1;
        format!("item_{kind}_{}{}", self.ids, self.next_item)
    }

    fn put(&mut self, t: Ts, happening: Happening) {
        self.seq += 1;
        self.pending.insert((t, self.seq), happening);
    }
}

fn response_of(h: &Happening) -> Option<u32> {
    match h {
        Happening::Created { r }
        | Happening::MessageAdded { r, .. }
        | Happening::Part { r }
        | Happening::Text { r, .. }
        | Happening::Audio { r, .. }
        | Happening::CallAdded { r }
        | Happening::MessageDone { r }
        | Happening::CallDone { r }
        | Happening::Done { r, .. } => Some(*r),
        _ => None,
    }
}

/// A message as the server creates it, its content left out; stamped when the session hears it.
pub fn message(id: &str, role: Role) -> ChatItem {
    ChatItem::Message(ChatMessage::new(id, role, Vec::new(), std::time::SystemTime::UNIX_EPOCH))
}

pub fn function_call(id: &str, name: &str) -> ChatItem {
    ChatItem::FunctionCall(FunctionCall {
        id: id.to_owned(),
        call_id: format!("call_{id}"),
        arguments: "{}".into(),
        name: name.to_owned(),
        created_at: std::time::SystemTime::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    })
}

pub fn later(t: Ts, ms: u64) -> Ts {
    t.saturating_add(Duration::from_millis(ms))
}
