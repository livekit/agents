use super::effects::KeytermEffect;
use crate::agent_session::VoiceDriver;
use crate::agent_session::effects::Timer;
use crate::generation::GenEvent;
use crate::telemetry;
use crate::trace::{DetectorName, DetectorStep, InputStep};
use lk_agents_abi::metrics::base::LlmMetrics;
use lk_agents_abi::resources::ResourceId;
pub use lk_agents_abi::turn::keyterms::{
    DetectionLlm, KeytermDetectionConfig, KeytermUpdate, KeytermsChanged, ResolvedKeytermDetection,
};
use lk_agents_abi::{ChatContext, ChatItem, ChatMessage, HostRef, Llm, Role, Ts, Vocabulary};
use std::sync::Arc;

/// Bound on a single pass so a stuck LLM call can't hold the single-flight guard; a timeout changes
/// nothing.
pub const DETECTION_TIMEOUT: f64 = 10.0;

/// A pending term not confirmed within this many passes is dropped.
pub const PENDING_TTL: u64 = 3;

/// How many transcript messages the detector is shown, newest first.
pub const MAX_TRANSCRIPT_MESSAGES: usize = 12;

/// Default extraction model when `keyterm_detection.llm` is unset (not the agent's own LLM).
pub const DEFAULT_DETECTION_MODEL: &str = "google/gemma-4-31b-it";

/// The tool the detector forces. Its arguments are the whole output of a pass.
pub const RECORD_KEYTERMS_TOOL: &str = "record_keyterms";

/// The built-in extraction prompt, verbatim: every spelling rule lives in this string.
pub const DEFAULT_KEYTERM_INSTRUCTIONS: &str = r#"You maintain STT keyterms that bias a recognizer toward the correct spelling of distinctive words (names, places, companies, products, technical terms). Each turn, adjust them with one `record_keyterms` call.

A WRONG spelling biases the recognizer for the rest of the call with no recovery, so precision beats coverage: apply only a spelling you can CORROBORATE, and when unsure change nothing.

USER lines are raw STT — often wrong, and the same error recurs, so repetition is NOT proof a spelling is right. ASSISTANT lines are the agent's own writing: trust the agent's confident use of its OWN names (brands, staff, locations) and confirm those promptly — but an assistant merely echoing the user's sounds, or hedging about a spelling, does NOT corroborate.

CONFIRM a pending term only when corroborated by one of:
  1. a letter-by-letter spell-out the assistant then accepts WITHOUT reservation — confirm exactly those letters, appending nothing;
  2. the assistant's own confident use of that exact distinctive spelling;
  3. an explicit user correction ("no, not X — it's Y").
Recurrence alone never confirms.

HEDGE RULE: if after a spell-out or name read-back the assistant signals the letters may be off ("for now", "with that caveat", "may have that slightly off", "did I catch that?", "to be confirmed", "I don't want to guess", "double-check"), the spelling is unreliable — keep the term PENDING and never confirm it, EVEN IF the user replies "yes". Only a cleanly accepted spell-out confirms.

Never apply: a user-line word that sounds like a known term (it's that term misheard); a distinctive name glued to an ordinary word ("Blue Haven Hotel" — keep the bare name pending); an odd phrase only the user says and the assistant never adopts; a fragment left by an interruption; ordinary words or fillers.

Report only CHANGES; never re-list an applied term.
  - `pending`: a distinctive term seen but not yet corroborated;
  - `confirm`: a pending term that just met the bar above;
  - `remove`: only a spelling the user just corrected away. Applied terms are otherwise sticky.
If nothing meets the bar this turn, change nothing."#;

/// `DriverBuilder::keyterms`: the resolved `stt_context_options` plus the host objects a pass needs.
pub struct KeytermSetup {
    /// `_resolve_detection(options["keyterm_detection"])`.
    pub detection: KeytermDetection,
    /// `stt_context_options["keyterms"]`: the user's static list.
    pub static_keyterms: Vec<String>,
    /// The detection LLM; `None` skips every pass, as the reference's `start()` does.
    pub llm: Option<Arc<dyn Llm>>,
    /// The host's `[record_keyterms]` tool list, for an LLM that needs the objects beside the spec.
    pub tools_host: Option<HostRef>,
}

pub(crate) struct KeytermSlot {
    pass: KeytermPass,
    detection: KeytermDetection,
    llm: Option<ResourceId>,
    tools_host: Option<HostRef>,
    inflight: Option<u64>,
    /// Whether an STT without the `keyterms` capability was told so once already.
    warned_unsupported: bool,
    /// The pass in flight's `keyterm_detection`, until it answers or fails.
    span: Option<crate::telemetry::Span<crate::telemetry::KeytermDetection>>,
}

/// The `record_keyterms` spec (`keyterm_detection.STTContextOptions`): three string lists, forced.
fn record_keyterms_spec() -> lk_agents_abi::ToolSpec {
    lk_agents_abi::ToolSpec {
        provider_tool: None,
        name: RECORD_KEYTERMS_TOOL.to_owned(),
        description: "Update the STT keyterms based on the latest transcript.\n\n\
            Args:\n    pending: Distinctive terms seen but not yet trusted — tracked, not applied.\n    \
            confirm: Pending terms the transcript has now corroborated — applied.\n    \
            remove: Only a spelling the user corrected away; applied terms are otherwise sticky."
            .to_owned(),
        json_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "pending": { "type": "array", "items": { "type": "string" } },
                "confirm": { "type": "array", "items": { "type": "string" } },
                "remove": { "type": "array", "items": { "type": "string" } },
            },
            "required": ["pending", "confirm", "remove"],
        }),
        flags: lk_agents_abi::ToolFlags::default(),
    }
}

impl KeytermSlot {
    pub(crate) fn resource(&self) -> Option<ResourceId> {
        self.llm
    }
    /// `KeytermDetector.start(session, stt)`: the slot, and the static terms to push at once.
    pub(crate) fn start(setup: KeytermSetup, llm: Option<ResourceId>) -> (Self, Vec<String>) {
        let pass = KeytermPass::new(&setup.detection, &setup.static_keyterms);
        let initial = pass.keyterms();
        let slot = KeytermSlot {
            pass,
            detection: setup.detection,
            llm,
            tools_host: setup.tools_host,
            inflight: None,
            warned_unsupported: false,
            span: None,
        };
        (slot, initial)
    }

    /// Whether a pass could run at all; the STT must also take keyterms.
    pub(crate) fn detecting(&self) -> bool {
        self.detection.enabled && self.llm.is_some()
    }

    /// One transition of the pass; what it asks for is the caller's to perform.
    pub(crate) fn step(&mut self, input: KeytermIn) -> Vec<KeytermEffect> {
        let pass = std::mem::replace(&mut self.pass, KeytermPass::Closed);
        let (next, out) = pass.step(input);
        self.pass = next;
        out
    }

    /// The answer to `req` arrived: `true` when it was the request in flight (its timer goes).
    pub(crate) fn answered(&mut self, req: u64) -> bool {
        if self.inflight == Some(req) {
            self.inflight = None;
            return true;
        }
        false
    }

    /// The request in flight timed out; runtime cancels its resource.
    pub(crate) fn time_out(&mut self) -> Option<u64> {
        self.inflight.take()
    }

    pub(crate) fn cancel(&mut self, req: u64) {
        if self.inflight == Some(req) {
            self.inflight = None;
        }
    }

    /// One request: `llm.chat(.., tools=[record_keyterms], tool_choice="required")`.
    pub(crate) fn request(
        &self,
        prompt: String,
        wall: std::time::SystemTime,
    ) -> lk_agents_abi::ChatRequest {
        let chat_ctx = ChatContext::new(vec![
            ChatItem::Message(ChatMessage::text(
                "keyterms_system",
                Role::System,
                self.detection.instructions(),
                wall,
            )),
            ChatItem::Message(ChatMessage::text("keyterms_user", Role::User, prompt, wall)),
        ]);
        lk_agents_abi::ChatRequest {
            chat_ctx,
            tools: vec![record_keyterms_spec()],
            tools_host: self.tools_host.clone(),
            // One attempt: a pass that fails makes no change and the next gate runs another.
            conn_options: lk_agents_abi::ApiConnectOptions { max_retry: 0, ..Default::default() },
            tool_choice: Some(lk_agents_abi::ToolChoice::Required),
            ..Default::default()
        }
    }
}

/// A provider operation's physical outcome, with its observation time supplied separately.
pub enum DetectionPassEvent {
    Chunk(lk_agents_abi::LlmChunk),
    OpenedFailed,
    Ended,
    StreamFailed,
}

/// The first extraction call and the last usage snapshot of one keyterm pass.
pub struct DetectionPass {
    req: u64,
    provider: lk_agents_abi::ProviderId,
    started_at: f64,
    arguments: Option<String>,
    usage: Option<lk_agents_abi::usage::TokenUsage>,
    request_id: String,
    ttft: Option<f64>,
    settled: bool,
}

impl DetectionPass {
    pub fn new(req: u64, provider: lk_agents_abi::ProviderId, started_at: f64) -> Self {
        Self {
            req,
            provider,
            started_at,
            arguments: None,
            usage: None,
            request_id: String::new(),
            ttft: None,
            settled: false,
        }
    }

    /// Opening failed, including when the runtime no longer holds the requested provider.
    pub fn opening_failed(req: u64) -> GenEvent {
        GenEvent::KeytermAnswer { req, arguments: None, failed: true, metrics: None }
    }

    pub fn handle(&mut self, event: DetectionPassEvent, now: f64) -> Option<GenEvent> {
        if self.settled {
            return None;
        }
        match event {
            DetectionPassEvent::Chunk(chunk) => {
                if self.ttft.is_none() {
                    self.ttft = Some((now - self.started_at).max(0.0));
                }
                if self.request_id.is_empty() {
                    self.request_id = chunk.id;
                }
                if let Some(delta) = chunk.delta
                    && self.arguments.is_none()
                    && let Some(call) =
                        delta.tool_calls.into_iter().find(|call| call.name == RECORD_KEYTERMS_TOOL)
                {
                    self.arguments = Some(call.arguments);
                }
                if chunk.usage.is_some() {
                    self.usage = chunk.usage.map(|usage| *usage);
                }
                None
            }
            DetectionPassEvent::OpenedFailed => {
                self.settled = true;
                Some(Self::opening_failed(self.req))
            }
            DetectionPassEvent::Ended => {
                self.settled = true;
                Some(self.finish(false, now))
            }
            DetectionPassEvent::StreamFailed => {
                self.settled = true;
                Some(self.finish(true, now))
            }
        }
    }

    fn finish(&mut self, failed: bool, now: f64) -> GenEvent {
        let duration = (now - self.started_at).max(0.0);
        let tokens = self.usage.take().unwrap_or_default();
        let metrics = (!failed).then(|| {
            Box::new(LlmMetrics {
                label: self.provider.label.clone(),
                request_id: std::mem::take(&mut self.request_id),
                timestamp: now,
                duration,
                ttft: self.ttft,
                cancelled: false,
                tokens_per_second: if duration > 0.0 && tokens.output_tokens > 0 {
                    tokens.output_tokens as f64 / duration
                } else {
                    0.0
                },
                tokens,
                speech_id: None,
                metadata: Some(self.provider.metadata()),
            })
        });
        GenEvent::KeytermAnswer { req: self.req, arguments: self.arguments.take(), failed, metrics }
    }
}

// ---------------------------------------------------------------------------------------------

// Options
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct KeytermDetection {
    pub enabled: bool,
    pub llm: Option<DetectionLlm>,
    /// `max(1, turn_interval)`: `0` means every turn.
    pub turn_interval: u32,
    pub max_keyterms: Option<usize>,
    /// `None` when unset or empty: an empty override means the default prompt.
    pub instructions: Option<String>,
    pub timeout: f64,
}

impl Default for KeytermDetection {
    fn default() -> Self {
        KeytermDetection {
            enabled: false,
            llm: None,
            turn_interval: 1,
            max_keyterms: None,
            instructions: None,
            timeout: DETECTION_TIMEOUT,
        }
    }
}

impl KeytermDetection {
    pub fn instructions(&self) -> &str {
        self.instructions.as_deref().unwrap_or(DEFAULT_KEYTERM_INSTRUCTIONS)
    }
}

/// Keyterm detection's resolution, as the host calls it.
pub struct KeytermRules;

impl lk_agents_abi::turn::keyterms::KeytermFunctions for KeytermRules {
    fn resolve_keyterm_detection(
        config: Option<KeytermDetectionConfig>,
    ) -> ResolvedKeytermDetection {
        let r = resolve_detection(config.as_ref());
        ResolvedKeytermDetection {
            enabled: r.enabled,
            turn_interval: r.turn_interval,
            max_keyterms: r.max_keyterms,
            timeout: r.timeout,
            instructions: r.instructions().to_owned(),
            model: resolve_detection_model(r.llm.as_ref()).map(str::to_owned),
            tool_name: RECORD_KEYTERMS_TOOL.to_owned(),
        }
    }
}

/// `_resolve_detection`: defaults, then whatever keys the caller set.
pub fn resolve_detection(config: Option<&KeytermDetectionConfig>) -> KeytermDetection {
    let d = KeytermDetection::default();
    let Some(c) = config else { return d };
    KeytermDetection {
        enabled: c.enabled.unwrap_or(d.enabled),
        llm: c.llm.clone(),
        // `max(1, ...)` from the reference's constructor, folded in here.
        turn_interval: c.turn_interval.unwrap_or(d.turn_interval).max(1),
        max_keyterms: c.max_keyterms,
        // Empty string -> the default prompt, per `options["instructions"] or DEFAULT`.
        instructions: c.instructions.clone().filter(|s| !s.is_empty()),
        timeout: c.timeout.unwrap_or(d.timeout),
    }
}

/// Which model to build, or `None` when the caller supplied an `LLM` object. Construction failures
/// are swallowed host-side, as `_resolve_detection_llm` does.
pub fn resolve_detection_model(configured: Option<&DetectionLlm>) -> Option<&str> {
    match configured {
        Some(DetectionLlm::Instance) => None,
        Some(DetectionLlm::Model(m)) => Some(m),
        None => Some(DEFAULT_DETECTION_MODEL),
    }
}

// --- The pass counter ---

/// Decides which user turns start a pass. Per activity (reset on `start`), unlike the keyterm state.
#[derive(Debug, Clone)]
pub struct TurnGate {
    turn_interval: u32,
    turn_count: u64,
}

impl TurnGate {
    pub fn new(turn_interval: u32) -> Self {
        TurnGate { turn_interval: turn_interval.max(1), turn_count: 0 }
    }

    /// `true` when this item should start a pass. Empty text isn't a turn (truthiness). The driver's
    /// single-flight check runs after this, so a turn during a pass still advances the counter.
    pub fn on_conversation_item(&mut self, role: Role, text: &str) -> bool {
        if role != Role::User || text.is_empty() {
            return false;
        }
        self.turn_count += 1;
        self.turn_count.is_multiple_of(u64::from(self.turn_interval))
    }
}

// --- The calculator ---
/// The keyterm set and the rules that grow it; owned by the session, so it survives handoffs.
#[derive(Debug, Clone, Default)]
pub struct KeytermState {
    static_terms: Vec<String>,
    /// Confirmed terms, oldest first: the order *is* the eviction order.
    detected: Vec<String>,
    /// Term -> the pass it was added on, in insertion order (shown to the detector in that order).
    pending: Vec<(String, u64)>,
    /// Detection-pass counter. Advances per LLM answer, which is what the TTL is measured in.
    tick: u64,
    max_keyterms: Option<usize>,
}

impl KeytermState {
    pub fn new(static_keyterms: &[String], max_keyterms: Option<usize>) -> Self {
        KeytermState {
            static_terms: dedup(static_keyterms.iter().map(String::as_str)),
            detected: Vec::new(),
            pending: Vec::new(),
            tick: 0,
            max_keyterms,
        }
    }

    /// The effective list applied to the STT: static terms + confirmed detected terms.
    pub fn keyterms(&self) -> Vec<String> {
        dedup(self.static_terms.iter().chain(self.detected.iter()).map(String::as_str))
    }

    /// Replace the static terms and return the effective list (pushed unconditionally).
    pub fn set_static_keyterms(&mut self, terms: &[String]) -> Vec<String> {
        self.static_terms = dedup(terms.iter().map(String::as_str));
        self.keyterms()
    }

    /// Current state for the detector: `(term, applied)`, applied first; static terms count as applied.
    pub fn current(&self) -> Vec<(String, bool)> {
        self.static_terms
            .iter()
            .map(|t| (t.clone(), true))
            .chain(self.detected.iter().map(|t| (t.clone(), true)))
            .chain(self.pending.iter().map(|(t, _)| (t.clone(), false)))
            .collect()
    }

    /// Fold one LLM answer in; `Some` only when the effective list changed. Order matters: advance
    /// the pass counter; `remove` (never static); `pending` (after remove, so a term in both is
    /// pending again); `confirm` appends to applied; expire past [`PENDING_TTL`]; evict the oldest
    /// over `max_keyterms`.
    pub fn apply(&mut self, update: &KeytermUpdate) -> Option<KeytermsChanged> {
        let before = self.keyterms();
        self.tick += 1;

        for term in &update.remove {
            self.pending.retain(|(t, _)| t != term);
            self.detected.retain(|t| t != term);
        }

        for term in &update.pending {
            // `if term and ...`: an empty string is not a candidate.
            if term.is_empty() || self.static_terms.contains(term) {
                continue;
            }
            if !self.detected.contains(term) && !self.pending.iter().any(|(t, _)| t == term) {
                self.pending.push((term.clone(), self.tick));
            }
        }

        for term in &update.confirm {
            if term.is_empty() || self.static_terms.contains(term) {
                continue;
            }
            self.pending.retain(|(t, _)| t != term);
            if !self.detected.contains(term) {
                self.detected.push(term.clone());
            }
        }

        let tick = self.tick;
        self.pending.retain(|(_, added)| tick - added < PENDING_TTL);

        if let Some(max) = self.max_keyterms {
            while self.detected.len() > max {
                self.detected.remove(0);
            }
        }

        let after = self.keyterms();
        if after == before {
            return None;
        }
        Some(KeytermsChanged {
            added: after.iter().filter(|t| !before.contains(t)).cloned().collect(),
            removed: before.iter().filter(|t| !after.contains(t)).cloned().collect(),
            keyterms: after,
        })
    }
}

/// `list(dict.fromkeys(...))`: first occurrence wins, order preserved.
fn dedup<'a>(terms: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for t in terms {
        if !out.iter().any(|e| e == t) {
            out.push(t.to_string());
        }
    }
    out
}

// --- Rendering the pass input ---

/// One transcript line; `text` is `text_content`, markup already stripped by the caller.
#[derive(Debug, Clone)]
pub struct TranscriptItem<'a> {
    pub role: Role,
    pub text: &'a str,
}

/// `_format_input`: recent transcript plus current keyterms, `None` with no user/assistant text.
/// Walked newest-first so a long call is truncated at the old end.
pub fn format_detector_input(
    items: &[TranscriptItem<'_>],
    current: &[(&str, bool)],
) -> Option<String> {
    let mut turns: Vec<String> = Vec::new();
    for item in items.iter().rev() {
        if !matches!(item.role, Role::User | Role::Assistant) {
            continue;
        }
        if item.text.is_empty() {
            continue;
        }
        // Drop blank lines so the blank line between turns marks the boundary.
        let body =
            item.text.lines().filter(|l| !l.trim().is_empty()).collect::<Vec<_>>().join("\n");
        turns.push(format!("{}: {}", role_upper(item.role), body));
        if turns.len() >= MAX_TRANSCRIPT_MESSAGES {
            break;
        }
    }
    if turns.is_empty() {
        return None;
    }
    turns.reverse();

    let applied = join_or_none(current.iter().filter(|(_, ok)| *ok).map(|(t, _)| *t));
    let candidates = join_or_none(current.iter().filter(|(_, ok)| !*ok).map(|(t, _)| *t));
    // Always show both lists (even empty) so the model has explicit state to diff against.
    Some(format!(
        "## Transcript (USER = raw STT, may be wrong; ASSISTANT = correct spelling)\n{}\n\n\
         ## Applied keyterms (biasing the recognizer now)\n{}\n\n\
         ## Candidate keyterms (seen, not yet applied)\n{}\n\n\
         Update the keyterms from the latest turns, then call `record_keyterms` once.",
        turns.join("\n\n"),
        applied,
        candidates,
    ))
}

fn role_upper(role: Role) -> &'static str {
    match role {
        Role::User => "USER",
        Role::Assistant => "ASSISTANT",
        Role::System => "SYSTEM",
        Role::Developer => "DEVELOPER",
    }
}

/// `", ".join(terms) or "(none)"`: the empty join is falsy, so it becomes the placeholder.
fn join_or_none<'a>(terms: impl Iterator<Item = &'a str>) -> String {
    let joined = terms.collect::<Vec<_>>().join(", ");
    if joined.is_empty() { "(none)".to_string() } else { joined }
}

// --- Reading the pass output ---

/// `_parse_tool_call`'s second half: the filtered `record_keyterms` arguments, `None` on no call or
/// unparsable arguments. Terms are kept unstripped, as the reference sends them; a non-list value
/// yields nothing.
pub fn parse_record_keyterms(args: Option<&serde_json::Value>) -> KeytermUpdate {
    let Some(args) = args else { return KeytermUpdate::default() };
    let terms = |key: &str| -> Vec<String> {
        args.get(key)
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .filter(|s| !s.trim().is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    };
    KeytermUpdate { pending: terms("pending"), confirm: terms("confirm"), remove: terms("remove") }
}

// --- The pass: single-flight as a state, the request and the push as effects ---

#[derive(Debug, Clone, PartialEq, Vocabulary)]
pub enum KeytermIn {
    /// A committed conversation item. Only a non-empty *user* item advances the turn gate.
    Item {
        role: Role,
        text: String,
    },
    Answer {
        req: u64,
        update: KeytermUpdate,
    },
    /// Request `req` failed or timed out: no change, the single-flight slot frees.
    Failed {
        req: u64,
    },
    /// The static list changed: the effective list is pushed unconditionally.
    StaticTerms(Vec<String>),
    /// A handoff. A different STT gets the full set re-pushed; the same one keeps it.
    AgentUpdated {
        same_stt: bool,
    },
    Close,
}

#[derive(Debug, Clone)]
pub struct Pass {
    gate: TurnGate,
    state: KeytermState,
    /// The recent transcript, chronological, capped so a long call is truncated at the old end.
    transcript: Vec<(Role, String)>,
    next_req: u64,
}

/// The keyterm pass: one pass over the STT transcript at a time.
///
/// | state \ event | `Item` | `Answer` | `Failed` | `StaticTerms` | `AgentUpdated` | `Close` |
/// |---|---|---|---|---|---|---|
/// | `Idle`    | record; gate due -> `Request` -> `Pending` | stale, ignored | stale, ignored | set, `Push` | different STT -> `Push` | -> `Closed` |
/// | `Pending` | record (the gate still advances; no second request) | own req -> apply, changed -> `Push`, -> `Idle`; other -> ignored | own req -> `Idle`; other -> ignored | set, `Push` | different STT -> `Push` | `Cancel` -> `Closed` |
/// | `Closed`  | - | - | - | - | - | - |
#[derive(Debug, Clone)]
pub enum KeytermPass {
    Idle(Pass),
    Pending { pass: Pass, req: u64 },
    Closed,
}

impl KeytermPass {
    pub fn new(detection: &KeytermDetection, static_keyterms: &[String]) -> Self {
        KeytermPass::Idle(Pass {
            gate: TurnGate::new(detection.turn_interval),
            state: KeytermState::new(static_keyterms, detection.max_keyterms),
            transcript: Vec::new(),
            next_req: 1,
        })
    }

    /// The effective list (static + confirmed), or empty once closed.
    pub fn keyterms(&self) -> Vec<String> {
        match self {
            KeytermPass::Idle(p) | KeytermPass::Pending { pass: p, .. } => p.state.keyterms(),
            KeytermPass::Closed => Vec::new(),
        }
    }

    pub fn step(self, input: KeytermIn) -> (KeytermPass, Vec<KeytermEffect>) {
        let mut out = Vec::new();
        let next = match (self, input) {
            // --------------------------------------------------------------------------- Idle
            (KeytermPass::Idle(mut p), KeytermIn::Item { role, text }) => {
                let due = p.record(role, &text);
                match due.then(|| p.request(&mut out)).flatten() {
                    Some(req) => KeytermPass::Pending { pass: p, req },
                    None => KeytermPass::Idle(p),
                }
            }
            (KeytermPass::Idle(p), KeytermIn::Answer { .. } | KeytermIn::Failed { .. }) => {
                KeytermPass::Idle(p)
            }
            (KeytermPass::Idle(mut p), KeytermIn::StaticTerms(terms)) => {
                p.set_static(&terms, &mut out);
                KeytermPass::Idle(p)
            }
            (KeytermPass::Idle(p), KeytermIn::AgentUpdated { same_stt }) => {
                p.on_agent_updated(same_stt, &mut out);
                KeytermPass::Idle(p)
            }
            (KeytermPass::Idle(_), KeytermIn::Close) => KeytermPass::Closed,
            // ------------------------------------------------------------------------ Pending
            (KeytermPass::Pending { mut pass, req }, KeytermIn::Item { role, text }) => {
                // The counter advances; the pass in flight is the single flight.
                pass.record(role, &text);
                KeytermPass::Pending { pass, req }
            }
            (
                KeytermPass::Pending { mut pass, req },
                KeytermIn::Answer { req: answered, update },
            ) => {
                if answered != req {
                    return (KeytermPass::Pending { pass, req }, out);
                }
                if let Some(changed) = pass.state.apply(&update) {
                    out.push(KeytermEffect::Push(changed.keyterms));
                }
                KeytermPass::Idle(pass)
            }
            (KeytermPass::Pending { pass, req }, KeytermIn::Failed { req: failed }) => {
                if failed == req {
                    KeytermPass::Idle(pass)
                } else {
                    KeytermPass::Pending { pass, req }
                }
            }
            (KeytermPass::Pending { mut pass, req }, KeytermIn::StaticTerms(terms)) => {
                pass.set_static(&terms, &mut out);
                KeytermPass::Pending { pass, req }
            }
            (KeytermPass::Pending { pass, req }, KeytermIn::AgentUpdated { same_stt }) => {
                pass.on_agent_updated(same_stt, &mut out);
                KeytermPass::Pending { pass, req }
            }
            (KeytermPass::Pending { pass: _, req }, KeytermIn::Close) => {
                out.push(KeytermEffect::Cancel { req });
                KeytermPass::Closed
            }
            (KeytermPass::Closed, _) => KeytermPass::Closed,
        };
        (next, out)
    }
}

impl Pass {
    /// Record one item; returns whether the gate says a pass is due.
    fn record(&mut self, role: Role, text: &str) -> bool {
        if matches!(role, Role::User | Role::Assistant) && !text.is_empty() {
            self.transcript.push((role, text.to_owned()));
            // Twice the window the prompt shows: enough that role filtering never starves it.
            let cap = MAX_TRANSCRIPT_MESSAGES * 2;
            if self.transcript.len() > cap {
                let drop = self.transcript.len() - cap;
                self.transcript.drain(..drop);
            }
        }
        self.gate.on_conversation_item(role, text)
    }

    /// One request, or `None` when there is no user/assistant text to show yet.
    fn request(&mut self, out: &mut Vec<KeytermEffect>) -> Option<u64> {
        let items: Vec<TranscriptItem<'_>> = self
            .transcript
            .iter()
            .map(|(role, text)| TranscriptItem { role: *role, text })
            .collect();
        let current = self.state.current();
        let current: Vec<(&str, bool)> =
            current.iter().map(|(t, applied)| (t.as_str(), *applied)).collect();
        let prompt = format_detector_input(&items, &current)?;
        let req = self.next_req;
        self.next_req += 1;
        out.push(KeytermEffect::Request { req, prompt });
        Some(req)
    }

    fn set_static(&mut self, terms: &[String], out: &mut Vec<KeytermEffect>) {
        let list = self.state.set_static_keyterms(terms);
        out.push(KeytermEffect::Push(list));
    }

    /// A different STT starts from nothing: re-push the full set. The same one already has it.
    fn on_agent_updated(&self, same_stt: bool, out: &mut Vec<KeytermEffect>) {
        if same_stt {
            return;
        }
        let list = self.state.keyterms();
        if !list.is_empty() {
            out.push(KeytermEffect::Push(list));
        }
    }
}

/// The detector as the driver runs it: started with the session, fed the conversation's items,
/// its requests run off the loop and bounded.
impl VoiceDriver {
    /// `KeytermDetector.start(session, stt)`: the state survives handoffs, and static terms reach
    /// the recogniser at once.
    pub(crate) fn keyterms_start(&mut self, setup: KeytermSetup, llm: Option<ResourceId>) {
        let (slot, initial) = KeytermSlot::start(setup, llm);
        self.session.providers.extend(llm);
        self.session.keyterms = Some(slot);
        if !initial.is_empty() {
            self.push_keyterms(initial);
        }
    }

    pub(crate) fn keyterms_step(&mut self, now: Ts, input: KeytermIn) {
        let Some(slot) = self.session.keyterms.as_mut() else { return };
        let effects = slot.step(input);
        self.perform_keyterms(now, effects);
    }

    /// `_on_conversation_item_added`; fed only while a pass could run, as the reference subscribes.
    pub(crate) fn keyterms_on_item(&mut self, now: Ts, item: &ChatItem) {
        let detecting = self.session.keyterms.as_ref().is_some_and(KeytermSlot::detecting)
            && self.session.stt.as_ref().is_some_and(|stt| stt.capabilities.keyterms);
        if !detecting {
            return;
        }
        let ChatItem::Message(m) = item else { return };
        let text = m.raw_text_content().unwrap_or_default();
        self.keyterms_step(now, KeytermIn::Item { role: m.role, text });
    }

    /// The STT leg changed under a handoff: a different recogniser gets the full list.
    pub(crate) fn keyterms_on_stt_swap(&mut self, now: Ts, same_stt: bool) {
        if self.session.keyterms.is_none() {
            return;
        }
        self.record_trace(DetectorStep::KeytermsRebound { same_stt });
        self.keyterms_step(now, KeytermIn::AgentUpdated { same_stt });
    }

    pub(crate) fn on_keyterm_answer(
        &mut self,
        now: Ts,
        req: u64,
        arguments: Option<String>,
        failed: bool,
        metrics: Option<Box<LlmMetrics>>,
    ) {
        let Some(slot) = self.session.keyterms.as_mut() else { return };
        if !slot.answered(req) {
            return;
        }
        self.record_effect(crate::agent_session::effects::VoiceEffect::CancelKeyterms(req));
        self.clear_deadline(Timer::KeytermTimeout);
        // Rides the session stream like the reply LLM's metrics, with no speech.
        if let Some(m) = &metrics {
            self.publish_event(crate::events::SessionEvent::MetricsCollected {
                metrics: Box::new(lk_agents_abi::metrics::base::AgentMetrics::Llm((**m).clone())),
                speech: None,
                counted: false,
            });
        }
        if failed {
            return self.keyterms_failed(now, req);
        }
        self.record_trace(InputStep::LegVerdict { leg: DetectorName::Keyterms, req });
        // Strict JSON; a malformed document is an empty update.
        let parsed =
            arguments.as_deref().and_then(|a| serde_json::from_str::<serde_json::Value>(a).ok());
        let update = parse_record_keyterms(parsed.as_ref());
        let before = self.session.keyterms.as_ref().map(|s| s.pass.keyterms()).unwrap_or_default();
        self.keyterms_step(now, KeytermIn::Answer { req, update });
        let after = self.session.keyterms.as_ref().map(|s| s.pass.keyterms()).unwrap_or_default();
        self.keyterm_span_ended(Some((before.as_slice(), after.as_slice())));
    }

    /// `asyncio.wait_for(.., timeout)` expired: no change, the slot frees.
    pub(crate) fn on_keyterm_timeout(&mut self, now: Ts) {
        let Some(slot) = self.session.keyterms.as_mut() else { return };
        let Some(req) = slot.time_out() else { return };
        self.record_effect(crate::agent_session::effects::VoiceEffect::CancelKeyterms(req));
        self.keyterms_failed(now, req);
    }

    fn keyterms_failed(&mut self, now: Ts, req: u64) {
        self.record_trace(InputStep::LegFailed { leg: DetectorName::Keyterms, recoverable: true });
        self.keyterm_span_ended(None);
        self.keyterms_step(now, KeytermIn::Failed { req });
    }

    fn perform_keyterms(&mut self, now: Ts, effects: Vec<KeytermEffect>) {
        for effect in effects {
            match effect {
                KeytermEffect::Request { req, prompt } => self.request_keyterms(now, req, prompt),
                KeytermEffect::Push(terms) => self.push_keyterms(terms),
                KeytermEffect::Cancel { req } => {
                    if let Some(slot) = self.session.keyterms.as_mut() {
                        slot.cancel(req);
                    }
                    self.record_effect(crate::agent_session::effects::VoiceEffect::CancelKeyterms(
                        req,
                    ));
                    self.clear_deadline(Timer::KeytermTimeout);
                }
            }
        }
    }

    fn push_keyterms(&mut self, terms: Vec<String>) {
        let stt = self.session.stt.as_ref();
        let warn_unsupported = stt.is_some_and(|stt| !stt.capabilities.keyterms)
            && self
                .session
                .keyterms
                .as_mut()
                .is_some_and(|slot| !std::mem::replace(&mut slot.warned_unsupported, true));
        self.record_effect(crate::agent_session::effects::VoiceEffect::PushKeyterms {
            stt: stt.map(|stt| stt.id),
            terms: terms.clone(),
            warn_unsupported,
        });
        self.record_trace(DetectorStep::KeytermsPushed { terms });
    }

    /// One request, bounded by the pass's timeout, run off the loop.
    fn request_keyterms(&mut self, now: Ts, req: u64, prompt: String) {
        let Some(slot) = self.session.keyterms.as_mut() else { return };
        let Some(llm) = slot.llm else {
            // The machine asked with no model to ask: fed back as a failure so the slot frees.
            self.keyterms_step(now, KeytermIn::Failed { req });
            return;
        };
        // Build the request while borrowing its owner, then release that borrow before
        // recording driver work. No host code is called while the slot is borrowed.
        let timeout = std::time::Duration::from_secs_f64(slot.detection.timeout.max(0.0));
        let request = slot.request(prompt, self.step.wall());
        slot.inflight = Some(req);
        self.record_trace(InputStep::LegRequested { leg: DetectorName::Keyterms, req });
        self.set_deadline(Timer::KeytermTimeout, now.saturating_add(timeout));
        let model = self.session.llm.as_ref().filter(|l| l.id == llm).map(|l| &l.provider);
        let decl = telemetry::KeytermDetection {
            model: model.map(|l| l.model.clone()).unwrap_or_default(),
            provider: model
                .and_then(|l| lk_agents_core::telemetry::gen_ai::provider_name(&l.provider))
                .unwrap_or_default(),
        };
        // A pass still open (its answer never came) ends first.
        self.keyterm_span_ended(None);
        let span = self.session.span.as_ref().map(|root| self.telemetry.open_in(root, decl));
        let context = span.as_ref().and_then(|s| self.telemetry.context(s));
        if let Some(slot) = self.session.keyterms.as_mut() {
            slot.span = span;
        } else if let Some(span) = span {
            self.telemetry
                .end(span, telemetry::KeytermDetectionEnd { count: 0, added: 0, removed: 0 });
        }
        self.record_effect(crate::agent_session::effects::VoiceEffect::RequestKeyterms {
            llm,
            req,
            request: Box::new(request),
            span: context,
        });
    }

    /// The pass answered with the effective lists before and after it, or failed (`None`): its
    /// `keyterm_detection` ends.
    pub(crate) fn keyterm_span_ended(&mut self, lists: Option<(&[String], &[String])>) {
        let Some(span) = self.session.keyterms.as_mut().and_then(|s| s.span.take()) else {
            return;
        };
        let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        let end = match lists {
            Some((before, after)) => telemetry::KeytermDetectionEnd {
                count: count(after.len()),
                added: count(after.iter().filter(|t| !before.contains(t)).count()),
                removed: count(before.iter().filter(|t| !after.contains(t)).count()),
            },
            None => telemetry::KeytermDetectionEnd { count: 0, added: 0, removed: 0 },
        };
        self.telemetry.end(span, end);
    }
}
