//! The session's trace (`telemetry/traces.py`'s span tree): which OTel spans open and close, under
//! which parent, with which attributes and duration metrics. Decided here from the session's own
//! events and the driver's hook calls; the host only materialises each `span_started` /
//! `span_ended` record with its OTel SDK.
//!
//! ```text
//! agent_session                         the session, construction to `close`
//! ├── session_start                     construction to the opening cycle
//! │   └── start_agent_activity > on_enter
//! ├── user_turn                         first speech of a turn to its commit
//! │   ├── user_speaking
//! │   └── eou_wait > eou_detection      end of speech to the decision; one prediction
//! ├── on_user_turn_completed
//! ├── agent_turn                        one speech, created to done; a `generation` per step
//! │   ├── agent_speaking
//! │   ├── llm_node > llm_request > llm_request_run
//! │   ├── tts_node > tts_request > tts_request_run
//! │   ├── realtime_inference > realtime_metrics
//! │   └── function_tool
//! ├── update_agent                      a handoff: the old agent's exit to the new one's entry
//! │   ├── drain_agent_activity > on_exit
//! │   └── start_agent_activity > on_enter
//! ├── keyterm_detection > llm_request > llm_request_run
//! ├── amd                               the AMD detector's attach to its verdict
//! └── session_close > drain_agent_activity > on_exit
//! ```
//!
//! The inference spans are dated from their metrics record (`timestamp - duration`), which the
//! driver emits when the request ends; a request with no speech (a warmup, a detector's) has no
//! node to nest under and opens none, unless its owner nests it (`keyterm_answered`).

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{HookId, HookKind};
use crate::events::{SessionEvent, SpeechOutcome, ToolEndStatus, ToolLifecycle};
use crate::speech_handle::SpeechHandle;
use lk_agents_abi::metrics::base::{AgentMetrics, LlmMetrics, Metadata};
use lk_agents_core::telemetry::gen_ai::provider_name;
use lk_agents_abi::voice::events::{
    AgentState, SpanEndedWire, SpanEvent, SpanMetric, SpanStartedWire, UserState,
};
use lk_agents_abi::{Extra, Ts};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

/// The root's id.
pub const ROOT_SPAN: &str = "agent_session";

/// Every span the session emits and the parents it may have (`None`: the host's ambient context),
/// as `tests/trace_schema.py`'s `SPAN_PARENTS` has them.
pub const SPAN_PARENTS: &[(&str, &[Option<&str>])] = &[
    ("agent_session", &[None]),
    ("session_start", &[Some("agent_session")]),
    ("session_close", &[Some("agent_session")]),
    ("update_agent", &[Some("agent_session")]),
    ("start_agent_activity", &[Some("session_start"), Some("update_agent")]),
    ("on_enter", &[Some("start_agent_activity")]),
    ("drain_agent_activity", &[Some("update_agent"), Some("session_close"), Some("agent_session")]),
    ("on_exit", &[Some("drain_agent_activity")]),
    ("user_turn", &[Some("agent_session")]),
    ("user_speaking", &[Some("user_turn"), Some("agent_session")]),
    ("eou_wait", &[Some("user_turn")]),
    ("eou_detection", &[Some("eou_wait")]),
    ("on_user_turn_completed", &[Some("user_turn"), Some("agent_session")]),
    ("agent_turn", &[Some("agent_session")]),
    ("agent_speaking", &[Some("agent_turn")]),
    ("llm_node", &[Some("agent_turn")]),
    ("tts_node", &[Some("agent_turn")]),
    ("function_tool", &[Some("agent_turn")]),
    ("realtime_inference", &[Some("agent_turn")]),
    ("realtime_metrics", &[Some("realtime_inference"), Some("agent_turn")]),
    ("llm_request", &[Some("llm_node"), Some("keyterm_detection"), Some("amd")]),
    ("llm_request_run", &[Some("llm_request")]),
    ("tts_request", &[Some("tts_node")]),
    ("tts_request_run", &[Some("tts_request")]),
    ("keyterm_detection", &[Some("agent_turn"), Some("agent_session")]),
    ("amd", &[Some("agent_session")]),
];

/// The edges where a child may end after its parent (`MAY_OUTLIVE_PARENT`).
pub const MAY_OUTLIVE_PARENT: &[(&str, &str)] = &[("on_enter", "start_agent_activity")];

/// An attribute list in insertion order, as the dashboard diffs it.
#[derive(Default)]
pub(crate) struct Attrs(Extra);

impl Attrs {
    pub(crate) fn new() -> Self {
        Attrs::default()
    }

    pub(crate) fn set(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.0.insert(key.to_owned(), value.into());
        self
    }

    pub(crate) fn opt(self, key: &str, value: Option<impl Into<Value>>) -> Self {
        match value {
            Some(v) => self.set(key, v),
            None => self,
        }
    }
}

/// A span still open: its start, for the duration metric and a back-dated end's floor.
struct Open {
    id: String,
    start: f64,
}

struct AgentTurn {
    open: Open,
    handle: Arc<SpeechHandle>,
    agent: String,
    /// The speech is done but a tool it started still runs: the turn ends with the last one.
    done: Option<SpeechOutcome>,
}

struct Tool {
    open: Open,
    name: String,
    speech: SpeechId,
}

/// A hook's span, and the activity span it runs in when that one ends with it.
struct Hook {
    open: Open,
    activity: Option<Open>,
}

/// The open spans, by what they belong to. Every open id is in `order`, so the close ends what is
/// left innermost first, `session_close` and `agent_session` last.
#[derive(Default)]
pub(crate) struct SessionTrace {
    /// When the session was built, wall seconds: `agent_session` starts there.
    pub(crate) built: f64,
    next: u64,
    root: Option<String>,
    starting: Option<Open>,
    closing: Option<String>,
    updating: Option<Open>,
    user_turn: Option<Open>,
    user_speaking: Option<Open>,
    eou_wait: Option<Open>,
    /// The `agent_speaking` span and the turn it is in.
    agent_speaking: Option<(Open, Option<SpeechId>)>,
    turns: BTreeMap<SpeechId, AgentTurn>,
    tools: BTreeMap<String, Tool>,
    hooks: BTreeMap<HookId, Hook>,
    floor: Option<SpeechId>,
    keyterm_pass: Option<f64>,
    amd: Option<Open>,
    order: Vec<String>,
    ended: bool,
}

impl SessionTrace {
    fn live(&self) -> bool {
        self.root.is_some() && !self.ended
    }

    fn start(
        &mut self,
        name: &str,
        parent: Option<String>,
        at: f64,
        attrs: Attrs,
        out: &mut Vec<SessionEvent>,
    ) -> Open {
        self.next += 1;
        let id = match parent {
            None => ROOT_SPAN.to_owned(),
            Some(_) => format!("{name}_{}", self.next),
        };
        self.order.push(id.clone());
        out.push(SessionEvent::SpanStarted(Box::new(SpanStartedWire {
            span_id: id.clone(),
            parent_id: parent,
            name: name.to_owned(),
            start_time: at,
            attributes: attrs.0,
        })));
        Open { id, start: at }
    }

    fn end_with(
        &mut self,
        open: Open,
        at: f64,
        attrs: Attrs,
        error: Option<String>,
        (events, metrics): (Vec<SpanEvent>, Vec<SpanMetric>),
        out: &mut Vec<SessionEvent>,
    ) {
        self.order.retain(|id| *id != open.id);
        out.push(SessionEvent::SpanEnded(Box::new(SpanEndedWire {
            span_id: open.id,
            end_time: at.max(open.start),
            attributes: attrs.0,
            error,
            events,
            metrics,
        })));
    }

    fn end(&mut self, open: Open, at: f64, out: &mut Vec<SessionEvent>) {
        self.end_with(open, at, Attrs::new(), None, Default::default(), out);
    }

    /// Spans whose bounds are already known, each nested in the one before: started outermost
    /// first, ended innermost first.
    fn spans(
        &mut self,
        parent: Option<String>,
        (from, to): (f64, f64),
        chain: Vec<(&str, Attrs)>,
        out: &mut Vec<SessionEvent>,
    ) {
        let mut parent = parent;
        let mut opened = Vec::new();
        for (name, attrs) in chain {
            let open = self.start(name, parent, from, attrs, out);
            parent = Some(open.id.clone());
            opened.push(open);
        }
        for open in opened.into_iter().rev() {
            self.end(open, to, out);
        }
    }

    fn root(&self) -> Option<String> {
        self.root.clone()
    }

    fn turn(&self, speech: SpeechId) -> Option<String> {
        self.turns.get(&speech).map(|t| t.open.id.clone())
    }

    /// `AgentSession.start`: `agent_session` and `session_start` from the session's construction;
    /// [`SessionTrace::opened`] ends `session_start`.
    pub(crate) fn open(&mut self, now: f64, agent: &str, out: &mut Vec<SessionEvent>) {
        if self.root.is_some() || self.ended {
            return;
        }
        let built = self.built.min(now);
        let attrs = Attrs::new()
            .set("lk.agent_label", agent)
            .set("gen_ai.operation.name", "invoke_workflow")
            .set("gen_ai.workflow.name", "agent_session");
        self.root = Some(self.start("agent_session", None, built, attrs, out).id);
        let parent = self.root();
        self.starting = Some(self.start("session_start", parent, built, Attrs::new(), out));
    }

    /// The opening cycle is done.
    pub(crate) fn opened(&mut self, now: f64, out: &mut Vec<SessionEvent>) {
        if let Some(open) = self.starting.take() {
            self.end(open, now, out);
        }
    }

    /// `_aclose_impl`'s `session_close`, from the request to the `close` event.
    pub(crate) fn closing(
        &mut self,
        now: f64,
        reason: &str,
        drain: bool,
        error: Option<&str>,
        out: &mut Vec<SessionEvent>,
    ) {
        if self.closing.is_some() || !self.live() {
            return;
        }
        let attrs = Attrs::new()
            .set("lk.close_reason", reason)
            .set("lk.close.drain", drain)
            .opt("exception.type", error);
        let parent = self.root();
        self.closing = Some(self.start("session_close", parent, now, attrs, out).id);
    }

    /// A `VoiceTask` suspends `previous`: its agent's `on_enter` runs under this `update_agent`,
    /// which that hook, or [`SessionTrace::agent_updated`] without one, ends.
    pub(crate) fn agent_suspended(
        &mut self,
        now: f64,
        previous: &str,
        out: &mut Vec<SessionEvent>,
    ) {
        if !self.live() || self.updating.is_some() {
            return;
        }
        let attrs = Attrs::new().set("lk.previous_agent_label", previous);
        let parent = self.root();
        self.updating = Some(self.start("update_agent", parent, now, attrs, out));
    }

    /// The update is over with no `on_enter` to end it.
    pub(crate) fn agent_updated(&mut self, now: f64, out: &mut Vec<SessionEvent>) {
        if let Some(update) = self.updating.take() {
            self.end(update, now, out);
        }
    }

    /// A callback the host was asked to run, in the activity span the reference runs it in:
    /// `on_enter` in `start_agent_activity`, `on_exit` in `drain_agent_activity` (a handoff's
    /// under `update_agent`), `on_user_turn_completed` in the session. Decided where the hook is
    /// issued, so each parent is one only decisions open and close.
    pub(crate) fn hook_issued(
        &mut self,
        now: f64,
        id: HookId,
        kind: HookKind,
        agent: &str,
        out: &mut Vec<SessionEvent>,
    ) {
        if !self.live() {
            return;
        }
        let label = || Attrs::new().set("lk.agent_label", agent);
        let hook = match kind {
            HookKind::OnEnter => {
                let parent =
                    self.starting.as_ref().or(self.updating.as_ref()).map(|o| o.id.clone());
                let attrs = label()
                    .set("gen_ai.operation.name", "create_agent")
                    .set("gen_ai.agent.name", agent);
                let activity =
                    self.start("start_agent_activity", parent.or(self.root()), now, attrs, out);
                let open = self.start("on_enter", Some(activity.id.clone()), now, label(), out);
                // `on_enter` runs as a task the activity does not wait for; the entry ends a
                // handoff.
                self.end(activity, now, out);
                if let Some(update) = self.updating.take() {
                    self.end(update, now, out);
                }
                Hook { open, activity: None }
            }
            HookKind::OnExit => {
                let parent = match self.closing.clone() {
                    Some(closing) => closing,
                    None => {
                        let update = self
                            .updating
                            .get_or_insert_with(|| Open { id: String::new(), start: now });
                        if update.id.is_empty() {
                            let attrs = Attrs::new().set("lk.previous_agent_label", agent);
                            let parent = self.root();
                            let open = self.start("update_agent", parent, now, attrs, out);
                            self.updating = Some(open);
                        }
                        self.updating.as_ref().map(|o| o.id.clone()).unwrap_or_default()
                    }
                };
                let activity = self.start("drain_agent_activity", Some(parent), now, label(), out);
                let open = self.start("on_exit", Some(activity.id.clone()), now, label(), out);
                Hook { open, activity: Some(activity) }
            }
            // In the session: the turn's own span ends as its commit reaches the host.
            HookKind::OnUserTurnCompleted => {
                let parent = self.root();
                let open = self.start("on_user_turn_completed", parent, now, label(), out);
                Hook { open, activity: None }
            }
            HookKind::OnUserTurnExceeded
            | HookKind::FillerSource
            | HookKind::FunctionToolsExecuted => return,
        };
        self.hooks.insert(id, hook);
    }

    pub(crate) fn hook_finished(
        &mut self,
        now: f64,
        id: HookId,
        failed: bool,
        out: &mut Vec<SessionEvent>,
    ) {
        let Some(hook) = self.hooks.remove(&id) else { return };
        let error = failed.then(|| "the callback raised or timed out".to_owned());
        self.end_with(hook.open, now, Attrs::new(), error, Default::default(), out);
        if let Some(activity) = hook.activity {
            self.end(activity, now, out);
        }
    }

    /// The AMD detector attached (`None`: `amd` opens) or released its verdict (it ends).
    pub(crate) fn amd(
        &mut self,
        now: f64,
        verdict: Option<&lk_agents_abi::voice::amd::AmdPrediction>,
        out: &mut Vec<SessionEvent>,
    ) {
        if !self.live() {
            return;
        }
        let Some(p) = verdict else {
            if self.amd.is_none() {
                let parent = self.root();
                self.amd = Some(self.start("amd", parent, now, Attrs::new(), out));
            }
            return;
        };
        let Some(open) = self.amd.take() else { return };
        let attrs = Attrs::new()
            .set("lk.amd.category", p.category.as_str())
            .set("lk.amd.reason", p.reason.as_str())
            .set("lk.amd.speech_duration", p.speech_duration.as_secs_f64())
            .set("lk.amd.delay", p.delay.as_secs_f64())
            .set("lk.pii.amd.transcript", p.transcript.as_str());
        self.end_with(open, now, attrs, None, Default::default(), out);
    }

    /// A keyterm detection pass was requested.
    pub(crate) fn keyterm_requested(&mut self, now: f64) {
        self.keyterm_pass = Some(now);
    }

    /// The pass answered with the effective list before and after it, or failed (`None`); its
    /// request, when it reported one, nests inside.
    pub(crate) fn keyterm_answered(
        &mut self,
        now: f64,
        lists: Option<(&[String], &[String])>,
        request: Option<&LlmMetrics>,
        out: &mut Vec<SessionEvent>,
    ) {
        let Some(start) = self.keyterm_pass.take() else { return };
        if !self.live() {
            return;
        }
        let attrs = match lists {
            Some((before, after)) => Attrs::new()
                .set("lk.keyterms.count", after.len())
                .set("lk.keyterms.added", after.iter().filter(|t| !before.contains(t)).count())
                .set("lk.keyterms.removed", before.iter().filter(|t| !after.contains(t)).count()),
            None => Attrs::new(),
        };
        let parent = self.root();
        let open = self.start("keyterm_detection", parent, start, attrs, out);
        if let Some(m) = request {
            let from = (m.timestamp - m.duration.max(0.0)).clamp(start, now);
            let chain = llm_request(m).into_iter().skip(1).collect();
            self.spans(Some(open.id.clone()), (from, m.timestamp.clamp(from, now)), chain, out);
        }
        let error = lists.is_none().then(|| "the keyterm detection pass failed".to_owned());
        self.end_with(open, now, Attrs::new(), error, Default::default(), out);
    }

    /// What `ev` opens or closes. `wall` dates a session instant; `now` is the step's wall time.
    pub(crate) fn observe(
        &mut self,
        now: f64,
        wall: &dyn Fn(Ts) -> f64,
        agent: &str,
        ev: &SessionEvent,
        out: &mut Vec<SessionEvent>,
    ) {
        if !self.live() {
            return;
        }
        match ev {
            SessionEvent::UserStateChanged { old, new, at } => {
                let at = wall(*at);
                if *old == UserState::Speaking {
                    if let Some(open) = self.user_speaking.take() {
                        self.end(open, at, out);
                    }
                    // The end of speech starts the wait for the turn's decision.
                    if let (Some(turn), None) = (&self.user_turn, &self.eou_wait) {
                        let parent = Some(turn.id.clone());
                        self.eou_wait = Some(self.start("eou_wait", parent, at, Attrs::new(), out));
                    }
                }
                if *new == UserState::Speaking {
                    self.end_eou_wait(at, "user_resumed", out);
                    self.open_user_turn(at, out);
                    let parent = self.user_turn.as_ref().map(|t| t.id.clone());
                    self.user_speaking =
                        Some(self.start("user_speaking", parent, at, Attrs::new(), out));
                }
            }
            SessionEvent::AgentStateChanged { old, new } => {
                if *old == AgentState::Speaking {
                    if let Some((open, _)) = self.agent_speaking.take() {
                        self.end(open, now, out);
                    }
                }
                if *new == AgentState::Speaking {
                    // The speech holding the floor, else the newest one still open.
                    let speech = self
                        .floor
                        .filter(|s| self.turns.contains_key(s))
                        .or_else(|| self.turns.keys().next_back().copied());
                    let parent = speech.and_then(|s| self.turn(s)).or(self.root());
                    let open = self.start("agent_speaking", parent, now, Attrs::new(), out);
                    self.agent_speaking = Some((open, speech));
                }
            }
            // A transcript without a VAD edge still opens the turn it belongs to.
            SessionEvent::UserInputTranscribed(_) => self.open_user_turn(now, out),
            // A prediction is made for a turn waiting on its decision (a late transcript can ask
            // for one before the speech edge).
            SessionEvent::EotPrediction(e) => {
                let bounds = (now - e.inference_duration.max(0.0), now);
                self.open_user_turn(bounds.0, out);
                if self.eou_wait.is_none() {
                    let parent = self.user_turn.as_ref().map(|t| t.id.clone());
                    self.eou_wait =
                        Some(self.start("eou_wait", parent, bounds.0, Attrs::new(), out));
                }
                let parent = self.eou_wait.as_ref().map(|o| o.id.clone());
                let attrs = Attrs::new()
                    .set("lk.eou.probability", e.probability)
                    .set("lk.eou.unlikely_threshold", e.threshold)
                    .set("lk.eou.endpointing_delay", e.delay.as_secs_f64());
                self.spans(parent, bounds, vec![("eou_detection", attrs)], out);
            }
            SessionEvent::UserTurnCompleted(e) => {
                self.end_eou_wait(now, "committed", out);
                if let Some(open) = self.user_speaking.take() {
                    self.end(open, now, out);
                }
                if let Some(open) = self.user_turn.take() {
                    let attrs = Attrs::new()
                        .set("lk.pii.user_transcript", e.new_transcript.as_str())
                        .set("lk.transcript_confidence", e.transcript_confidence);
                    self.end_with(open, now, attrs, None, Default::default(), out);
                }
            }
            SessionEvent::SpeechCreated { speech, .. } => {
                let Some(id) = speech.created() else { return };
                let attrs = Attrs::new()
                    .set("lk.speech_id", speech.speech_id())
                    .set("gen_ai.operation.name", "invoke_agent")
                    .set("gen_ai.agent.name", agent);
                let parent = self.root();
                let open = self.start("agent_turn", parent, now, attrs, out);
                let turn = AgentTurn {
                    open,
                    handle: Arc::clone(speech),
                    agent: agent.to_owned(),
                    done: None,
                };
                self.turns.insert(id, turn);
            }
            SessionEvent::SpeechDone { speech, outcome } => {
                self.end_turn(now, *speech, *outcome, out)
            }
            SessionEvent::FloorChanged { holder } => self.floor = *holder,
            SessionEvent::ToolExecutionUpdated(ToolLifecycle::Started { call }) => {
                let Some(speech) = self.floor.filter(|s| self.turns.contains_key(s)) else {
                    return;
                };
                let parent = self.turn(speech);
                let attrs = Attrs::new()
                    .set("lk.function_tool.id", call.call_id.as_str())
                    .set("lk.function_tool.name", call.name.as_str())
                    .set("lk.pii.function_tool.arguments", call.arguments.as_str())
                    .set("gen_ai.operation.name", "execute_tool")
                    .set("gen_ai.tool.name", call.name.as_str())
                    .set("gen_ai.tool.type", "function")
                    .set("gen_ai.tool.call.id", call.call_id.as_str())
                    .set("gen_ai.agent.name", agent);
                let open = self.start("function_tool", parent, now, attrs, out);
                let tool = Tool { open, name: call.name.clone(), speech };
                self.tools.insert(call.call_id.clone(), tool);
            }
            SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended {
                call_id,
                message,
                status,
                ..
            }) => {
                let Some(tool) = self.tools.remove(call_id) else { return };
                let is_error = *status == ToolEndStatus::Error;
                let error_type = is_error.then_some("tool_error");
                let attrs = Attrs::new()
                    .opt("lk.pii.function_tool.output", message.as_deref())
                    .set("lk.function_tool.is_error", is_error)
                    .opt("error.type", error_type);
                let metric = SpanMetric {
                    instrument: "gen_ai.execute_tool.duration".to_owned(),
                    value: (now - tool.open.start).max(0.0),
                    attributes: Attrs::new()
                        .set("gen_ai.operation.name", "execute_tool")
                        .set("gen_ai.tool.name", tool.name.as_str())
                        .opt("error.type", error_type)
                        .0,
                };
                self.end_with(tool.open, now, attrs, None, (Vec::new(), vec![metric]), out);
                let done = self.turns.get(&tool.speech).and_then(|t| t.done);
                if let Some(outcome) = done {
                    self.end_turn(now, tool.speech, outcome, out);
                }
            }
            SessionEvent::MetricsCollected { metrics, speech: Some(speech) } => {
                if let Some(parent) = self.turn(*speech) {
                    self.inference(metrics, parent, out);
                }
            }
            SessionEvent::Close { .. } => self.close(now, out),
            _ => {}
        }
    }

    fn open_user_turn(&mut self, at: f64, out: &mut Vec<SessionEvent>) {
        if self.user_turn.is_none() {
            let parent = self.root();
            self.user_turn = Some(self.start("user_turn", parent, at, Attrs::new(), out));
        }
    }

    fn end_eou_wait(&mut self, at: f64, outcome: &str, out: &mut Vec<SessionEvent>) {
        if let Some(open) = self.eou_wait.take() {
            let attrs = Attrs::new()
                .set("lk.eou.outcome", outcome)
                .set("lk.eou.wait_duration", (at - open.start).max(0.0));
            self.end_with(open, at, attrs, None, Default::default(), out);
        }
    }

    /// The speech is done: its `agent_speaking` ends, and its `agent_turn` with one `generation`
    /// event per step and `gen_ai.invoke_agent.duration`, or with the last tool it still runs.
    fn end_turn(
        &mut self,
        now: f64,
        speech: SpeechId,
        outcome: SpeechOutcome,
        out: &mut Vec<SessionEvent>,
    ) {
        if self.agent_speaking.as_ref().is_some_and(|(_, s)| *s == Some(speech)) {
            if let Some((open, _)) = self.agent_speaking.take() {
                self.end(open, now, out);
            }
        }
        if self.tools.values().any(|t| t.speech == speech) {
            if let Some(turn) = self.turns.get_mut(&speech) {
                turn.done = Some(outcome);
            }
            return;
        }
        let Some(turn) = self.turns.remove(&speech) else { return };
        let speech_id = turn.handle.speech_id().to_owned();
        let steps = turn.handle.num_steps();
        let generation = |n: u32| format!("{speech_id}_{n}");
        let events = (1..=steps)
            .map(|n| SpanEvent {
                name: "generation".to_owned(),
                time: turn.open.start,
                attributes: Attrs::new()
                    .set("lk.generation_id", generation(n))
                    .opt("lk.parent_generation_id", (n > 1).then(|| generation(n - 1)))
                    .0,
            })
            .collect();
        let attrs = Attrs::new()
            .opt("lk.generation_id", (steps > 0).then(|| generation(steps)))
            .set("lk.generation_count", steps)
            .set("lk.interrupted", outcome == SpeechOutcome::Interrupted);
        let error = (outcome == SpeechOutcome::Failed).then(|| match turn.handle.exception() {
            Ok(Some(e)) => e.to_string(),
            _ => "the speech failed".to_owned(),
        });
        let metric = SpanMetric {
            instrument: "gen_ai.invoke_agent.duration".to_owned(),
            value: (now - turn.open.start).max(0.0),
            attributes: Attrs::new()
                .set("gen_ai.operation.name", "invoke_agent")
                .set("gen_ai.agent.name", turn.agent.as_str())
                .0,
        };
        self.end_with(turn.open, now, attrs, error, (events, vec![metric]), out);
    }

    /// The node, request and attempt spans of one finished inference request, in `parent`.
    fn inference(&mut self, metrics: &AgentMetrics, parent: String, out: &mut Vec<SessionEvent>) {
        let (bounds, chain) = match metrics {
            AgentMetrics::Llm(m) => {
                ((m.timestamp - m.duration.max(0.0), m.timestamp), llm_request(m))
            }
            AgentMetrics::Tts(m) => {
                let node = Attrs::new()
                    .set("lk.tts.streaming", m.streamed)
                    .set("lk.tts.label", m.label.as_str())
                    .set("lk.response.ttfb", m.ttfb);
                let request = model_attrs(m.metadata.as_ref()).set("lk.response.ttfb", m.ttfb);
                let attempt = Attrs::new().set("lk.retry_count", 0);
                let chain = vec![
                    ("tts_node", node),
                    ("tts_request", request),
                    ("tts_request_run", attempt),
                ];
                ((m.timestamp - m.duration.max(0.0), m.timestamp), chain)
            }
            AgentMetrics::RealtimeModel(m) => {
                let inference = model_attrs(m.metadata.as_ref())
                    .set("gen_ai.operation.name", "generate_content")
                    .set("gen_ai.response.id", m.request_id.as_str())
                    .set("lk.response.ttft", m.ttft);
                let usage = Attrs::new()
                    .set("gen_ai.usage.input_tokens", m.input_tokens)
                    .set("gen_ai.usage.output_tokens", m.output_tokens)
                    .set("gen_ai.usage.input_cached_tokens", m.input_token_details.cached_tokens)
                    .set("gen_ai.usage.audio.input_tokens", m.input_token_details.audio_tokens)
                    .set("gen_ai.usage.text.input_tokens", m.input_token_details.text_tokens)
                    .set("gen_ai.usage.audio.output_tokens", m.output_token_details.audio_tokens)
                    .set("gen_ai.usage.text.output_tokens", m.output_token_details.text_tokens);
                let chain = vec![("realtime_inference", inference), ("realtime_metrics", usage)];
                ((m.timestamp - m.duration.max(0.0), m.timestamp), chain)
            }
            _ => return,
        };
        self.spans(Some(parent), bounds, chain, out);
    }

    /// `close`: whatever is still open ends, innermost first, then `session_close`, then the root.
    fn close(&mut self, now: f64, out: &mut Vec<SessionEvent>) {
        let last: Vec<String> = self.closing.iter().chain(&self.root).cloned().collect();
        let open: Vec<String> =
            self.order.iter().rev().filter(|id| !last.contains(id)).cloned().collect();
        for id in open.into_iter().chain(last) {
            self.end(Open { id, start: f64::MIN }, now, out);
        }
        *self = SessionTrace { ended: true, ..SessionTrace::default() };
    }
}

/// `llm_node > llm_request > llm_request_run` for one LLM request (`set_request_attributes`,
/// `set_usage_attributes`).
fn llm_request(m: &LlmMetrics) -> Vec<(&'static str, Attrs)> {
    let node = Attrs::new().set("lk.response.ttft", m.ttft);
    let request = model_attrs(m.metadata.as_ref())
        .set("gen_ai.operation.name", "chat")
        .set("gen_ai.request.stream", true)
        .set("gen_ai.response.id", m.request_id.as_str())
        .set("gen_ai.usage.input_tokens", m.prompt_tokens)
        .set("gen_ai.usage.output_tokens", m.completion_tokens)
        .set("gen_ai.usage.cache_read.input_tokens", m.prompt_cached_tokens)
        .set("gen_ai.usage.cache_write.input_tokens", m.cache_creation_tokens)
        .set("gen_ai.usage.reasoning.output_tokens", m.reasoning_tokens)
        .set("gen_ai.response.time_to_first_chunk", m.ttft);
    let attempt = model_attrs(m.metadata.as_ref()).set("lk.retry_count", 0);
    vec![("llm_node", node), ("llm_request", request), ("llm_request_run", attempt)]
}

/// The provider in the GenAI registry's spelling, and the model.
fn model_attrs(metadata: Option<&Metadata>) -> Attrs {
    let provider = metadata.and_then(|m| m.model_provider.as_deref()).and_then(provider_name);
    let model = metadata.and_then(|m| m.model_name.clone());
    Attrs::new().opt("gen_ai.provider.name", provider).opt("gen_ai.request.model", model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ends(out: &[SessionEvent]) -> Vec<String> {
        out.iter()
            .filter_map(|e| match e {
                SessionEvent::SpanEnded(s) => Some(s.span_id.clone()),
                _ => None,
            })
            .collect()
    }

    fn started(out: &[SessionEvent]) -> Vec<(String, Option<String>)> {
        out.iter()
            .filter_map(|e| match e {
                SessionEvent::SpanStarted(s) => Some((s.name.clone(), s.parent_id.clone())),
                _ => None,
            })
            .collect()
    }

    fn opened() -> SessionTrace {
        let mut t = SessionTrace { built: 1.0, ..SessionTrace::default() };
        let mut out = Vec::new();
        t.open(2.0, "agent", &mut out);
        t.opened(2.0, &mut out);
        t
    }

    // Catches: the root ending before what nests under it, or a span left open at the close.
    #[test]
    fn the_close_ends_everything_innermost_first_and_the_root_last() {
        let mut t = opened();
        let mut out = Vec::new();
        t.closing(3.0, "user_initiated", true, None, &mut out);
        t.hook_issued(3.0, HookId(1), HookKind::OnExit, "agent", &mut out);
        assert_eq!(
            started(&out),
            [
                ("session_close".to_owned(), Some(ROOT_SPAN.to_owned())),
                ("drain_agent_activity".to_owned(), Some("session_close_3".to_owned())),
                ("on_exit".to_owned(), Some("drain_agent_activity_4".to_owned())),
            ]
        );
        out.clear();
        let close = SessionEvent::Close {
            reason: lk_agents_abi::voice::events::CloseReason::UserInitiated,
            error: None,
        };
        t.observe(4.0, &|_| 0.0, "agent", &close, &mut out);
        assert_eq!(
            ends(&out),
            ["on_exit_5", "drain_agent_activity_4", "session_close_3", ROOT_SPAN]
        );
        out.clear();
        t.observe(5.0, &|_| 0.0, "agent", &close, &mut out);
        assert!(out.is_empty(), "nothing after the close: {out:?}");
    }

    // Catches: a handoff's exit and entry not grouped under one `update_agent`, or the opening
    // agent's entry outside `session_start`.
    #[test]
    fn a_handoff_is_one_update_agent_from_the_exit_to_the_entry() {
        let mut t = SessionTrace::default();
        let mut out = Vec::new();
        t.open(1.0, "a", &mut out);
        t.hook_issued(1.0, HookId(1), HookKind::OnEnter, "a", &mut out);
        t.opened(1.0, &mut out);
        assert_eq!(
            started(&out)[2],
            ("start_agent_activity".to_owned(), Some("session_start_2".to_owned()))
        );
        out.clear();
        t.hook_issued(2.0, HookId(2), HookKind::OnExit, "a", &mut out);
        t.hook_finished(2.5, HookId(2), false, &mut out);
        t.hook_issued(3.0, HookId(3), HookKind::OnEnter, "b", &mut out);
        let names: Vec<_> =
            started(&out).into_iter().map(|(n, p)| (n, p.unwrap_or_default())).collect();
        assert_eq!(names[0], ("update_agent".to_owned(), ROOT_SPAN.to_owned()));
        assert_eq!(names[1].1, "update_agent_5");
        assert_eq!(names[3], ("start_agent_activity".to_owned(), "update_agent_5".to_owned()));
        assert!(ends(&out).contains(&"update_agent_5".to_owned()), "the entry ends the handoff");
    }

    // Catches: a hook kind with no span in the reference getting one, or a failed one ending ok.
    #[test]
    fn only_the_agent_callbacks_are_spans_and_a_failed_one_is_an_error() {
        let mut t = opened();
        let mut out = Vec::new();
        t.hook_issued(2.0, HookId(1), HookKind::FillerSource, "agent", &mut out);
        t.hook_finished(2.0, HookId(1), false, &mut out);
        assert!(out.is_empty());
        t.hook_issued(2.0, HookId(2), HookKind::OnUserTurnCompleted, "agent", &mut out);
        t.hook_finished(3.0, HookId(2), true, &mut out);
        let SessionEvent::SpanEnded(end) = &out[1] else { panic!("{out:?}") };
        assert!(end.error.is_some());
    }

    // Catches: the wait for the turn's decision not opening at the end of speech, or ending
    // without an outcome.
    #[test]
    fn the_end_of_speech_waits_for_the_decision_under_the_turn() {
        let mut t = opened();
        let mut out = Vec::new();
        let edge = |old, new| SessionEvent::UserStateChanged { old, new, at: Ts::ZERO };
        t.observe(3.0, &|_| 3.0, "a", &edge(UserState::Listening, UserState::Speaking), &mut out);
        t.observe(4.0, &|_| 4.0, "a", &edge(UserState::Speaking, UserState::Listening), &mut out);
        let names: Vec<_> = started(&out).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, ["user_turn", "user_speaking", "eou_wait"]);
        out.clear();
        t.observe(4.0, &|_| 5.0, "a", &edge(UserState::Listening, UserState::Speaking), &mut out);
        let SessionEvent::SpanEnded(wait) = &out[0] else { panic!("{out:?}") };
        assert_eq!(wait.attributes["lk.eou.outcome"], "user_resumed");
        assert_eq!(wait.attributes["lk.eou.wait_duration"], 1.0);
    }
}
