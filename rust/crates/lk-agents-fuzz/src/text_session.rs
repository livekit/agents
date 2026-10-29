//! A backend conversation (`lk_agents_core::text_session::TextSession`, text-session.md) over
//! several requests: each turn reconstructs the session from the state the last one returned, as
//! an endpoint does from its saved routing state, and the person's next message starts it.
//!
//! The front desk's model answers in text, hands the booking to a child agent through its agent
//! tool (the child answers with a structured reference or asks a question), calls it twice in one
//! reply, with arguments the host refuses, or fails. Under approval it repeats the call the
//! person agreed to, changes it, or drops it. A recursive desk delegates to itself until the
//! depth bound refuses the ninth frame. The host binds arguments, renders read-backs, validates
//! outputs and settles cancelled work, each after its own delay; the endpoint delivers each
//! settled invocation's dialogue before releasing it; the client cancels some turns midway.
//!
//! The rules, over every turn's effects and returned state:
//! * `text_session_finishes_once`: each turn ends once, with no new invocation after its end,
//!   and never successfully after a cancel;
//! * `text_session_invocations_end_once`: every invocation started settles or parks once;
//! * `text_session_depth_is_bounded`: no chain of frames exceeds eight, and the depth refusal
//!   comes only at that depth;
//! * `text_session_parent_call_resolved_once`: a child's parent call gets one output in the
//!   parent's history once the child completed, never two, across reconstructions;
//! * `text_session_approval_precedes_child`: under approval a child starts only for the
//!   arguments the person was asked about in an earlier turn;
//! * `text_session_history_mode`: a fresh child starts without its parent's dialogue; a forked
//!   one never with a call that has no output.

use lk_agents_voice::tts::markup::ChatMessageExt;
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, UNIX_EPOCH};

use fastrand::Rng;
use lk_agents_abi::agent::run::RunOutcome;
use lk_agents_abi::agent::{
    AgentDefinition, ArgumentIdentity, FunctionToolInfo, OutputCandidate, OutputSpec, ReadBack,
    ReadBackAnswer, Retries,
};
use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::resources::{LlmRef, ResourceId};
use lk_agents_abi::{
    ChatItem, ChatMessage, ChoiceDelta, FunctionToolCall, LlmChunk, ProviderId, Role, ToolSpec, Ts,
    WallAnchor,
};
use lk_agents_core::agent_run::{self, OutputVerdict};
use lk_agents_core::text_session::{
    ChatHistory, ChildCallStatus, Effect, Event, LocalAgentTool, RoutingError, SessionAgent,
    TextSession, TextSessionConfig, TextSessionInput, TextSessionState,
};
use serde_json::{Value, json};

use crate::sched::Timeline;
use crate::sched::{Sampling, Sched, Step as Picked, Tick};

/// The scheduler's own random stream off the seed: the bytes `sched`.
const SCHED_STREAM: u64 = u64::from_be_bytes(*b"\0\0\0sched");
/// The deepest chain of frames (`text-session.md`: eight by default).
pub const MAX_DEPTH: usize = 8;
const ROOT: &str = "desk";
const CHILD: &str = "booking";
const TOOL: &str = "book";

/// Who reports: the model, the host's tool and validation work, the endpoint's delivery, or
/// the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Src {
    Model,
    Host,
    Delivery,
    Client,
}

impl crate::sched::Source for Src {}

/// The configuration a seed draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Setup {
    pub approval: bool,
    pub history: ChatHistory,
    pub merge: bool,
    /// The desk's agent tool hands to the desk itself.
    pub recursive: bool,
}

/// One turn: the state it started from, every effect, and the state it returned.
#[derive(Debug, Clone)]
pub struct Turn {
    pub before: Option<TextSessionState>,
    pub effects: Vec<Effect>,
    pub cancelled: bool,
    pub after: Option<TextSessionState>,
}

#[derive(Debug, Clone)]
pub struct Trace {
    pub setup: Setup,
    pub turns: Vec<Turn>,
}

/// A finished run: what broke, and what it reached.
#[derive(Debug)]
pub struct Outcome {
    pub seed: u64,
    pub violations: Vec<Finding>,
    pub reached: BTreeSet<&'static str>,
    pub trace: Trace,
}

/// A rule: its name, the kind of subject it concerns (`known_findings.tsv` keys open findings
/// by both), and its check.
type Rule = (&'static str, &'static str, fn(&Trace) -> Vec<String>);

pub const RULES: &[Rule] = &[
    ("text_session_finishes_once", "Turn", finishes_once),
    ("text_session_invocations_end_once", "Invocation", invocations_end_once),
    ("text_session_depth_is_bounded", "Frame", depth_bounded),
    ("text_session_parent_call_resolved_once", "ParentCall", parent_resolved_once),
    ("text_session_approval_precedes_child", "Child", approval_precedes_child),
    ("text_session_history_mode", "Child", history_mode),
];

/// One broken rule: its name, subject kind and what happened.
#[derive(Debug, Clone)]
pub struct Finding {
    pub contract: &'static str,
    pub subject: &'static str,
    pub text: String,
}

impl Finding {
    /// Whether this is one of the open findings (`known_findings.tsv`).
    pub fn known(&self) -> bool {
        crate::check::KNOWN_FINDINGS
            .iter()
            .any(|k| k.contract == self.contract && k.subject == self.subject)
    }
}

/// Every rule over `trace`.
pub fn check(trace: &Trace) -> Vec<Finding> {
    RULES
        .iter()
        .flat_map(|(contract, subject, rule)| {
            rule(trace).into_iter().map(move |text| Finding { contract, subject, text })
        })
        .collect()
}

fn config(setup: &Setup) -> TextSessionConfig {
    let tool = FunctionToolInfo {
        spec: ToolSpec {
            provider_tool: None,
            name: TOOL.into(),
            description: "Book a room.".into(),
            json_schema: json!({"type":"object", "properties":{"nights":{"type":"integer"}}}),
            flags: Default::default(),
        },
        raw_schema: false,
        needs_approval: setup.approval.then_some(ReadBack::Default),
        on_duplicate: DuplicateMode::Allow,
        duplicate_scope: DuplicateScope::Name,
    };
    let structured = OutputSpec::Structured {
        schema: json!({"type":"object", "properties":{"reference":{"type":"string"}}}),
    };
    let agent = |name: &str, tools, output| SessionAgent {
        definition: AgentDefinition {
            name: name.into(),
            instructions: format!("You are {name}."),
            dynamic_instructions: false,
            tools,
            output,
            retries: Retries::default(),
        },
        llm: Some(LlmRef { id: ResourceId(1), provider: ProviderId::of_label("fake") }),
        output_type_explicit: true,
        agent_tools: BTreeMap::new(),
    };
    let output = match setup.recursive {
        // a desk that is also its own child answers with a reference
        true => structured.clone(),
        false => OutputSpec::Text { validated: false },
    };
    let mut root = agent(ROOT, vec![tool], output);
    let target = if setup.recursive { ROOT } else { CHILD };
    root.agent_tools.insert(
        TOOL.into(),
        LocalAgentTool {
            agent_name: target.into(),
            chat_history: setup.history,
            merge_chat_history: setup.merge,
            context_scope: Some("draft".into()),
        },
    );
    let child = agent(CHILD, vec![], structured);
    TextSessionConfig::new(
        ROOT.into(),
        BTreeMap::from([(ROOT.into(), root), (CHILD.into(), child)]),
    )
}

fn input(state: Option<TextSessionState>, turn: usize) -> TextSessionInput {
    TextSessionInput {
        session_id: "conversation".into(),
        execution_id: format!("task-{turn}"),
        task_id: None,
        root_scope: "customer".into(),
        available_scopes: BTreeSet::from(["customer".into(), "draft".into()]),
        separate_contexts: false,
        state,
        wall: WallAnchor { wall: 1_700_000_000.0, at: Ts::ZERO },
        cleanup_timeout: Duration::from_secs(1),
        delegated: true,
    }
}

fn ts(ms: u64) -> Ts {
    Ts::from_nanos(ms * 1_000_000)
}

/// The fakes' side of a turn.
struct World {
    rng: Rng,
    /// Each invocation's agent, and whether it runs a child frame.
    agents: BTreeMap<String, (String, bool)>,
    /// Arguments the person was asked to approve, by the turn's starting state.
    asked: Vec<Value>,
    next_prepared: u64,
}

impl World {
    /// The model's reply to one request of `agent` at `depth`.
    fn reply(
        &mut self,
        agent: &str,
        child: bool,
        last_user: Option<String>,
    ) -> (String, Vec<(String, Value)>) {
        let rng = &mut self.rng;
        let call = |rng: &mut Rng| (TOOL.to_owned(), json!({"nights": rng.u32(1..14)}));
        // the booking agent answers; a recursive desk's child answers one time in three and
        // delegates again otherwise, down to the bound
        if agent == CHILD || (child && rng.u8(0..3) == 0) {
            return match rng.u8(0..10) {
                0..6 => (
                    String::new(),
                    vec![(
                        "final_result".into(),
                        json!({"reference": format!("R-{}", rng.u32(..))}),
                    )],
                ),
                _ => ("Which date?".into(), vec![]),
            };
        }
        // the person answered a read-back: repeat that call, change it, or drop it
        if let Some(asked) = self.asked.first().cloned()
            && last_user.is_some()
        {
            return match rng.u8(0..10) {
                0..6 => (String::new(), vec![(TOOL.into(), asked)]),
                6 | 7 => (String::new(), vec![call(rng)]),
                _ => ("I will not book it.".into(), vec![]),
            };
        }
        match rng.u8(0..20) {
            0..8 => ("How can I help?".into(), vec![]),
            8..16 => (String::new(), vec![call(rng)]),
            16 => (String::new(), vec![call(rng), call(rng)]),
            _ => (String::new(), vec![(TOOL.into(), json!({"nights": "two"}))]),
        }
    }

    /// The fakes answer what the session asked for.
    fn react(&mut self, effects: &[Effect], now: u64, timeline: &mut Timeline<Src, Event>) {
        for effect in effects {
            match effect {
                Effect::InvocationStarted { invocation_id, agent_name, frame_id, .. } => {
                    let child = !frame_id.value.ends_with("/frame_0");
                    self.agents.insert(invocation_id.clone(), (agent_name.clone(), child));
                }
                Effect::Run { invocation_id, effect } => {
                    self.run(invocation_id, effect, now, timeline)
                }
                Effect::InvocationSettled { invocation_id, result, .. } => {
                    let mut at = now;
                    for item in &result.new_messages {
                        if let ChatItem::Message(m) = item
                            && m.role == Role::Assistant
                        {
                            at += self.rng.u64(0..300);
                            let event = Event::DialogueDelivered {
                                invocation_id: invocation_id.clone(),
                                message: m.clone(),
                            };
                            timeline.at(at, Src::Delivery, event);
                        }
                    }
                    let release =
                        Event::InvocationReleased {
                        invocation_id: invocation_id.clone(),
                        delivered: Default::default(),
                    };
                    timeline.at(at + self.rng.u64(0..50), Src::Delivery, release);
                }
                Effect::InvocationParked { invocation_id, .. } => {
                    let release =
                        Event::InvocationReleased {
                        invocation_id: invocation_id.clone(),
                        delivered: Default::default(),
                    };
                    timeline.at(now + self.rng.u64(0..50), Src::Delivery, release);
                }
                _ => {}
            }
        }
    }

    fn run(
        &mut self,
        invocation: &str,
        effect: &agent_run::Effect,
        now: u64,
        timeline: &mut Timeline<Src, Event>,
    ) {
        use agent_run::Effect as E;
        use agent_run::Event as V;
        let send = |timeline: &mut Timeline<Src, Event>, src, at, event| {
            timeline.at(at, src, Event::Run { invocation_id: invocation.to_owned(), event });
        };
        match effect {
            E::LlmRequestStart { request, chat_ctx, .. } => {
                let (agent, child) =
                    self.agents.get(invocation).cloned().unwrap_or((ROOT.into(), false));
                let last_user = chat_ctx.items.iter().rev().find_map(|i| {
                    i.as_message().filter(|m| m.role == Role::User).and_then(|m| m.text_content())
                });
                let first = now + self.rng.u64(200..1_500);
                if self.rng.u8(0..30) == 0 {
                    send(
                        timeline,
                        Src::Model,
                        first,
                        V::LlmFailed { request: *request, error: "overloaded".into() },
                    );
                    return;
                }
                let (text, calls) = self.reply(&agent, child, last_user);
                let chunk = LlmChunk {
                    id: "chunk".into(),
                    delta: Some(ChoiceDelta {
                        role: Some(Role::Assistant),
                        content: (!text.is_empty()).then_some(text),
                        tool_calls: calls
                            .iter()
                            .enumerate()
                            .map(|(n, (name, args))| FunctionToolCall {
                                name: name.clone(),
                                arguments: args.to_string(),
                                call_id: format!("{invocation}-{request:?}-{n}"),
                                extra: None,
                            })
                            .collect(),
                        extra: None,
                    }),
                    usage: None,
                    flush: false,
                    node_text: false,
                };
                send(timeline, Src::Model, first, V::LlmChunk { request: *request, chunk });
                send(
                    timeline,
                    Src::Model,
                    first + self.rng.u64(1..100),
                    V::LlmEnded { request: *request },
                );
            }
            // a cancelled stream ends as the provider closes it
            E::LlmRequestCancel { request } => {
                send(
                    timeline,
                    Src::Model,
                    now + self.rng.u64(1..200),
                    V::LlmFailed { request: *request, error: "cancelled".into() },
                );
            }
            E::ToolPrepare { call, arguments, .. } => {
                let at = now + self.rng.u64(1..100);
                let event = match arguments.get("nights").and_then(Value::as_u64) {
                    Some(n) => {
                        self.next_prepared += 1;
                        V::ToolPrepared {
                            call: *call,
                            prepared: ResourceId(1_000 + self.next_prepared),
                            identity: ArgumentIdentity::Json(json!({"nights": n})),
                        }
                    }
                    None => V::ToolPrepareFailed {
                        call: *call,
                        failure: lk_agents_core::tool_calls::PrepareFailure::Refused(
                            "nights must be an integer".into(),
                        ),
                    },
                };
                send(timeline, Src::Host, at, event);
            }
            E::ReadBackRender { call, .. } => {
                let answer = ReadBackAnswer::Question("Book it?".into());
                send(
                    timeline,
                    Src::Host,
                    now + self.rng.u64(1..50),
                    V::ReadBackRendered { call: *call, result: Ok(answer) },
                );
            }
            E::OutputValidate { validation, candidate } => {
                let output = match candidate {
                    OutputCandidate::Value(v) => v.clone(),
                    OutputCandidate::Text(t) => Value::String(t.clone()),
                };
                let event = Event::OutputValidated {
                    invocation_id: invocation.to_owned(),
                    validation: *validation,
                    verdict: OutputVerdict::Accepted { text: None },
                    output: Some(output),
                };
                timeline.at(now + self.rng.u64(1..50), Src::Host, event);
            }
            E::HostCheckpoint { checkpoint, .. } => {
                send(
                    timeline,
                    Src::Host,
                    now + self.rng.u64(0..20),
                    V::HostCheckpointDone {
                        checkpoint: *checkpoint,
                        result: Ok(true),
                        tools: None,
                    },
                );
            }
            _ => {}
        }
    }
}

/// One seeded conversation, the pending reports picked in the `sampling` order.
pub fn run(seed: u64, sampling: Sampling) -> Outcome {
    let mut rng = Rng::with_seed(seed);
    let setup = Setup {
        approval: rng.u8(0..3) == 0,
        history: if rng.bool() { ChatHistory::Fresh } else { ChatHistory::Fork },
        merge: rng.bool(),
        recursive: rng.u8(0..5) == 0,
    };
    let cfg = config(&setup);
    let mut world = World {
        rng: Rng::with_seed(rng.u64(..)),
        agents: BTreeMap::new(),
        asked: Vec::new(),
        next_prepared: 0,
    };
    let mut trace = Trace { setup, turns: Vec::new() };
    let mut state: Option<TextSessionState> = None;
    for turn in 0..rng.usize(1..7) {
        let mut sched: Sched<Src, Event> =
            Sched::new(sampling, Rng::with_seed(seed ^ SCHED_STREAM ^ turn as u64), &[]);
        let mut timeline: Timeline<Src, Event> = Timeline::default();
        world.agents.clear();
        world.asked = state
            .iter()
            .flat_map(|s| s.frames.values())
            .flat_map(|f| &f.pending_calls)
            .filter(|c| c.status == ChildCallStatus::Approval)
            .map(|c| c.validated_arguments.clone())
            .collect();
        let mut session = match TextSession::new(cfg.clone(), input(state.clone(), turn)) {
            Ok(session) => session,
            Err(e) => {
                panic!("seed {seed} turn {turn}: the returned state does not reconstruct: {e:?}")
            }
        };
        let text = match rng.u8(0..4) {
            0 => "yes",
            1 => "Friday",
            _ => "I would like a room",
        };
        let start = ChatMessage::text(format!("human-{turn}"), Role::User, text, UNIX_EPOCH);
        let cancel_at = (rng.u8(0..8) == 0).then(|| rng.u64(0..4_000));
        if let Some(at) = cancel_at {
            timeline.at(at, Src::Client, Event::Cancel);
        }
        let mut record =
            Turn { before: state.clone(), effects: Vec::new(), cancelled: false, after: None };
        let effects = session.handle(Event::Start(start), ts(0));
        world.react(&effects, 0, &mut timeline);
        record.effects.extend(effects);
        let mut now = 0;
        for _ in 0..20_000 {
            timeline.release(now, &mut sched);
            let event = if session.wake_at().is_some_and(|at| at <= ts(now)) {
                Event::Wake
            } else {
                let wake = session.wake_at().map(|t| t.as_nanos().div_ceil(1_000_000));
                match sched.tick(now, &[], timeline.next_due().into_iter().chain(wake)) {
                    Tick::Step(Picked::Item { item, .. }) => item,
                    Tick::Step(Picked::Process(_)) => unreachable!("no processes"),
                    Tick::Advance(at) => {
                        now = at;
                        continue;
                    }
                    Tick::Quiet => break,
                }
            };
            record.cancelled |= matches!(event, Event::Cancel) && !finished(&record.effects);
            let effects = session.handle(event, ts(now));
            world.react(&effects, now, &mut timeline);
            record.effects.extend(effects);
        }
        record.after = record.effects.iter().find_map(|e| match e {
            Effect::Finished(Ok(turn)) => Some(turn.state.clone()),
            _ => None,
        });
        // a failed or cancelled turn saves nothing: the next request starts from the last state
        if record.after.is_some() {
            state = record.after.clone();
        }
        trace.turns.push(record);
    }
    let violations = check(&trace);
    Outcome { seed, violations, reached: reached(&trace), trace }
}

fn finished(effects: &[Effect]) -> bool {
    effects.iter().any(|e| matches!(e, Effect::Finished(_)))
}

/// What a trace reached, for the campaign's reach report.
pub fn reached(trace: &Trace) -> BTreeSet<&'static str> {
    let mut reached = BTreeSet::new();
    for turn in &trace.turns {
        if turn.cancelled {
            reached.insert("TextSessionCancelled");
        }
        if turn.before.is_some() {
            reached.insert("TextSessionReconstructed");
        }
        for e in &turn.effects {
            reached.insert(match e {
                Effect::InvocationStarted { agent_name, .. } if agent_name == CHILD => {
                    "TextSessionChildStarted"
                }
                Effect::InvocationParked { .. } => "TextSessionParentParked",
                Effect::RoutingRefused { error: RoutingError::MaxDepth { .. }, .. } => {
                    "TextSessionDepthRefused"
                }
                Effect::RoutingRefused { .. } => "TextSessionRoutingRefused",
                Effect::Finished(Ok(t))
                    if matches!(t.result.outcome, RunOutcome::InputRequired { .. }) =>
                {
                    "TextSessionInputRequired"
                }
                Effect::Finished(Ok(_)) => "TextSessionCompleted",
                Effect::Finished(Err(_)) => "TextSessionFailed",
                _ => continue,
            });
        }
        if let Some(after) = &turn.after {
            if after
                .frames
                .values()
                .any(|f| f.pending_calls.iter().any(|c| c.status == ChildCallStatus::Approval))
            {
                reached.insert("TextSessionApprovalAsked");
            }
            if after.frames.values().any(|f| f.completed) {
                reached.insert("TextSessionChildCompleted");
            }
        }
    }
    reached
}

fn finishes_once(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    for (n, turn) in trace.turns.iter().enumerate() {
        let ends: Vec<usize> = turn
            .effects
            .iter()
            .enumerate()
            .filter(|(_, e)| matches!(e, Effect::Finished(_)))
            .map(|(i, _)| i)
            .collect();
        if ends.len() != 1 {
            out.push(format!("turn {n} ended {} times", ends.len()));
            continue;
        }
        if turn.effects[ends[0]..].iter().any(|e| matches!(e, Effect::InvocationStarted { .. })) {
            out.push(format!("turn {n} started an invocation after its end"));
        }
        if turn.cancelled && turn.after.is_some() {
            out.push(format!("turn {n} was cancelled and returned a successful envelope"));
        }
    }
    out
}

fn invocations_end_once(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    for (n, turn) in trace.turns.iter().enumerate() {
        let mut ends: BTreeMap<&str, usize> = BTreeMap::new();
        for e in &turn.effects {
            match e {
                Effect::InvocationStarted { invocation_id, .. } => {
                    ends.entry(invocation_id).or_default();
                }
                Effect::InvocationSettled { invocation_id, .. }
                | Effect::InvocationParked { invocation_id, .. } => {
                    *ends.entry(invocation_id).or_default() += 1;
                }
                _ => {}
            }
        }
        let failed = turn.effects.iter().any(|e| matches!(e, Effect::Finished(Err(_))));
        for (id, count) in ends {
            if count > 1 || (count == 0 && !failed) {
                out.push(format!("turn {n}: invocation {id} ended {count} times"));
            }
        }
    }
    out
}

/// The length of the chain from `frame` to the root.
fn depth(state: &TextSessionState, frame: &lk_agents_core::text_session::FrameId) -> usize {
    let mut depth = 1;
    let mut at = frame.clone();
    while let Some(parent) = state.frames.get(&at).and_then(|f| f.parent.as_ref()) {
        depth += 1;
        at = parent.frame_id.clone();
        if depth > state.frames.len() {
            break;
        }
    }
    depth
}

fn depth_bounded(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    for (n, turn) in trace.turns.iter().enumerate() {
        if let Some(after) = &turn.after {
            let deepest = after.frames.keys().map(|f| depth(after, f)).max().unwrap_or(0);
            if deepest > MAX_DEPTH {
                out.push(format!("turn {n} returned a chain of {deepest} frames"));
            }
        }
        // the refusal comes only at the bound: one more child was asked for at depth eight
        let started =
            turn.effects.iter().filter(|e| matches!(e, Effect::InvocationStarted { .. })).count();
        let before = turn.before.as_ref().map_or(1, |s| depth(s, &s.active_frame_id));
        let refused = turn.effects.iter().any(|e| {
            matches!(e, Effect::RoutingRefused { error: RoutingError::MaxDepth { .. }, .. })
        });
        if refused && before + started.saturating_sub(1) < MAX_DEPTH {
            out.push(format!(
                "turn {n} refused for depth with {} frames reached",
                before + started.saturating_sub(1)
            ));
        }
    }
    out
}

fn parent_resolved_once(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    for (n, turn) in trace.turns.iter().enumerate() {
        let Some(after) = &turn.after else { continue };
        for frame in after.frames.values() {
            let Some(call) = &frame.parent else { continue };
            let Some(parent) = after.frames.get(&call.frame_id) else {
                out.push(format!("turn {n}: {} has no parent frame", frame.id.value));
                continue;
            };
            let outputs = parent
                .history
                .iter()
                .filter(
                    |i| matches!(i, ChatItem::FunctionCallOutput(o) if o.call_id == call.call_id),
                )
                .count();
            if outputs > 1 || (frame.completed && outputs != 1) {
                out.push(format!(
                    "turn {n}: the call {} of {} has {outputs} outputs (child completed: {})",
                    call.call_id, call.frame_id.value, frame.completed
                ));
            }
        }
    }
    out
}

/// The arguments a child started with: the last user message of its starting history.
fn child_arguments(history: &[ChatItem]) -> Option<Value> {
    history.iter().rev().find_map(|i| {
        let m = i.as_message().filter(|m| m.role == Role::User)?;
        serde_json::from_str(&m.text_content()?).ok()
    })
}

fn approval_precedes_child(trace: &Trace) -> Vec<String> {
    if !trace.setup.approval {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (n, turn) in trace.turns.iter().enumerate() {
        let asked: Vec<&Value> = turn
            .before
            .iter()
            .flat_map(|s| s.frames.values())
            .flat_map(|f| &f.pending_calls)
            .filter(|c| c.status == ChildCallStatus::Approval)
            .map(|c| &c.validated_arguments)
            .collect();
        let parents: BTreeSet<String> =
            turn.before.iter().flat_map(|s| s.frames.keys()).map(|f| f.value.clone()).collect();
        for e in &turn.effects {
            let Effect::InvocationStarted { frame_id, history, .. } = e else { continue };
            // a new child frame: not one the turn resumed
            if parents.contains(&frame_id.value) || frame_id.value.ends_with("/frame_0") {
                continue;
            }
            let args = child_arguments(history);
            if !args.as_ref().is_some_and(|a| asked.contains(&a)) {
                out.push(format!(
                    "turn {n}: a child started with {args:?}, the person was asked about {asked:?}"
                ));
            }
        }
    }
    out
}

fn history_mode(trace: &Trace) -> Vec<String> {
    let mut out = Vec::new();
    for (n, turn) in trace.turns.iter().enumerate() {
        let mut seen: BTreeSet<String> = turn
            .before
            .iter()
            .flat_map(|s| s.frames.values())
            .flat_map(|f| f.history.iter().map(|i| i.id().to_owned()))
            .collect();
        // the person's message of this turn is the parent's
        seen.insert(format!("human-{n}"));
        let resumed: BTreeSet<String> =
            turn.before.iter().flat_map(|s| s.frames.keys()).map(|f| f.value.clone()).collect();
        for e in &turn.effects {
            match e {
                Effect::InvocationStarted { frame_id, history, .. }
                    if !resumed.contains(&frame_id.value)
                        && !frame_id.value.ends_with("/frame_0") =>
                {
                    match trace.setup.history {
                        ChatHistory::Fresh => {
                            let inherited: Vec<&str> = history
                                .iter()
                                .filter_map(|i| i.as_message())
                                .filter(|m| seen.contains(&m.id) && m.role == Role::User)
                                .map(|m| m.id.as_str())
                                .collect();
                            if !inherited.is_empty() {
                                out.push(format!("turn {n}: fresh child {} starts with the parent's {inherited:?}", frame_id.value));
                            }
                        }
                        ChatHistory::Fork => {
                            let outputs: BTreeSet<&str> = history
                                .iter()
                                .filter_map(|i| match i {
                                    ChatItem::FunctionCallOutput(o) => Some(o.call_id.as_str()),
                                    _ => None,
                                })
                                .collect();
                            for i in history {
                                if let ChatItem::FunctionCall(c) = i
                                    && !outputs.contains(c.call_id.as_str())
                                {
                                    out.push(format!("turn {n}: forked child {} starts with the unresolved call {}", frame_id.value, c.call_id));
                                }
                            }
                        }
                    }
                }
                Effect::Emit { .. } | Effect::Run { .. } => {}
                Effect::InvocationSettled { result, .. } => {
                    seen.extend(result.all_messages.iter().map(|i| i.id().to_owned()))
                }
                _ => {}
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
