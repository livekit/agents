//! The session's spans (`telemetry.md`, "`ManualSpan`"; `telemetry-reference.md`, "Span trees").
//! Each span opens where its entity starts and ends where it is retired: the session and its
//! lifecycle, the agent activities and hooks, the user turns ([`user_turn`]), the agent turns
//! ([`agent_turn`]), AMD and keyterm detection. Ended spans leave each step as records on
//! [`SessionEvent::Telemetry`]; work that continues in the runtime gets its parent's `SpanRef`.

pub(crate) mod agent_turn;
pub(crate) mod user_turn;

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{HookId, HookKind};
use crate::audio_recognition::TurnId;
use crate::events::{ErrorEvent, SessionEvent, ToolLifecycle};
use lk_agents_abi::Ts;
use lk_agents_abi::telemetry::decl::{Content, Seconds, SpanDecl, SpanRef};
use lk_agents_abi::telemetry::record::{JobKey, Record};
use lk_agents_abi::telemetry::spans::common::{GenAiOperation, ParticipantAttrs};
use lk_agents_abi::telemetry::spans::session::{
    AgentSession, AgentSessionEnd, AgentStateChanged, Amd, AmdVerdict, CloseReason,
    DrainAgentActivity, KeytermDetection, KeytermDetectionEnd, OnEnter, OnExit,
    ParticipantDisconnected, ParticipantLinked, PauseAgentActivity, ResumeAgentActivity,
    SessionClose, SessionStart, StartAgentActivity, UpdateAgent, UserStateChanged,
};
use lk_agents_abi::telemetry::spans::user_turn::UserTurnOutcome;
use lk_agents_abi::voice::amd::AmdPrediction;
use lk_agents_core::telemetry::span::{IdSource, ManualSpan, SpanFailure, Trace, seeded_ids};
use std::collections::BTreeMap;

pub(crate) use user_turn::{Endpointing, Stt};

/// What the runtime supplies so the session's spans join the job's trace.
#[derive(Default)]
pub struct SessionTelemetry {
    /// The span the session continues: the job's `job_entrypoint`, or the caller's current span.
    pub parent: Option<SpanRef>,
    pub job: Option<JobKey>,
    /// Span and trace ids; `None` seeds them from the session id (tests, the DST).
    pub ids: Option<IdSource>,
}

/// A hook that timed out.
struct TimedOut;

impl SpanFailure for TimedOut {
    fn error_type(&self) -> String {
        "timeout".to_owned()
    }
    fn message(&self) -> String {
        "the callback did not return within its bound".to_owned()
    }
}

/// A hook that failed with no exception (its agent was gone, its call undeliverable).
struct Unanswered;

impl SpanFailure for Unanswered {
    fn error_type(&self) -> String {
        "unanswered".to_owned()
    }
    fn message(&self) -> String {
        "the callback could not run".to_owned()
    }
}

/// How a hook failed, as its span records it.
pub(crate) enum HookFailure<'a> {
    Raised(&'a lk_agents_abi::errors::HostError),
    TimedOut,
    Failed,
}

impl HookFailure<'_> {
    fn failure(&self) -> &dyn SpanFailure {
        match self {
            HookFailure::Raised(error) => *error,
            HookFailure::TimedOut => &TimedOut,
            HookFailure::Failed => &Unanswered,
        }
    }
}

/// The span an agent's activity runs in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Under {
    Start,
    Update,
    Close,
}

enum Hook {
    Enter { under: Under, activity: ManualSpan<StartAgentActivity>, hook: ManualSpan<OnEnter> },
    Exit { under: Under, drain: ManualSpan<DrainAgentActivity>, hook: ManualSpan<OnExit> },
    Turn(TurnId),
}

impl Hook {
    fn under(&self) -> Option<Under> {
        match self {
            Hook::Enter { under, .. } | Hook::Exit { under, .. } => Some(*under),
            Hook::Turn(_) => None,
        }
    }
}

pub(crate) struct SessionSpans {
    trace: Trace,
    session: String,
    parent: Option<SpanRef>,
    /// The last instant seen, for an end with no step.
    now: Ts,
    root: Option<ManualSpan<AgentSession>>,
    start: Option<ManualSpan<SessionStart>>,
    close: Option<ManualSpan<SessionClose>>,
    update: Option<ManualSpan<UpdateAgent>>,
    /// `session_start` and `update_agent` are over once the activities they hold end.
    start_done: bool,
    update_done: bool,
    hooks: BTreeMap<HookId, Hook>,
    user: user_turn::UserTurns,
    agent: agent_turn::AgentTurns,
    amd: Option<ManualSpan<Amd>>,
    keyterm: Option<ManualSpan<KeytermDetection>>,
    closed: bool,
}

impl SessionSpans {
    pub(crate) fn new(
        session: &str,
        telemetry: SessionTelemetry,
        now: crate::agent_session::Now,
    ) -> Self {
        let SessionTelemetry { parent, job, ids } = telemetry;
        let ids = ids.unwrap_or_else(|| seeded_ids(fnv(session)));
        let conversation = Some(session.to_owned());
        SessionSpans {
            trace: Trace::new(ids, now.at, now.wall, job, conversation),
            session: session.to_owned(),
            parent,
            now: now.at,
            root: None,
            start: None,
            close: None,
            update: None,
            start_done: false,
            update_done: false,
            hooks: BTreeMap::new(),
            user: Default::default(),
            agent: Default::default(),
            amd: None,
            keyterm: None,
            closed: false,
        }
    }

    /// The turns, with the trace and the session span, while the session is open.
    pub(crate) fn turns<R>(
        &mut self,
        now: Ts,
        f: impl FnOnce(
            &mut Trace,
            &ManualSpan<AgentSession>,
            &mut user_turn::UserTurns,
            &mut agent_turn::AgentTurns,
        ) -> R,
    ) -> Option<R> {
        self.now = self.now.max(now);
        let root = self.root.as_ref()?;
        Some(f(&mut self.trace, root, &mut self.user, &mut self.agent))
    }

    /// Ended spans since the last step.
    pub(crate) fn drain(&mut self) -> Vec<Record> {
        self.trace.drain()
    }

    /// `start()`, as the session is built: `agent_session` and `session_start` open.
    pub(crate) fn open(&mut self, now: Ts, agent: &str) {
        if self.root.is_some() || self.closed {
            return;
        }
        self.now = now;
        let decl = AgentSession {
            agent_label: agent.to_owned(),
            operation: GenAiOperation::InvokeWorkflow,
            workflow: AgentSession::NAME.to_owned(),
            dispatch_id: None,
            worker_id: None,
            agent_id: None,
        };
        let root = self.trace.open(now, self.parent, decl);
        self.start = Some(root.child(&mut self.trace, now, SessionStart));
        self.root = Some(root);
    }

    /// `session_start`'s context while it is open.
    pub(crate) fn starting(&self) -> Option<SpanRef> {
        self.start.as_ref().map(ManualSpan::context)
    }

    /// The opening cycle is done: `session_start` ends, unless the first `on_enter` still runs.
    pub(crate) fn opened(&mut self, now: Ts) {
        self.start_done = true;
        self.settle(now);
    }

    /// `session_start` and `update_agent` end once done and holding no running activity.
    fn settle(&mut self, now: Ts) {
        let busy = |under| self.hooks.values().any(|h| h.under() == Some(under));
        let (start, update) = (busy(Under::Start), busy(Under::Update));
        if self.start_done
            && !start
            && let Some(span) = self.start.take()
        {
            span.end(&mut self.trace, now, ());
        }
        if self.update_done
            && !update
            && let Some(span) = self.update.take()
        {
            self.update_done = false;
            span.end(&mut self.trace, now, ());
        }
    }

    /// A handoff's `update_agent`, opened now unless one is open.
    fn update<'a>(
        update: &'a mut Option<ManualSpan<UpdateAgent>>,
        done: &mut bool,
        trace: &mut Trace,
        root: &ManualSpan<AgentSession>,
        now: Ts,
        decl: impl FnOnce() -> UpdateAgent,
    ) -> &'a ManualSpan<UpdateAgent> {
        update.get_or_insert_with(|| {
            *done = false;
            root.child(trace, now, decl())
        })
    }

    /// `session_close`, from the request to the close.
    pub(crate) fn closing(
        &mut self,
        now: Ts,
        reason: lk_agents_abi::voice::events::CloseReason,
        drain: bool,
        error: Option<&ErrorEvent>,
    ) {
        use lk_agents_abi::voice::events::CloseReason as R;
        if self.close.is_some() {
            return;
        }
        self.now = self.now.max(now);
        let (Some(root), trace) = (self.root.as_ref(), &mut self.trace) else { return };
        let reason = match reason {
            R::Error => CloseReason::Error,
            R::JobShutdown => CloseReason::JobShutdown,
            R::ParticipantDisconnected => CloseReason::ParticipantDisconnected,
            R::UserInitiated => CloseReason::UserInitiated,
            R::TaskCompleted => CloseReason::TaskCompleted,
        };
        let decl = SessionClose { reason, drain, error: error.map(|e| e.error.error_type()) };
        self.close = Some(root.child(trace, now, decl));
    }

    /// A callback the host was asked to run, in the span its activity runs in: `on_enter` in
    /// `start_agent_activity`, `on_exit` in `drain_agent_activity`, `on_user_turn_completed` in
    /// its user turn.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn hook_issued(
        &mut self,
        now: Ts,
        id: HookId,
        kind: HookKind,
        agent: &str,
        turn: Option<TurnId>,
        arriving: Option<&str>,
        llm: Option<&lk_agents_abi::ProviderId>,
    ) {
        self.now = self.now.max(now);
        let (Some(root), trace) = (self.root.as_ref(), &mut self.trace) else { return };
        let label = agent.to_owned();
        let (update, done) = (&mut self.update, &mut self.update_done);
        let hook = match kind {
            HookKind::OnEnter => {
                let decl = StartAgentActivity {
                    agent_label: label.clone(),
                    operation: GenAiOperation::CreateAgent,
                    agent_name: label.clone(),
                    provider: llm.and_then(|l| {
                        lk_agents_core::telemetry::gen_ai::provider_name(&l.provider)
                    }),
                    model: llm.map(|l| l.model.clone()),
                };
                let (under, activity) = match &self.start {
                    Some(start) if update.is_none() => {
                        (Under::Start, start.child(trace, now, decl))
                    }
                    _ => {
                        let update = Self::update(update, done, trace, root, now, || UpdateAgent {
                            previous_agent_label: String::new(),
                            agent_label: label.clone(),
                        });
                        (Under::Update, update.child(trace, now, decl))
                    }
                };
                let hook = activity.child(trace, now, OnEnter { agent_label: label });
                Hook::Enter { under, activity, hook }
            }
            HookKind::OnExit => {
                let decl = DrainAgentActivity { agent_label: label.clone() };
                let (under, drain) = match &self.close {
                    Some(close) => (Under::Close, close.child(trace, now, decl)),
                    None => {
                        let update = Self::update(update, done, trace, root, now, || UpdateAgent {
                            previous_agent_label: label.clone(),
                            agent_label: arriving.unwrap_or_default().to_owned(),
                        });
                        (Under::Update, update.child(trace, now, decl))
                    }
                };
                let hook = drain.child(trace, now, OnExit { agent_label: label });
                Hook::Exit { under, drain, hook }
            }
            HookKind::OnUserTurnCompleted => {
                let Some(turn) = turn else { return };
                self.user.hook_issued(trace, now, turn, id, agent);
                Hook::Turn(turn)
            }
            HookKind::OnUserTurnExceeded
            | HookKind::FillerSource
            | HookKind::FunctionToolsExecuted => {
                return;
            }
        };
        self.hooks.insert(id, hook);
    }

    /// The callback returned, raised or timed out: its span ends, and the activity it ran in.
    pub(crate) fn hook_finished(
        &mut self,
        now: Ts,
        id: HookId,
        failure: Option<HookFailure<'_>>,
        stop_response: bool,
    ) {
        self.now = self.now.max(now);
        let Some(hook) = self.hooks.remove(&id) else { return };
        let failure = failure.as_ref().map(HookFailure::failure);
        let trace = &mut self.trace;
        match hook {
            Hook::Enter { under, activity, hook } => {
                if let Some(failure) = failure {
                    hook.fail(trace, now, failure);
                }
                hook.end(trace, now, ());
                activity.end(trace, now, ());
                match under {
                    Under::Start => self.start_done = true,
                    Under::Update => self.update_done = true,
                    Under::Close => {}
                }
            }
            Hook::Exit { drain, hook, .. } => {
                if let Some(failure) = failure {
                    hook.fail(trace, now, failure);
                }
                hook.end(trace, now, ());
                drain.end(trace, now, ());
            }
            Hook::Turn(turn) => self.user.hook_finished(trace, now, turn, failure, stop_response),
        }
        self.settle(now);
    }

    /// A `VoiceTask` suspends `previous` for `next`: `next`'s `on_enter` runs under this
    /// `update_agent`, which ends with it, or [`SessionSpans::agent_updated`] without one.
    pub(crate) fn agent_suspended(&mut self, now: Ts, previous: &str, next: &str) {
        self.now = self.now.max(now);
        let (Some(root), trace) = (self.root.as_ref(), &mut self.trace) else { return };
        let (update, done) = (&mut self.update, &mut self.update_done);
        let update = Self::update(update, done, trace, root, now, || UpdateAgent {
            previous_agent_label: previous.to_owned(),
            agent_label: next.to_owned(),
        });
        let pause = PauseAgentActivity { agent_label: previous.to_owned() };
        update.child(trace, now, pause).end(trace, now, ());
    }

    /// The task ended and its parent is back, without its hooks.
    pub(crate) fn agent_resumed(&mut self, now: Ts, previous: &str, next: &str) {
        self.agent_updated(now);
        self.now = self.now.max(now);
        let (Some(root), trace) = (self.root.as_ref(), &mut self.trace) else { return };
        let update = root.child(
            trace,
            now,
            UpdateAgent { previous_agent_label: previous.to_owned(), agent_label: next.to_owned() },
        );
        let resume = ResumeAgentActivity { agent_label: next.to_owned() };
        update.child(trace, now, resume).end(trace, now, ());
        update.end(trace, now, ());
    }

    /// The update is over with no `on_enter` to end it.
    pub(crate) fn agent_updated(&mut self, now: Ts) {
        if self.update.is_some() {
            self.update_done = true;
            self.settle(now);
        }
    }

    /// The AMD detector attached (`None`: `amd` opens) or released its verdict (it ends).
    pub(crate) fn amd(&mut self, now: Ts, verdict: Option<&AmdPrediction>) {
        self.now = self.now.max(now);
        let (Some(root), trace) = (self.root.as_ref(), &mut self.trace) else { return };
        let Some(p) = verdict else {
            if self.amd.is_none() {
                self.amd = Some(root.child(trace, now, Amd));
            }
            return;
        };
        let Some(span) = self.amd.take() else { return };
        let verdict = AmdVerdict {
            category: p.category.as_str().to_owned(),
            reason: p.reason.as_str().to_owned(),
            speech_duration: Seconds(p.speech_duration.as_secs_f64()),
            delay: Seconds(p.delay.as_secs_f64()),
            transcript: Content(p.transcript.clone()),
        };
        span.end(trace, now, Some(verdict));
    }

    /// A keyterm pass was requested, under the replying turn's span when one is given. Its
    /// `llm_request` nests under the returned span.
    pub(crate) fn keyterm_requested(
        &mut self,
        now: Ts,
        llm: Option<&lk_agents_abi::ProviderId>,
    ) -> Option<SpanRef> {
        self.keyterm_answered(now, None);
        self.now = self.now.max(now);
        let (Some(root), trace) = (self.root.as_ref(), &mut self.trace) else { return None };
        let decl = KeytermDetection {
            model: llm.map(|l| l.model.clone()).unwrap_or_default(),
            provider: llm
                .and_then(|l| lk_agents_core::telemetry::gen_ai::provider_name(&l.provider))
                .unwrap_or_default(),
        };
        let span = root.child(trace, now, decl);
        let context = span.context();
        self.keyterm = Some(span);
        Some(context)
    }

    /// The pass answered with the effective lists before and after it, or failed (`None`).
    pub(crate) fn keyterm_answered(&mut self, now: Ts, lists: Option<(&[String], &[String])>) {
        let Some(span) = self.keyterm.take() else { return };
        let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        let end = match lists {
            Some((before, after)) => KeytermDetectionEnd {
                count: count(after.len()),
                added: count(after.iter().filter(|t| !before.contains(t)).count()),
                removed: count(before.iter().filter(|t| !after.contains(t)).count()),
            },
            None => KeytermDetectionEnd { count: 0, added: 0, removed: 0 },
        };
        span.end(&mut self.trace, now, end);
    }

    /// What a session event records: state changes on the session, the turns it moves.
    pub(crate) fn observe(&mut self, now: Ts, ev: &SessionEvent) {
        self.now = self.now.max(now);
        let (Some(root), trace) = (self.root.as_ref(), &mut self.trace) else { return };
        match ev {
            SessionEvent::AgentStateChanged { old, new } => {
                let event =
                    AgentStateChanged { old: old.as_str().into(), new: new.as_str().into() };
                root.event(trace, now, event);
                if *old == lk_agents_abi::voice::events::AgentState::Speaking {
                    self.agent.stopped(trace, now);
                }
            }
            SessionEvent::UserStateChanged { old, new, at } => {
                let event = UserStateChanged { old: old.as_str().into(), new: new.as_str().into() };
                root.event(trace, (*at).min(now), event);
            }
            SessionEvent::OverlappingSpeech(e) => self.user.overlap(e),
            SessionEvent::UserTranscriptionTimeout(_) => {
                self.user.abandon(trace, now, UserTurnOutcome::TranscriptionTimeout);
            }
            SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended { call_id, .. }) => {
                self.agent.tool_ended(trace, now, call_id);
            }
            SessionEvent::ConversationItemAdded { item, source, .. } => {
                use crate::agent::Source;
                use lk_agents_core::tts::markup::ChatMessageExt;
                let lk_agents_abi::ChatItem::Message(m) = &**item else { return };
                let text = m.text_content().unwrap_or_default();
                match source {
                    Source::Reply { speech, .. } | Source::Say { speech } => {
                        self.agent.response(*speech, &text)
                    }
                    Source::UserInput { .. } | Source::UserTyped => {
                        self.user.typed(trace, root, now, &text)
                    }
                    _ => {}
                }
            }
            SessionEvent::Close { .. } => self.close_all(now),
            _ => {}
        }
    }

    /// The linked participant joined or left.
    pub(crate) fn participant(&mut self, now: Ts, identity: &str, left: Option<&str>) {
        self.now = self.now.max(now);
        let (Some(root), trace) = (self.root.as_ref(), &mut self.trace) else { return };
        let participant = ParticipantAttrs {
            id: String::new(),
            identity: Content(identity.to_owned()),
            kind: String::new(),
        };
        match left {
            None => root.event(trace, now, ParticipantLinked { participant }),
            Some(reason) => root.event(
                trace,
                now,
                ParticipantDisconnected { participant, reason: reason.into() },
            ),
        }
    }

    /// A speech's first generation: its turn opens; the runtime's work for it nests under the
    /// returned span.
    pub(crate) fn speech_started(
        &mut self,
        now: Ts,
        speech: SpeechId,
        agent: &str,
        user: Option<lk_agents_abi::MetricsReport>,
    ) -> Option<SpanRef> {
        let session = self.session.clone();
        self.turns(now, |trace, root, _, turns| {
            turns.started(trace, root, now, &session, speech, agent, user)
        })
    }

    /// The close: every span still open ends, children first, then `session_close` and the
    /// session. The sweep after it finds nothing.
    fn close_all(&mut self, now: Ts) {
        let Some(root) = self.root.take() else { return };
        self.closed = true;
        let trace = &mut self.trace;
        self.agent.close(trace, now);
        self.user.close(trace, now);
        for (_, hook) in std::mem::take(&mut self.hooks) {
            match hook {
                Hook::Enter { activity, hook, .. } => {
                    hook.end(trace, now, ());
                    activity.end(trace, now, ());
                }
                Hook::Exit { drain, hook, .. } => {
                    hook.end(trace, now, ());
                    drain.end(trace, now, ());
                }
                Hook::Turn(_) => {}
            }
        }
        if let Some(span) = self.amd.take() {
            span.end(trace, now, None);
        }
        self.keyterm_answered(now, None);
        let trace = &mut self.trace;
        for span in [self.start.take()].into_iter().flatten() {
            span.end(trace, now, ());
        }
        if let Some(update) = self.update.take() {
            update.end(trace, now, ());
        }
        if let Some(close) = self.close.take() {
            close.end(trace, now, ());
        }
        let end = AgentSessionEnd {
            blocking_count: None,
            blocking_total: None,
            blocking_max: None,
            sip: Default::default(),
            sip_phone_number: None,
        };
        root.end(trace, now, end);
        trace.close(now);
    }
}

/// A session dropped without its close (a test, a cut-off run) still ends what it opened.
impl Drop for SessionSpans {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let now = self.now;
            self.close_all(now);
        }
    }
}

/// The default id seed: stable per session id.
fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

#[cfg(test)]
#[path = "spans_tests.rs"]
mod tests;
