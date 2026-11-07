//! Voice-run identity, result selection and recording. A run settles into the `RunResult` a text
//! run settles into: its items, usage and outcome are the core's. Work membership and liveness
//! stay in lifecycle owners.

use crate::agent::Source;
use crate::agent_session::VoiceDriver;
use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::events::{
    DiagnosticEvent, RunFailure, RunId, RunOutcome, RunRef, RunRefusal, SessionEvent,
};
use lk_agents_abi::agent::run::{
    Output, RunError, RunInputRequirement, RunOutcome as Outcome, RunResult, RunSettled,
};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{ChatItem, HostRef, Role, Ts};
use lk_agents_core::tts::markup::ChatMessageExt;
use lk_agents_core::usage::{Ledger, ModelKind};
use std::collections::BTreeMap;

/// `session._run(user_input=)`: the remote session's typed input and the voice test harness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSpec {
    /// The run's opening reply; `None` owns whatever follows (e.g. `on_enter`).
    pub user_input: Option<String>,
    /// The host's own object for the run (the future its `_run()` awaits), handed back when the
    /// run settles.
    pub host: Option<HostRef>,
}

struct OpenRun {
    run: RunRef,
    participated: bool,
    input_requirements: Vec<RunInputRequirement>,
    /// `__last_speech_handle`, and the error it failed with: they decide the verdict.
    last: Option<(SpeechId, Option<ProviderError>)>,
    /// What the run recorded, each item once by identity, in time order.
    items: Vec<ChatItem>,
    /// The requests and tool bodies of the speeches the run owns (usage.md).
    usage: Ledger,
}

impl OpenRun {
    /// The run's item, replacing an earlier version of it; only messages, calls and outputs.
    fn record(&mut self, item: &ChatItem) {
        if !matches!(
            item,
            ChatItem::Message(_) | ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_)
        ) {
            return;
        }
        if let Some(known) = self.items.iter_mut().find(|known| {
            known.id() == item.id()
                && std::mem::discriminant(*known) == std::mem::discriminant(item)
        }) {
            *known = item.clone();
            return;
        }
        let index = self
            .items
            .iter()
            .rposition(|known| known.created_at() <= item.created_at())
            .map_or(0, |index| index + 1);
        self.items.insert(index, item.clone());
    }

    /// The settled run: the last speech's failure or output decides; without one, the last
    /// assistant reply's text, else no output. `history` is the session's after the run.
    fn settle(
        mut self,
        output: Option<RunOutcome>,
        history: Vec<ChatItem>,
    ) -> (RunRef, Result<Box<RunSettled>, RunFailure>) {
        self.usage.settle();
        let failure = |error, provider_error, input_requirements| RunFailure {
            error,
            provider_error,
            input_requirements,
        };
        let (outcome, value) = match (self.last.and_then(|(_, error)| error), output) {
            // `__last_speech_handle._error` (`run_result.RunResult._mark_done`).
            (Some(error), _) => {
                let model = RunError::Model(error.to_string());
                return (self.run, Err(failure(model, Some(error), self.input_requirements)));
            }
            (None, Some(RunOutcome::Failed(error))) => {
                return (self.run, Err(failure(error, None, self.input_requirements)));
            }
            (None, Some(RunOutcome::Completed { validation, final_output })) => {
                (Outcome::Completed { output: Output::Validated(validation) }, Some(final_output))
            }
            (None, Some(RunOutcome::InputRequired(question))) => {
                (Outcome::InputRequired(question), None)
            }
            (None, None) => {
                let text = self.items.iter().rev().find_map(|item| match item {
                    ChatItem::Message(message) if message.role == Role::Assistant => Some(message),
                    _ => None,
                });
                match text.and_then(|message| message.text_content()) {
                    Some(text) => (Outcome::Completed { output: Output::Text(text) }, None),
                    None => (Outcome::NoOutput, None),
                }
            }
        };
        let verbatim = match &outcome {
            Outcome::InputRequired(question) if question.verbatim => {
                Some(question.question.clone())
            }
            _ => None,
        };
        let result = RunResult {
            outcome,
            interrupted: false,
            new_messages: self.items,
            all_messages: history,
            usage: self.usage.snapshot(),
            verbatim,
            pending_call: None,
            input_requirements: self.input_requirements,
        };
        (self.run, Ok(Box::new(RunSettled { result, output: value })))
    }

    /// The session closed under the run: `RuntimeError` (`AgentSession._aclose_impl`).
    fn close(self) -> (RunRef, Result<Box<RunSettled>, RunFailure>) {
        let failure = RunFailure {
            error: RunError::SessionClosed,
            provider_error: None,
            input_requirements: self.input_requirements,
        };
        (self.run, Err(failure))
    }
}

/// The driver's run state: at most one open run and the recorded tool outputs.
#[derive(Default)]
pub(super) struct Runs {
    open: Option<OpenRun>,
    next: u64,
    /// accepted output's value by the speech whose tool called it, until that speech ends.
    pub(crate) outputs: BTreeMap<SpeechId, RunOutcome>,
}

impl Runs {
    /// `Command::Run`: open, or refuse: a run already open, or a closing session.
    pub(super) fn open(&mut self, spec: &RunSpec, closing: bool) -> Result<RunRef, RunRefusal> {
        if self.open.is_some() {
            return Err(RunRefusal::Nested);
        }
        if closing {
            return Err(RunRefusal::Closing);
        }
        self.next += 1;
        let run = RunRef { id: RunId(self.next), host: spec.host.clone() };
        self.open = Some(OpenRun {
            run: run.clone(),
            participated: false,
            input_requirements: Vec::new(),
            last: None,
            items: Vec::new(),
            usage: Ledger::default(),
        });
        Ok(run)
    }

    /// The open run's verdict will name `speech` when it settles.
    pub(super) fn names(&self, speech: SpeechId) -> bool {
        self.open.as_ref().and_then(|r| r.last.as_ref()).is_some_and(|(last, _)| *last == speech)
    }

    /// The open run, as its events name it.
    pub(super) fn current(&self) -> Option<RunRef> {
        self.open.as_ref().map(|r| r.run.clone())
    }

    /// Capture runs wait for their first admitted work instead of completing while empty.
    pub(super) fn participated(&mut self) {
        if let Some(run) = &mut self.open {
            run.participated = true;
        }
    }

    /// The open run's item: one its speeches recorded, or a call one of them started.
    pub(crate) fn record(&mut self, item: &ChatItem) {
        if let Some(run) = &mut self.open {
            run.record(item);
        }
    }

    /// The open run's usage, fed alongside the session's for the requests its speeches start.
    pub(crate) fn usage(&mut self) -> Option<&mut Ledger> {
        self.open.as_mut().map(|run| &mut run.usage)
    }

    pub(crate) fn retain_input_requirements(&mut self, requirements: &[RunInputRequirement]) {
        if let Some(run) = &mut self.open {
            for requirement in requirements {
                if !run.input_requirements.iter().any(|retained| {
                    retained.invocation_id == requirement.invocation_id
                        && retained.call.call_id == requirement.call.call_id
                }) {
                    run.input_requirements.push(requirement.clone());
                }
            }
        }
    }

    /// Keep only the last completed participating speech's result. Intermediate host values
    /// are released as soon as result precedence moves to another speech.
    fn speech_ended(&mut self, speech: SpeechId, error: Option<ProviderError>) {
        if let Some(run) = &mut self.open
            && let Some((previous, _)) = run.last.replace((speech, error))
            && previous != speech
        {
            self.outputs.remove(&previous);
        }
    }

    /// Settle after attributed owner lifecycles finish; a capture run first needs participation.
    /// The run and the last speech's output, when it is over.
    fn decide(&mut self, has_work: bool) -> Option<(OpenRun, Option<RunOutcome>)> {
        let r = self.open.as_ref()?;
        if !r.participated || has_work {
            return None;
        }
        let output = r.last.as_ref().and_then(|(speech, _)| self.outputs.remove(speech));
        self.open.take().map(|run| (run, output))
    }

    fn close(&mut self) -> Option<OpenRun> {
        self.outputs.clear();
        self.open.take()
    }
}

/// The open run's ownership, as the driver records it while the run is open.
impl VoiceDriver {
    /// `Command::Run`: `VoiceSession._run()`.
    pub(crate) fn on_run(&mut self, now: Ts, spec: RunSpec) {
        let closing = self.session.closing.is_some();
        let run = match self.session.runs.open(&spec, closing) {
            Ok(run) => run,
            Err(cause) => {
                self.publish_event(SessionEvent::RunRefused { cause, host: spec.host });
                return;
            }
        };
        // Before the reply, so the run's start precedes its items on the host stream.
        self.publish_event(SessionEvent::RunStarted { run });
        if let Some(text) = spec.user_input {
            self.run_reply(now, Some(text));
        }
    }

    /// The run's own `generate_reply(..)`, pipeline or realtime; owned like any session reply.
    fn run_reply(&mut self, now: Ts, user_input: Option<String>) {
        if self.activity.realtime().is_some() {
            let role =
                crate::agent_session::machine::activity::realtime::RealtimeReplyRole::VoiceRun;
            self.realtime_reply(now, user_input, None, None, None, None, role);
            return;
        }
        let _ =
            self.generate_reply_speech(now, user_input, None, None, None, None, None, false, None);
    }

    /// Membership lives with the scheduler entry and is installed before admission can reject
    /// a speech. Completion includes playout: only this terminal release chooses the result.
    pub(crate) fn run_speech_ended(&mut self, speech: SpeechId, error: Option<ProviderError>) {
        if self.run_contains_speech(speech) {
            self.session.runs.speech_ended(speech, error);
        } else {
            self.session.runs.outputs.remove(&speech);
        }
    }

    pub(crate) fn run_contains_speech(&self, speech: SpeechId) -> bool {
        self.session
            .runs
            .current()
            .is_some_and(|run| self.activity.speech_belongs_to_run(speech, run.id))
    }

    /// A provider request of `speech` started: the open run counts it when the speech is its.
    pub(crate) fn start_request(
        &mut self,
        speech: Option<SpeechId>,
        kind: ModelKind,
        provider: &str,
        model: &str,
    ) -> u64 {
        let request = self.session.start_request(kind, provider, model);
        if speech.is_some_and(|speech| self.run_contains_speech(speech))
            && let Some(usage) = self.session.runs.usage()
        {
            usage.start(request, kind, provider, model);
        }
        request
    }

    /// The reply of `speech` ended on a framework failure (a budget spent, a validator failed): a
    /// run settling on the speech fails with it, as `agent.run()` does.
    pub(crate) fn run_failed(&mut self, speech: SpeechId, error: RunError) {
        if let RunError::ToolRetriesExceeded { tool, retries } = &error {
            let d = DiagnosticEvent::error(crate::events::diagnostics::TOOL_RETRIES_EXCEEDED)
                .with("function", tool.clone())
                .with("max_retries", retries.to_string())
                .with("speech_id", speech.0.to_string());
            self.publish_event(SessionEvent::Diagnostic(d));
        }
        self.session.runs.outputs.insert(speech, RunOutcome::Failed(error));
    }

    /// Whether an item is the open run's: its own speeches' items and the handoff record. The
    /// `run(user_input=)` user message is not, as in the reference (`RunResult.events` opens with
    /// the reply), and neither is a committed call: a call became the run's when it started.
    pub(crate) fn run_owns_item(&self, source: &Source) -> bool {
        let Some(open) = self.session.runs.current() else { return false };
        match source {
            Source::Reply { speech, .. }
            | Source::Say { speech }
            | Source::ToolOutput { speech, .. }
            | Source::ToolUpdate { speech, .. } => {
                self.activity.speech_belongs_to_run(*speech, open.id)
            }
            Source::Handoff => true,
            Source::UserInput { .. }
            | Source::ToolCall { .. }
            | Source::UserTurn { .. }
            | Source::ConfigUpdate
            | Source::HostCtx
            | Source::UserTyped
            | Source::Remote { .. } => false,
        }
    }

    /// The settle check after every wake.
    pub(crate) fn run_check(&mut self) {
        let has_work = self.session.runs.current().is_some_and(|run| {
            self.activity.has_run_work(run.id) || self.session.idle.has_run_work(run.id)
        });
        if let Some((open, output)) = self.session.runs.decide(has_work) {
            let history = self.session.chat_ctx.history().cloned().collect();
            let (run, result) = open.settle(output, history);
            self.publish_event(SessionEvent::RunSettled { run, result });
        }
    }

    /// The close's hard edge: an open run fails `SessionClosed`, before `Close`.
    pub(crate) fn run_close(&mut self) {
        if let Some(open) = self.session.runs.close() {
            let (run, result) = open.close();
            self.publish_event(SessionEvent::RunSettled { run, result });
        }
    }

    /// The loop ended with a run open: failed `SessionClosed` directly on the stream.
    pub(crate) fn run_close_at_end(&mut self) {
        if let Some(open) = self.session.runs.close() {
            let (run, result) = open.close();
            self.emit_now(SessionEvent::RunSettled { run, result });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::ChatMessage;
    use lk_agents_abi::agent::run::RunQuestion;
    use std::time::{Duration, SystemTime};

    fn capture() -> Runs {
        let mut runs = Runs::default();
        runs.open(&RunSpec { user_input: None, host: None }, false).unwrap();
        runs
    }

    #[test]
    fn capture_waits_for_participation_and_then_for_owner_liveness() {
        let mut runs = capture();
        assert!(runs.decide(false).is_none());
        runs.participated();
        assert!(runs.decide(true).is_none());
        assert!(runs.decide(false).is_some());
        assert!(runs.decide(false).is_none(), "settlement is emitted once");
    }

    #[test]
    fn result_precedence_releases_non_selected_values() {
        let mut runs = capture();
        runs.participated();
        let first = SpeechId(1);
        let last = SpeechId(2);
        runs.outputs.insert(
            first,
            RunOutcome::InputRequired(RunQuestion {
                question: "old?".into(),
                verbatim: true,
                answer_tool: None,
            }),
        );
        runs.speech_ended(first, None);
        assert!(runs.outputs.contains_key(&first));
        runs.speech_ended(last, None);
        assert!(!runs.outputs.contains_key(&first));
        let (open, output) = runs.decide(false).expect("settled");
        assert!(output.is_none());
        assert!(matches!(open.last, Some((id, None)) if id == last));
        assert!(runs.outputs.is_empty());
    }

    #[test]
    fn close_releases_all_result_references_and_settles_once() {
        let mut runs = capture();
        runs.outputs.insert(SpeechId(1), RunOutcome::Failed(RunError::Cancelled));
        assert!(runs.close().is_some());
        assert!(runs.outputs.is_empty());
        assert!(runs.close().is_none());
    }

    #[test]
    fn last_completed_participating_speech_preserves_failure_precedence() {
        let mut runs = capture();
        runs.participated();
        let speech = SpeechId(7);
        runs.outputs.insert(
            speech,
            RunOutcome::InputRequired(RunQuestion {
                question: "ignored by speech error?".into(),
                verbatim: true,
                answer_tool: None,
            }),
        );
        let error = ProviderError::Local(lk_agents_abi::errors::ApiError::new("tts failed"));
        runs.speech_ended(speech, Some(error.clone()));
        let (open, output) = runs.decide(false).expect("settled");
        assert!(runs.outputs.is_empty());
        let (_, result) = open.settle(output, Vec::new());
        let failure = result.expect_err("the speech's failure decides");
        assert_eq!(failure.provider_error, Some(error));
        assert!(matches!(failure.error, RunError::Model(_)));
    }

    fn reply(id: &str, text: &str, seconds: u64) -> ChatItem {
        let at = SystemTime::UNIX_EPOCH + Duration::from_secs(seconds);
        ChatMessage::text(id, Role::Assistant, text, at).into()
    }

    #[test]
    fn recording_replaces_identity_orders_equal_stamps_and_answers_the_last_reply() {
        let mut runs = capture();
        runs.participated();
        runs.record(&reply("later", "old", 2));
        runs.record(&reply("earlier", "first", 1));
        runs.record(&reply("later", "updated", 2));
        runs.record(&reply("equal", "last", 2));
        let (open, output) = runs.decide(false).expect("settled");
        let (_, result) = open.settle(output, Vec::new());
        let settled = result.expect("completed");
        assert_eq!(
            settled.result.new_messages.iter().map(ChatItem::id).collect::<Vec<_>>(),
            vec!["earlier", "later", "equal"]
        );
        assert!(settled.result.usage.settled);
        assert!(matches!(
            &settled.result.outcome,
            Outcome::Completed { output: Output::Text(text) } if text == "last"
        ));
    }

    #[test]
    fn a_question_is_the_outcome_not_the_preceding_assistant_text() {
        let mut runs = capture();
        runs.participated();
        runs.record(&reply("message", "earlier reply", 0));
        let speech = SpeechId(3);
        let question =
            RunQuestion { question: "Confirm?".into(), verbatim: true, answer_tool: None };
        runs.outputs.insert(speech, RunOutcome::InputRequired(question.clone()));
        runs.speech_ended(speech, None);
        let (open, output) = runs.decide(false).expect("settled");
        let settled = open.settle(output, Vec::new()).1.expect("asks");
        assert_eq!(settled.result.outcome, Outcome::InputRequired(question));
        assert_eq!(settled.result.verbatim.as_deref(), Some("Confirm?"));
    }

    #[test]
    fn a_last_assistant_message_without_text_is_no_output() {
        let mut runs = capture();
        runs.participated();
        runs.record(&reply("old", "older reply", 0));
        let mut latest = ChatMessage::text("new", Role::Assistant, "", SystemTime::UNIX_EPOCH);
        latest.content.clear();
        runs.record(&latest.into());
        let (open, output) = runs.decide(false).expect("settled");
        let settled = open.settle(output, Vec::new()).1.expect("settles");
        assert_eq!(settled.result.outcome, Outcome::NoOutput);
    }

    #[test]
    fn the_run_counts_only_its_own_speeches_requests() {
        use crate::agent_session::machine::activity::scheduling::SpeechKind;
        use crate::agent_session::machine::activity::speech::SpeechSpec;
        use crate::generation::ReplySource;
        let mut driver = crate::agent_session::tests::driver();
        driver.on_run(Ts::ZERO, RunSpec { user_input: None, host: None });
        let speech = driver.start_speech(
            Ts::ZERO,
            SpeechSpec::new(SpeechKind::Say, ReplySource::Text("owned".into()), true),
        );
        driver.start_request(Some(speech), ModelKind::Tts, "cartesia", "sonic");
        driver.start_request(None, ModelKind::Llm, "openai", "gpt");
        assert_eq!(driver.session.runs.usage().expect("open").snapshot().requests, 1);
        assert_eq!(driver.session.usage.snapshot().requests, 2);
    }
    #[test]
    fn no_host_enter_completion_counts_as_participation_without_stranding_capture() {
        let mut driver = crate::agent_session::tests::driver();
        driver.on_run(Ts::ZERO, RunSpec { user_input: None, host: None });
        driver.issue_on_enter(Ts::ZERO);
        driver.run_check();
        let settled = driver
            .settle_step()
            .into_iter()
            .filter(|effect| {
                matches!(effect,
                    crate::agent_session::effects::VoiceEffect::Emit(event)
                        if matches!(event, SessionEvent::RunSettled { .. })
                )
            })
            .count();
        assert_eq!(settled, 1);
        assert!(driver.session.runs.current().is_none());
    }

    #[test]
    fn rejected_speech_is_attributed_before_its_immediate_terminal_release() {
        use crate::agent_session::machine::activity::scheduling::{PauseCause, SpeechKind};
        use crate::agent_session::machine::activity::speech::SpeechSpec;
        use crate::generation::ReplySource;
        let mut driver = crate::agent_session::tests::driver();
        driver.on_run(Ts::ZERO, RunSpec { user_input: None, host: None });
        driver.pause_scheduling(Ts::ZERO, PauseCause::Drain);
        let speech = driver.start_speech(
            Ts::ZERO,
            SpeechSpec::new(SpeechKind::Say, ReplySource::Text("hello".into()), true),
        );
        assert!(
            driver.session.runs.names(speech),
            "the rejected admission still determines its run result"
        );
        driver.run_check();
        assert!(
            driver.session.runs.current().is_none(),
            "a terminal speech must not remain pending"
        );
        let settled = driver.settle_step().into_iter().filter(|effect| {
            matches!(
                effect,
                crate::agent_session::effects::VoiceEffect::Emit(SessionEvent::RunSettled { .. })
            )
        });
        assert_eq!(settled.count(), 1);
    }

    #[test]
    fn retained_requirements_survive_result_precedence_and_close() {
        let mut runs = capture();
        let requirement = RunInputRequirement {
            invocation_id: "speech-1".into(),
            call: lk_agents_abi::a2a::turn::PendingCall {
                name: "flight".into(),
                call_id: "call-1".into(),
                answer_tool: Some("answer".into()),
            },
            question: "Which day?".into(),
            verbatim: true,
        };
        runs.retain_input_requirements(std::slice::from_ref(&requirement));
        // A coordinated step and its terminal envelope report the same request. Keep one
        // correlated obligation even when success/failure projections overlap.
        runs.retain_input_requirements(std::slice::from_ref(&requirement));
        runs.speech_ended(SpeechId(1), None);
        runs.speech_ended(SpeechId(2), None);
        assert_eq!(runs.close().unwrap().input_requirements, [requirement]);
    }

    #[test]
    fn hold_membership_survives_waiting_to_granted_and_ends_on_release() {
        use crate::agent_session::{HoldId, IdleHolds, IdleWaiter};
        let run = RunId(3);
        let hold = HoldId(4);
        let mut idle = IdleHolds::default();
        idle.ask(IdleWaiter {
            run: Some(run),
            hold: None,
            speech: Some(SpeechId(2)),
            grant: Some(hold),
            call: None,
        });
        assert!(idle.has_run_work(run));
        assert!(!idle.has_run_work(RunId(99)));
        assert_eq!(idle.admit().len(), 1);
        assert!(idle.has_run_work(run), "granting must not lose attribution");
        assert!(idle.release(hold));
        assert!(!idle.has_run_work(run));
        assert!(!idle.release(hold), "duplicate release is inert");
    }

    #[test]
    fn withdrawing_a_pending_hold_removes_its_run_obligation() {
        use crate::agent_session::{HoldId, IdleHolds, IdleWaiter};
        let run = RunId(3);
        let mut idle = IdleHolds::default();
        let speech = SpeechId(2);
        idle.ask(IdleWaiter {
            run: Some(run),
            hold: None,
            speech: Some(speech),
            grant: Some(HoldId(4)),
            call: None,
        });
        assert_eq!(idle.withdraw(speech).len(), 1);
        assert!(!idle.has_run_work(run));
    }

    #[test]
    fn unrelated_speech_neither_extends_the_run_nor_replaces_its_result() {
        use crate::agent_session::machine::activity::scheduling::SpeechKind;
        use crate::agent_session::machine::activity::speech::SpeechSpec;
        use crate::generation::ReplySource;
        let mut driver = crate::agent_session::tests::driver();
        driver.on_run(Ts::ZERO, RunSpec { user_input: None, host: None });
        let host = driver.start_speech(
            Ts::ZERO,
            SpeechSpec::new(SpeechKind::Say, ReplySource::Text("owned".into()), true),
        );
        let unrelated = driver.start_speech(
            Ts::ZERO,
            SpeechSpec {
                voice_run: crate::agent_session::machine::activity::scheduling::VoiceRunParticipation::Outside,
                ..SpeechSpec::new(SpeechKind::Say, ReplySource::Text("incidental".into()), true)
            },
        );
        driver.session.runs.outputs.insert(
            host,
            RunOutcome::InputRequired(RunQuestion {
                question: "owned?".into(),
                verbatim: true,
                answer_tool: None,
            }),
        );
        driver.session.runs.outputs.insert(
            unrelated,
            RunOutcome::InputRequired(RunQuestion {
                question: "unrelated?".into(),
                verbatim: true,
                answer_tool: None,
            }),
        );
        driver.release_speech(Ts::ZERO, host);
        driver.run_check();
        assert!(driver.session.runs.current().is_none(), "unrelated speech cannot extend the run");
        driver.release_speech(Ts::ZERO, unrelated);
        let settled: Vec<_> = driver
            .settle_step()
            .into_iter()
            .filter_map(|effect| match effect {
                crate::agent_session::effects::VoiceEffect::Emit(SessionEvent::RunSettled {
                    result: Ok(settled),
                    ..
                }) => Some(settled.result.outcome),
                _ => None,
            })
            .collect();
        assert!(
            matches!(settled.as_slice(), [Outcome::InputRequired(ask)] if ask.question == "owned?")
        );
        assert!(driver.session.runs.outputs.is_empty());
    }
}
