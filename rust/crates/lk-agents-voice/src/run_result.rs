//! Voice-run identity and result selection. Work membership and liveness stay in lifecycle owners.

use crate::agent::Source;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::VoiceDriver;
use crate::events::{
    DiagnosticEvent, RunFailure, RunId, RunOutcome, RunRef, RunRefusal, SessionEvent,
};
use lk_agents_abi::agent::run::{RunError, RunInputRequirement};
use lk_agents_abi::{HostRef, Ts};
use std::collections::BTreeMap;

/// `session._run(user_input=)`: the remote session's typed input and the voice test harness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSpec {
    /// The run's opening reply; `None` owns whatever follows (e.g. `on_enter`).
    pub user_input: Option<String>,
    /// The host's own object for the run (its `RunResult`), handed back on every run event.
    pub host: Option<HostRef>,
}

/// What a speech's reply settled on, for the run ending on the speech.
#[derive(Debug, Clone, PartialEq)]
pub enum RunOutput {
    /// The output the reply's agent accepted.
    Value(HostRef),
    /// The reply ended on a framework outcome (retries spent, a validator failed).
    Failed(RunError),
    /// A tool asked the person (`InputRequired`).
    Ask { question: String, verbatim: bool, answer_tool: Option<String> },
}

struct OpenRun {
    run: RunRef,
    participated: bool,
    input_requirements: Vec<RunInputRequirement>,
    /// `__last_speech_handle`, and whether it failed: its error and output decide the verdict.
    last: Option<(SpeechId, bool)>,
}

pub(super) enum Decided {
    /// Still owed something, or nothing was released since the last check.
    Open,
    /// Nothing is owned: the run is over.
    Settled {
        run: RunRef,
        outcome: RunOutcome,
        speech: Option<SpeechId>,
        input_requirements: Vec<RunInputRequirement>,
    },
}

/// The driver's run state: at most one open run and the recorded tool outputs.
#[derive(Default)]
pub(super) struct Runs {
    open: Option<OpenRun>,
    next: u64,
    /// accepted output's value by the speech whose tool called it, until that speech ends.
    pub(crate) outputs: BTreeMap<SpeechId, RunOutput>,
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
        });
        Ok(run)
    }

    /// The open run's verdict will name `speech` when it settles.
    pub(super) fn names(&self, speech: SpeechId) -> bool {
        self.open.as_ref().and_then(|r| r.last).is_some_and(|(last, _)| last == speech)
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
    fn speech_ended(&mut self, speech: SpeechId, failed: bool) {
        if let Some(run) = &mut self.open
            && let Some((previous, _)) = run.last.replace((speech, failed))
            && previous != speech
        {
            self.outputs.remove(&previous);
        }
    }

    /// Settle after attributed owner lifecycles finish; a capture run first needs participation.
    pub(super) fn decide(&mut self, has_work: bool) -> Decided {
        let Some(r) = self.open.as_ref() else { return Decided::Open };
        if !r.participated || has_work {
            return Decided::Open;
        }
        let speech = r.last.map(|(speech, _)| speech);
        let output = speech.and_then(|speech| self.outputs.remove(&speech));
        let failed = r.last.is_some_and(|(_, failed)| failed);
        let outcome = match (failed, output) {
            // `__last_speech_handle._error` (`run_result.RunResult._mark_done`).
            (true, _) => RunOutcome::Failed(RunFailure::Speech),
            // `isinstance(final_output, BaseException)` (`.resolve_expressive_options`).
            (false, Some(RunOutput::Failed(e))) => RunOutcome::Failed(RunFailure::Run(e)),
            (false, Some(RunOutput::Value(v))) => RunOutcome::Completed { final_output: Some(v) },
            (false, Some(RunOutput::Ask { question, verbatim, answer_tool })) => {
                RunOutcome::InputRequired { question, verbatim, answer_tool }
            }
            (false, None) => RunOutcome::Completed { final_output: None },
        };
        let run = r.run.clone();
        let input_requirements = self.open.take().expect("open run checked").input_requirements;
        Decided::Settled { run, outcome, speech, input_requirements }
    }

    pub(super) fn close(&mut self) -> Option<(RunRef, Option<SpeechId>, Vec<RunInputRequirement>)> {
        self.outputs.clear();
        self.open.take().map(|r| (r.run, r.last.map(|(speech, _)| speech), r.input_requirements))
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
            self.realtime_reply(now, user_input, None, None, None, None, false, true);
            return;
        }
        let _ =
            self.generate_reply_speech(now, user_input, None, None, None, None, None, false, None);
    }

    /// Membership lives with the scheduler entry and is installed before admission can reject
    /// a speech. Completion includes playout: only this terminal release chooses the result.
    pub(crate) fn run_speech_ended(&mut self, speech: SpeechId, failed: bool) {
        if self.run_contains_speech(speech) {
            self.session.runs.speech_ended(speech, failed);
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
        self.session.runs.outputs.insert(speech, RunOutput::Failed(error));
    }

    /// The run an item belongs to: the open run for its own speeches and the handoff record. The
    /// `run(user_input=)` user message is not the run's, as in the reference (`RunResult.events`
    /// opens with the reply), and neither is a committed call: a call became the run's when it
    /// started (`RunItemAdded`).
    pub(crate) fn run_of_item(&self, source: &Source) -> Option<RunRef> {
        let open = self.session.runs.current()?;
        match source {
            Source::Reply { speech, .. }
            | Source::Say { speech }
            | Source::ToolOutput { speech, .. }
            | Source::ToolUpdate { speech, .. } => {
                self.activity.speech_belongs_to_run(*speech, open.id).then_some(open)
            }
            Source::Handoff => Some(open),
            Source::UserInput { .. }
            | Source::ToolCall { .. }
            | Source::UserTurn { .. }
            | Source::ConfigUpdate
            | Source::HostCtx
            | Source::UserTyped
            | Source::Remote { .. } => None,
        }
    }

    /// The settle check after every wake.
    pub(crate) fn run_check(&mut self) {
        let has_work = self.session.runs.current().is_some_and(|run| {
            self.activity.has_run_work(run.id) || self.session.idle.has_run_work(run.id)
        });
        if let Decided::Settled { run, outcome, speech, input_requirements } =
            self.session.runs.decide(has_work)
        {
            self.publish_event(SessionEvent::RunSettled {
                run,
                outcome,
                speech,
                input_requirements,
            });
        }
    }

    /// The close's hard edge: an open run ends `Closed`, before `Close`.
    pub(crate) fn run_close(&mut self) {
        if let Some((run, speech, input_requirements)) = self.session.runs.close() {
            self.publish_event(SessionEvent::RunSettled {
                run,
                outcome: RunOutcome::Closed,
                speech,
                input_requirements,
            });
        }
    }

    /// The loop ended with a run open: settled `Closed` directly on the stream.
    pub(crate) fn run_close_at_end(&mut self) {
        if let Some((run, speech, input_requirements)) = self.session.runs.close() {
            self.emit_now(SessionEvent::RunSettled {
                run,
                outcome: RunOutcome::Closed,
                speech,
                input_requirements,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture() -> Runs {
        let mut runs = Runs::default();
        runs.open(&RunSpec { user_input: None, host: None }, false).unwrap();
        runs
    }

    #[test]
    fn capture_waits_for_participation_and_then_for_owner_liveness() {
        let mut runs = capture();
        assert!(matches!(runs.decide(false), Decided::Open));
        runs.participated();
        assert!(matches!(runs.decide(true), Decided::Open));
        assert!(matches!(runs.decide(false), Decided::Settled { .. }));
        assert!(matches!(runs.decide(false), Decided::Open), "settlement is emitted once");
    }

    #[test]
    fn result_precedence_releases_non_selected_values() {
        let mut runs = capture();
        runs.participated();
        let first = SpeechId(1);
        let last = SpeechId(2);
        runs.outputs.insert(
            first,
            RunOutput::Ask { question: "old?".into(), verbatim: true, answer_tool: None },
        );
        runs.speech_ended(first, false);
        assert!(runs.outputs.contains_key(&first));
        runs.speech_ended(last, false);
        assert!(!runs.outputs.contains_key(&first));
        assert!(matches!(runs.decide(false), Decided::Settled {
            outcome: RunOutcome::Completed { final_output: None }, speech: Some(id), ..
        } if id == last));
        assert!(runs.outputs.is_empty());
    }

    #[test]
    fn close_releases_all_result_references_and_settles_once() {
        let mut runs = capture();
        runs.outputs.insert(SpeechId(1), RunOutput::Failed(RunError::Cancelled));
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
            RunOutput::Ask {
                question: "ignored by speech error?".into(),
                verbatim: true,
                answer_tool: None,
            },
        );
        runs.speech_ended(speech, true);
        assert!(matches!(
            runs.decide(false),
            Decided::Settled { outcome: RunOutcome::Failed(RunFailure::Speech), .. }
        ));
        assert!(runs.outputs.is_empty());
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
        use crate::agent_activity::scheduling::{PauseCause, SpeechKind};
        use crate::agent_activity::speech::SpeechSpec;
        use crate::generation::ReplySource;
        let mut driver = crate::agent_session::tests::driver();
        driver.on_run(Ts::ZERO, RunSpec { user_input: None, host: None });
        driver.pause_scheduling(Ts::ZERO, PauseCause::Drain);
        let speech = driver.start_speech(
            Ts::ZERO,
            SpeechSpec::new(SpeechKind::Say, ReplySource::Text("hello".into()), true),
        );
        driver.run_check();
        assert!(
            driver.session.runs.current().is_none(),
            "a terminal speech must not remain pending"
        );
        let settled = driver.settle_step().into_iter().find_map(|effect| match effect {
            crate::agent_session::effects::VoiceEffect::Emit(SessionEvent::RunSettled {
                speech,
                ..
            }) => Some(speech),
            _ => None,
        });
        assert_eq!(
            settled,
            Some(Some(speech)),
            "the rejected admission still determines its run result"
        );
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
        runs.speech_ended(SpeechId(1), false);
        runs.speech_ended(SpeechId(2), false);
        let (_, _, requirements) = runs.close().unwrap();
        assert_eq!(requirements, [requirement]);
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
        use crate::agent_activity::scheduling::SpeechKind;
        use crate::agent_activity::speech::SpeechSpec;
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
                participates_in_voice_run: false,
                ..SpeechSpec::new(SpeechKind::Say, ReplySource::Text("incidental".into()), true)
            },
        );
        driver.session.runs.outputs.insert(
            host,
            RunOutput::Ask { question: "owned?".into(), verbatim: true, answer_tool: None },
        );
        driver.session.runs.outputs.insert(
            unrelated,
            RunOutput::Ask { question: "unrelated?".into(), verbatim: true, answer_tool: None },
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
                    outcome,
                    ..
                }) => Some(outcome),
                _ => None,
            })
            .collect();
        assert!(
            matches!(settled.as_slice(), [RunOutcome::InputRequired { question, .. }] if question == "owned?")
        );
        assert!(driver.session.runs.outputs.is_empty());
    }
}
