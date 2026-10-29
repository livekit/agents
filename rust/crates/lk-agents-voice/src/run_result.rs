//! `VoiceSession._run()`: the run open on the session, what it owns, and its verdict.

use crate::agent::Source;
use crate::agent_activity::handoff::HandoffId;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::Out;
use crate::agent_session::{Driver, HoldId, HookId, Origin};
use crate::events::{
    DiagnosticEvent, RunFailure, RunId, RunOutcome, RunRef, RunRefusal, SessionEvent,
};
use crate::trace::HandoffStep;
use lk_agents_abi::agent::run::RunError;
use lk_agents_abi::{HostRef, Ts};
use std::collections::{BTreeMap, BTreeSet};

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

/// One thing a run owns while it is live. The run settles when none is left.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, lk_agents_abi::Vocabulary)]
pub enum Owned {
    /// A session-level speech created while the run was open, or the run's own reply.
    Speech(SpeechId),
    /// An `on_enter` issued during the run.
    Hook(HookId),
    /// A handoff begun during the run, until it landed or was refused.
    Handoff(HandoffId),
    /// A foreground hold taken by an owned speech's tool, until released.
    Hold(HoldId),
}

struct OpenRun {
    run: RunRef,
    owned: BTreeSet<Owned>,
    /// `__last_speech_handle`, and whether it failed: its error and output decide the verdict.
    last: Option<(SpeechId, bool)>,
}

pub(super) enum Decided {
    /// Still owed something, or nothing was released since the last check.
    Open,
    /// Nothing is owned: the run is over.
    Settled { run: RunRef, outcome: RunOutcome, speech: Option<SpeechId> },
}

/// The driver's run state: at most one open run and the recorded tool outputs.
#[derive(Default)]
pub(super) struct Runs {
    open: Option<OpenRun>,
    next: u64,
    /// accepted output's value by the speech whose tool called it, until that speech ends.
    pub(crate) outputs: BTreeMap<SpeechId, RunOutput>,
    /// A release happened since the last `decide`: the check is due.
    dirty: bool,
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
        self.open = Some(OpenRun { run: run.clone(), owned: BTreeSet::new(), last: None });
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

    pub(super) fn owns(&self, what: Owned) -> bool {
        self.open.as_ref().is_some_and(|r| r.owned.contains(&what))
    }

    /// Own `what`, if a run is open and it is not owned already: `Some(run)` when newly owned.
    pub(super) fn own(&mut self, what: Owned) -> Option<RunId> {
        let r = self.open.as_mut()?;
        r.owned.insert(what).then_some(r.run.id)
    }

    /// Release `what`; an ended speech (and whether it failed) becomes the verdict speech.
    pub(super) fn release(
        &mut self,
        what: Owned,
        ended: Option<(SpeechId, bool)>,
    ) -> Option<RunId> {
        let r = self.open.as_mut()?;
        if !r.owned.remove(&what) {
            return None;
        }
        if ended.is_some() {
            r.last = ended;
        }
        self.dirty = true;
        Some(r.run.id)
    }

    /// The verdict, once nothing is owned and something was released (`_mark_done_if_needed`).
    pub(super) fn decide(&mut self) -> Decided {
        if !self.dirty {
            return Decided::Open;
        }
        self.dirty = false;
        let Some(r) = self.open.as_ref() else { return Decided::Open };
        if !r.owned.is_empty() {
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
        self.open = None;
        Decided::Settled { run, outcome, speech }
    }

    pub(super) fn close(&mut self) -> Option<(RunRef, Option<SpeechId>)> {
        self.dirty = false;
        self.open.take().map(|r| (r.run, r.last.map(|(speech, _)| speech)))
    }
}

/// The open run's ownership, as the driver records it while the run is open.
impl Driver {
    /// `Command::Run`: `VoiceSession._run()`.
    pub(crate) fn on_run(&mut self, now: Ts, spec: RunSpec, out: &mut Out) {
        let closing = self.session.closing.is_some();
        let run = match self.session.runs.open(&spec, closing) {
            Ok(run) => run,
            Err(cause) => {
                out.machine().emit(SessionEvent::RunRefused { cause, host: spec.host });
                return;
            }
        };
        // Before the reply, so the run's start precedes its items on the host stream.
        out.machine().emit(SessionEvent::RunStarted { run });
        if let Some(text) = spec.user_input {
            self.run_reply(now, Some(text), out);
        }
    }

    /// The run's own `generate_reply(..)`, pipeline or realtime; owned like any session reply.
    fn run_reply(&mut self, now: Ts, user_input: Option<String>, out: &mut Out) {
        if self.activity.rt_session.is_some() {
            self.realtime_reply(now, user_input, None, None, None, None, false, out);
            return;
        }
        let _ = self
            .generate_reply_speech(now, user_input, None, None, None, None, None, false, None, out);
    }

    pub(crate) fn run_own(&mut self, what: Owned, out: &mut Out) {
        if let Some(run) = self.session.runs.own(what) {
            out.trace(HandoffStep::RunOwned { run, what });
        }
    }

    pub(crate) fn run_release(&mut self, what: Owned, out: &mut Out) {
        if let Some(run) = self.session.runs.release(what, None) {
            out.trace(HandoffStep::RunReleased { run, what });
        }
    }

    /// A speech left the loop (`release_speech`): the run's verdict moves to it.
    pub(crate) fn run_speech_ended(&mut self, speech: SpeechId, failed: bool, out: &mut Out) {
        let what = Owned::Speech(speech);
        match self.session.runs.release(what, Some((speech, failed))) {
            Some(run) => out.trace(HandoffStep::RunReleased { run, what }),
            // Outside any run: its recorded output goes with it.
            None => {
                self.session.runs.outputs.remove(&speech);
            }
        }
    }

    /// The reply of `speech` ended on a framework failure (a budget spent, a validator failed): a
    /// run settling on the speech fails with it, as `agent.run()` does.
    pub(crate) fn run_failed(&mut self, speech: SpeechId, error: RunError, out: &mut Out) {
        if let RunError::ToolRetriesExceeded { tool, retries } = &error {
            let d = DiagnosticEvent::error(crate::events::diagnostics::TOOL_RETRIES_EXCEEDED)
                .with("function", tool.clone())
                .with("max_retries", retries.to_string())
                .with("speech_id", speech.0.to_string());
            out.machine().emit(SessionEvent::Diagnostic(d));
        }
        self.session.runs.outputs.insert(speech, RunOutput::Failed(error));
    }

    /// A `WaitForIdleAndHold` from an owned speech's tool is the run's.
    pub(crate) fn run_own_hold(&mut self, origin: &Origin, hold: HoldId, out: &mut Out) {
        let Origin::Tool { call_id } = origin else { return };
        if let Some(speech) = self.activity.tools.speech_of(call_id)
            && self.session.runs.owns(Owned::Speech(speech))
        {
            self.run_own(Owned::Hold(hold), out);
        }
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
                self.session.runs.owns(Owned::Speech(*speech)).then_some(open)
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
    pub(crate) fn run_check(&mut self, out: &mut Out) {
        if let Decided::Settled { run, outcome, speech } = self.session.runs.decide() {
            out.machine().emit(SessionEvent::RunSettled { run, outcome, speech });
        }
    }

    /// The close's hard edge: an open run ends `Closed`, before `Close`.
    pub(crate) fn run_close(&mut self, out: &mut Out) {
        if let Some((run, speech)) = self.session.runs.close() {
            out.machine().emit(SessionEvent::RunSettled {
                run,
                outcome: RunOutcome::Closed,
                speech,
            });
        }
    }

    /// The loop ended with a run open: settled `Closed` directly on the stream.
    pub(crate) fn run_close_at_end(&mut self) {
        if let Some((run, speech)) = self.session.runs.close() {
            self.emit_now(SessionEvent::RunSettled { run, outcome: RunOutcome::Closed, speech });
        }
    }
}
