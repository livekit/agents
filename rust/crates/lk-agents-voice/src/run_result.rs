use crate::agent_activity::hooks::HandoffId;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{HoldId, HookId};
use crate::events::{RunFailure, RunId, RunOutcome, RunRefusal};
use crate::speech_handle::SpeechHandle;
use lk_agents_abi::HostRef;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// `session.run(user_input=, output_type=, ..)` or `start(capture_run=True)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSpec {
    /// The run's opening reply; `None` owns whatever follows (e.g. `on_enter`).
    pub user_input: Option<String>,
    /// `output_type is not None`: a run ending without output is re-prompted.
    pub expects_output: bool,
    /// `RunOutputOptions.max_retries` (default 2; `output_options=None` is 0).
    pub max_retries: u32,
    /// `RunOutputOptions.retry_instructions`: the per-turn system message of the re-prompt.
    pub retry_instructions: String,
}

/// What a tool handed `AgentTask.complete()` from inside a speech (`_maybe_run_final_output`).
#[derive(Debug, Clone, PartialEq)]
pub enum RunOutput {
    /// `complete(result)`: the interned result object.
    Value(HostRef),
    /// `complete(exception)`: the interned exception, which the run raises.
    Error(HostRef),
}

/// One thing a run owns while it is live. The run settles when none is left.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, lk_agents_abi::Vocabulary)]
pub enum Owned {
    /// A session-level speech created while the run was open, or the run's own reply.
    Speech(SpeechId),
    /// An `on_enter` issued during the run, or the `DescribeAgent` a tool's handoff waits on.
    Hook(HookId),
    /// A frame change begun during the run, until it landed or was refused.
    Handoff(HandoffId),
    /// A foreground hold taken by an owned speech's tool, until released.
    Hold(HoldId),
}

struct OpenRun {
    id: RunId,
    owned: BTreeSet<Owned>,
    /// `__last_speech_handle`: its error and output decide the verdict.
    last: Option<Arc<SpeechHandle>>,
    expects_output: bool,
    retries_left: u32,
    retries: u32,
    retry_instructions: String,
}

pub(super) enum Decided {
    /// Still owed something, or nothing was released since the last check.
    Open,
    /// Nothing is owned, the run expects an output and the last speech produced none: re-prompt.
    Retry { run: RunId, attempt: u32, instructions: String },
    /// Nothing is owned: the run is over.
    Settled { run: RunId, outcome: RunOutcome, speech: Option<Arc<SpeechHandle>> },
}

/// The driver's run state: at most one open run and the recorded tool outputs.
#[derive(Default)]
pub(super) struct Runs {
    open: Option<OpenRun>,
    next: u64,
    /// `AgentTask.complete()`'s value by the speech whose tool called it, until that speech ends.
    outputs: BTreeMap<SpeechId, RunOutput>,
    /// A release happened since the last `decide`: the check is due.
    dirty: bool,
}

impl Runs {
    /// `Command::Run`: open, or refuse -- a run already open, or a closing session.
    pub(super) fn open(&mut self, spec: &RunSpec, closing: bool) -> Result<RunId, RunRefusal> {
        if self.open.is_some() {
            return Err(RunRefusal::Nested);
        }
        if closing {
            return Err(RunRefusal::Closing);
        }
        self.next += 1;
        let id = RunId(self.next);
        self.open = Some(OpenRun {
            id,
            owned: BTreeSet::new(),
            last: None,
            expects_output: spec.expects_output,
            retries_left: spec.max_retries,
            retries: 0,
            retry_instructions: spec.retry_instructions.clone(),
        });
        Ok(id)
    }

    pub(super) fn id(&self) -> Option<RunId> {
        self.open.as_ref().map(|r| r.id)
    }

    pub(super) fn owns(&self, what: Owned) -> bool {
        self.open.as_ref().is_some_and(|r| r.owned.contains(&what))
    }

    /// The foreground holds the open run owns (`session._foreground_guards` it watches).
    pub(super) fn owned_holds(&self) -> Vec<HoldId> {
        let Some(r) = &self.open else { return Vec::new() };
        r.owned.iter().filter_map(|o| if let Owned::Hold(h) = o { Some(*h) } else { None }).collect()
    }

    /// Own `what`, if a run is open and it is not owned already: `Some(run)` when newly owned.
    pub(super) fn own(&mut self, what: Owned) -> Option<RunId> {
        let r = self.open.as_mut()?;
        r.owned.insert(what).then_some(r.id)
    }

    /// Release `what`; `ended` becomes the verdict speech (`None` for an inline task's suspension).
    pub(super) fn release(
        &mut self,
        what: Owned,
        ended: Option<&Arc<SpeechHandle>>,
    ) -> Option<RunId> {
        let r = self.open.as_mut()?;
        if !r.owned.remove(&what) {
            return None;
        }
        if let Some(h) = ended {
            r.last = Some(Arc::clone(h));
        }
        self.dirty = true;
        Some(r.id)
    }

    pub(super) fn record_output(&mut self, speech: SpeechId, output: RunOutput) {
        self.outputs.insert(speech, output);
    }

    /// A speech left the loop outside any run: its recorded output goes with it.
    pub(super) fn forget_speech(&mut self, speech: SpeechId) {
        self.outputs.remove(&speech);
    }

    /// The verdict, once nothing is owned and something was released (`_mark_done_if_needed`).
    pub(super) fn decide(&mut self) -> Decided {
        if !self.dirty {
            return Decided::Open;
        }
        self.dirty = false;
        let Some(r) = self.open.as_mut() else { return Decided::Open };
        if !r.owned.is_empty() {
            return Decided::Open;
        }
        let speech = r.last.clone();
        let output = speech.as_ref().and_then(|h| self.outputs.remove(&h.id()));
        let failed = speech.as_ref().is_some_and(|h| matches!(h.exception(), Ok(Some(_))));
        let outcome = match (failed, output) {
            // `__last_speech_handle._error` (`run_result.RunResult._mark_done`).
            (true, _) => RunOutcome::Failed(RunFailure::Speech),
            // `isinstance(final_output, BaseException)` (`.resolve_expressive_options`).
            (false, Some(RunOutput::Error(e))) => RunOutcome::Failed(RunFailure::Output(e)),
            (false, Some(RunOutput::Value(v))) => RunOutcome::Completed { final_output: Some(v) },
            (false, None) => {
                // Only the no-output case retries, and only with a speech to read.
                if r.expects_output && speech.is_some() && r.retries_left > 0 {
                    r.retries_left -= 1;
                    r.retries += 1;
                    return Decided::Retry {
                        run: r.id,
                        attempt: r.retries,
                        instructions: r.retry_instructions.clone(),
                    };
                }
                RunOutcome::Completed { final_output: None }
            }
        };
        let run = r.id;
        self.open = None;
        Decided::Settled { run, outcome, speech }
    }

    pub(super) fn close(&mut self) -> Option<(RunId, Option<Arc<SpeechHandle>>)> {
        self.dirty = false;
        self.open.take().map(|r| (r.id, r.last))
    }
}
