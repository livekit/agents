//! Releasing what a run owes, then reporting what it found.
//!
//! Beside the session it drives and not in `search`: replay needs exactly this and nothing of
//! the campaign, so changing how the search picks inputs cannot break a saved case. The rules run
//! once, over the finished recording; the only check between inputs is that the run is still
//! moving.
use super::*;

// ==============================================================================================
// The release phase
// ==============================================================================================

/// How far one release round may move the clock. Bounded, so one timer that re-arms itself cannot
/// run the phase away.
pub(crate) const RELEASE_CAP_MS: u64 = 10_000;

impl Run {
    /// One round of releasing everything the case left pending: end every stream, answer every
    /// tool, end every chat, report every flushed segment, let every synthesis through, close
    /// every utterance, release every hold and claim, and answer every hook.
    ///
    /// Read off the live set, which is already the answer to "what has this run opened and not
    /// closed". Empty means the run is quiet and the settle can begin.
    pub fn releases(&self) -> Vec<Input> {
        let blank = || AgentText(String::new());
        let each: [(Kind, &dyn Fn(u32) -> Input); 8] = [
            (Kind::Stream, &|nth| Input::FeedEnd { stream: Ref::new(nth) }),
            (Kind::Chat, &|nth| Input::LlmEnd { chat: Ref::new(nth) }),
            // A prediction left pending holds the endpointing phase open until it times out.
            (Kind::Prediction, &|nth| Input::EotAnswer {
                prediction: Ref::new(nth),
                probability: Confidence(90),
                backchannel: None,
            }),
            (Kind::Segment, &|nth| Input::PlayoutFinished { segment: Ref::new(nth) }),
            (Kind::Tts, &|nth| Input::TtsRelease { tts: Ref::new(nth) }),
            // A recognition stream left mid-utterance keeps the caller "speaking", which keeps the
            // session from ever being idle.
            (Kind::Utterance, &|nth| Input::SttEnd { utterance: Ref::new(nth) }),
            (Kind::Hold, &|nth| Input::ReleaseIdleHold { hold: Ref::new(nth) }),
            (Kind::Hook, &|nth| Input::HookDone {
                hook: Ref::new(nth),
                answer: HookAnswer::Returned,
                text: blank(),
            }),
        ];
        let mut out: Vec<Input> = Vec::new();
        for (kind, make) in each {
            out.extend(self.record.live().of(kind).into_iter().map(make));
        }
        // A realtime turn the server is still speaking is the case's to end.
        if let Some(socket) = self.record.live().of(Kind::Realtime).first().copied() {
            for _ in self.record.live().of(Kind::Generation) {
                out.push(Input::RealtimeEvent {
                    socket: Ref::new(socket),
                    event: RtEvent::GenerationDone,
                });
            }
        }
        // A person the case left mid-sentence keeps the driver waiting for the end of their turn,
        // and a reply behind that turn never starts. End the speech on every detector that heard
        // a start without an end.
        for nth in self.record.live().of(Kind::Vad) {
            let still_speaking =
                self.record.applied().iter().filter(|a| a.accepted).fold(0i64, |n, a| {
                    match &a.input {
                        Input::VadStart { vad, .. } if vad.nth == nth => n + 1,
                        Input::VadEnd { vad, .. } if vad.nth == nth => n - 1,
                        _ => n,
                    }
                });
            if still_speaking > 0 {
                out.push(Input::VadEnd { vad: Ref::new(nth), silence_ms: Millis(500) });
            }
        }
        // A running tool is answered; one that detached is ended by the host's final update.
        for nth in self.record.live().of(Kind::Tool) {
            out.push(if self.tool_detached(nth) {
                Input::ToolUpdate { tool: Ref::new(nth), progress: None }
            } else {
                Input::ToolAnswered {
                    tool: Ref::new(nth),
                    answer: ToolAnswer::Return,
                    text: blank(),
                }
            });
        }
        out.extend(
            (0..self.outstanding(Input::ClaimUserTurn {}, Input::ReleaseUserTurn {}))
                .map(|_| Input::ReleaseUserTurn {}),
        );
        // A latched pause stops the scheduler authorizing anything at all, so a speech waiting on
        // it is waiting on the case, not on the driver.
        out.extend(
            (0..self.outstanding(Input::PauseAuthorization {}, Input::ResumeAuthorization {}))
                .map(|_| Input::ResumeAuthorization {}),
        );
        // A turn left mid-decision keeps the session from being idle too, and no timer will move
        // it: a transcript that never reached a final leaves the turn parked, waiting for words
        // that are not coming. `ClearTurn` is the product's own neutral release, whereas a
        // synthetic final would invent caller words and a commit would ask for a reply.
        if self.turn_undecided() {
            out.push(Input::ClearTurn {});
        }
        // A navigator the case started is the case's to close. Left running it does what a
        // navigator does -- prompts the line again at every silence, forever -- and a release
        // phase that only answers what is owed never gets to quiet.
        if self.ivr_navigating() {
            out.push(Input::Ivr {
                input: IvrInput::Close,
                silence: Millis(0),
                text: UserText(String::new()),
            });
        }
        out
    }

    /// Whether the IVR navigator has acted since the case last closed it: the navigator's own
    /// triggers, against the case's closes, in order.
    fn ivr_navigating(&self) -> bool {
        let last_close = self
            .record
            .sent(|i| matches!(i, Input::Ivr { input: IvrInput::Close, .. }).then_some(()))
            .filter(|(_, (), accepted)| *accepted)
            .map(|(at, (), _)| at.seq)
            .last();
        self.record
            .journal_where(|step| match step {
                Step::Detector(DetectorStep::IvrTriggered { .. }) => Some(()),
                _ => None,
            })
            .any(|(at, ())| last_close.is_none_or(|close| at.seq > close))
    }

    /// How far to move the clock in one release round: just past the furthest deadline the driver
    /// has armed, so every timer it is waiting on fires and whatever that frees becomes releasable.
    /// Applying an input never yields, so without this the driver never runs between rounds and
    /// the next round reads fakes that have not moved.
    pub fn release_advance(&self) -> Input {
        let now = self.record.now().at_ms;
        let furthest = self
            .record
            .journal_where(|step| match step {
                Step::Core(CoreStep::TimerArmed { at, .. }) => Some(*at),
                _ => None,
            })
            .map(|(_, at)| at.saturating_duration_since(Ts::ZERO).as_millis() as u64)
            .max()
            .unwrap_or(now);
        Input::Advance { ms: Millis((furthest.saturating_sub(now) + 1).min(RELEASE_CAP_MS)) }
    }

    /// Whether the driver is still holding a turn open.
    ///
    /// Read from the driver's own decisions rather than from the case's inputs: an end of speech
    /// the case delivered can declare a turn after the case has already cleared one, so what the
    /// case sent is not what the session currently holds.
    fn turn_undecided(&self) -> bool {
        let mut declared = false;
        for (_, step) in self.record.journal() {
            match step {
                Step::Decided(Decision::TurnDeclared { .. }) => declared = true,
                Step::Turn(TurnStep::EndOfTurn { .. } | TurnStep::CommitAnswered { .. }) => {
                    declared = false
                }
                Step::Core(CoreStep::CommandReceived { kind, .. })
                    if *kind == "TurnCommand::Clear" =>
                {
                    declared = false
                }
                _ => {}
            }
        }
        declared
    }

    /// How many `open` inputs the case applied without a matching `close`, in order. A close with
    /// nothing to close is a no-op in the product, so the count never goes below zero.
    fn outstanding(&self, open: Input, close: Input) -> u32 {
        let mut count: u32 = 0;
        for applied in self.record.applied() {
            if applied.input == open {
                count += 1;
            } else if applied.input == close {
                count = count.saturating_sub(1);
            }
        }
        count
    }
}

// ==============================================================================================
// Running a case to the end
// ==============================================================================================

/// How many quiet release rounds end a run. Two, because one can just be a gap between timers.
const QUIET_ROUNDS: u32 = 2;

/// The most release rounds a run is given.
///
/// Each round can create the work the next one releases, so the loop can fail to terminate.
/// Reaching this is a finding, not a limit to raise.
const MAX_RELEASE_ROUNDS: u32 = 100;

/// What one run found.
///
/// `expectations` separates a contract that passed from one that never applied. A report that
/// cannot tell those apart shows a rule as green long after its state stopped being reachable.
#[derive(Debug, Default)]
pub struct Outcome {
    pub case: Case,
    pub failures: Vec<Failure>,
    pub expectations: BTreeMap<&'static str, u32>,
    /// Journal variants this run produced, as `Family::Variant`.
    pub reached: Vec<String>,
    /// Ordered pairs of consecutive journal steps. Variants are a set and cannot see ordering;
    /// a driver's bugs are about ordering, so this is the metric that discriminates.
    pub edges: BTreeSet<(String, String)>,
    /// For each input the run sent, the kinds it was actually observed to open. Checked against
    /// what the table declares, so a wrong row cannot hide.
    pub opened: BTreeMap<&'static str, BTreeSet<Kind>>,
    /// The run stopped before its close: the session livelocked, or never went quiet. What a
    /// run never reached cannot be judged, so a reader of what is left live must not read it.
    pub stopped_early: bool,
    /// `(input, phase)`: which inputs landed in which phase of the conversation, read off the
    /// journal as it stood when each input was applied. COVERAGE.md's operation x phase table.
    pub cells: BTreeSet<(&'static str, &'static str)>,
}

impl Outcome {
    pub fn ok(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Run one case to completion in its own simulation.
///
/// One case, one runtime: the seeded scheduler is only reproducible from its start, so two cases
/// sharing a runtime would each begin at a different point in the stream and neither would replay
/// on its own.
pub fn run(case: Case) -> Outcome {
    simulate_with(case.schedule, run_in(case))
}

async fn run_in(case: Case) -> Outcome {
    let mut run = Run::start(&case.setup, case.schedule, case.policy).await;
    let mut outcome = Outcome { case: case.clone(), ..Outcome::default() };

    for input in &case.inputs {
        run.apply(input).await;
        if let Some(found) = guard(&run, &mut outcome) {
            return found;
        }
    }
    finish(run, outcome).await
}

/// Let everything in flight finish, then close the session and join its tasks.
///
/// Most of the interesting driver states are on the far side of a run's armed deadlines and owed
/// answers, so stopping mid-turn would never reach them.
pub async fn finish(mut run: Run, mut outcome: Outcome) -> Outcome {
    let mut quiet = 0;
    let mut rounds = 0;
    while quiet < QUIET_ROUNDS {
        rounds += 1;
        if rounds > MAX_RELEASE_ROUNDS {
            let never_quiet = Failure {
                contract: "session_settles".into(),
                subject: Subject::Session,
                detail: format!(
                    "still producing work to release after {MAX_RELEASE_ROUNDS} rounds and \
                     {} inputs: the run never went quiet",
                    outcome.case.inputs.len()
                ),
            };
            outcome.stopped_early = true;
            // What the run reached still counts, and its rules still get their say.
            record_outcome(&run, &mut outcome);
            outcome.failures.push(never_quiet);
            return outcome;
        }
        let mut offered = run.releases();
        if offered.is_empty() {
            quiet += 1;
        } else {
            quiet = 0;
        }
        // Advance regardless, so an armed deadline with nothing else to do still fires.
        offered.push(run.release_advance());
        for input in offered {
            outcome.case.inputs.push(input.clone());
            run.apply(&input).await;
            if let Some(found) = guard(&run, &mut outcome) {
                return found;
            }
        }
    }

    for input in [Input::Settle {}, Input::Close { drain: true }, Input::AwaitExit {}] {
        outcome.case.inputs.push(input.clone());
        run.apply(&input).await;
        if let Some(found) = guard(&run, &mut outcome) {
            return found;
        }
    }

    run.join_tasks().await;
    // What the tasks did on their way out is part of the run.
    run.collect();
    record_outcome(&run, &mut outcome);
    outcome
}

/// The one check that runs after every input: that the run is still moving. The rules run once,
/// at the end, over the whole recording -- they read the whole recording anyway, and run after
/// every input they were most of a survey's wall time and a case stopped at its first finding,
/// which cost the rest of what that run would have reached.
///
/// Moving is counted in steps at one virtual instant, never in real seconds, so a slow machine
/// cannot change a verdict. A livelocked session was stopped, and nothing applied after it would
/// reach it: the run ends there. What can block without stepping at all -- a task that never
/// yields -- is the watchdog's (`simulate_with`), which aborts rather than judges.
pub fn guard(run: &Run, outcome: &mut Outcome) -> Option<Outcome> {
    run.record.livelock()?;
    record_outcome(run, outcome);
    outcome.stopped_early = true;
    Some(std::mem::take(outcome))
}

fn record_outcome(run: &Run, outcome: &mut Outcome) {
    let (failures, expectations) = check_all(&run.record);
    outcome.failures = failures;
    outcome.expectations = expectations;
    outcome.reached = run.record.journal_names().into_iter().collect();
    let names: Vec<String> = run.record.journal().iter().map(|(_, s)| step_name(s)).collect();
    outcome.edges = names.windows(2).map(|w| (w[0].clone(), w[1].clone())).collect();
    outcome.opened = opened_by_input(&run.record);
    outcome.cells = cells_by_input(&run.record);
}

/// The phases of a conversation, in COVERAGE.md's column order.
pub const PHASES: [&str; 14] = [
    "starting",
    "idle",
    "listening",
    "person_speaking",
    "away",
    "thinking",
    "speaking",
    "overlap",
    "tool_running",
    "turn_hook",
    "draining",
    "on_exit",
    "on_enter",
    "closing",
];

/// For each input, every phase the conversation was in when it was applied.
fn cells_by_input(record: &Recording) -> BTreeSet<(&'static str, &'static str)> {
    use lk_agents_abi::voice::events::{AgentState, UserState};
    use lk_agents_voice::agent_session::HookKind;
    use lk_agents_voice::events::SessionEvent;
    use lk_agents_voice::trace::{CloseStep, HandoffStep, ToolStep};

    let (mut agent, mut user) = (AgentState::Initializing, UserState::Listening);
    let (mut tools, mut turn_hooks, mut on_exit, mut on_enter) = (0i32, 0i32, 0i32, 0i32);
    let (mut draining, mut closing) = (false, false);
    let mut journal = record.journal().iter().peekable();
    let mut out = BTreeSet::new();
    for applied in record.applied() {
        while let Some((_, step)) = journal.next_if(|(at, _)| at.seq < applied.at.seq) {
            match step {
                Step::Emitted(SessionEvent::AgentStateChanged { new, .. }) => agent = *new,
                Step::Emitted(SessionEvent::UserStateChanged { new, .. }) => user = *new,
                Step::Tool(ToolStep::CallStarted { .. }) => tools += 1,
                Step::Tool(ToolStep::CallEnded { .. }) => tools -= 1,
                Step::Handoff(HandoffStep::DrainStarted { .. }) => draining = true,
                Step::Handoff(HandoffStep::Drained { .. }) => draining = false,
                Step::Handoff(HandoffStep::HookIssued { kind, .. }) => match kind {
                    HookKind::OnExit => on_exit += 1,
                    HookKind::OnEnter => on_enter += 1,
                    HookKind::OnUserTurnCompleted => turn_hooks += 1,
                    _ => {}
                },
                Step::Handoff(HandoffStep::HookFinished { kind, .. }) => match kind {
                    HookKind::OnExit => on_exit -= 1,
                    HookKind::OnEnter => on_enter -= 1,
                    HookKind::OnUserTurnCompleted => turn_hooks -= 1,
                    _ => {}
                },
                Step::Close(CloseStep::Requested { .. }) => closing = true,
                _ => {}
            }
        }
        let person = user == UserState::Speaking;
        let phases = [
            agent == AgentState::Initializing,
            agent == AgentState::Idle,
            agent == AgentState::Listening && !person,
            person,
            user == UserState::Away,
            agent == AgentState::Thinking,
            agent == AgentState::Speaking,
            person && agent == AgentState::Speaking,
            tools > 0,
            turn_hooks > 0,
            draining,
            on_exit > 0,
            on_enter > 0,
            closing,
        ];
        let name = applied.input.name();
        out.extend(PHASES.iter().zip(phases).filter(|(_, on)| *on).map(|(p, _)| (name, *p)));
    }
    out
}

/// For each input the case sent, the kinds it opened *directly*, by exact stamp.
///
/// A `say` does not open its speech here: that arrives from the driver's journal at the next
/// input which yields, and no timing rule can say which earlier input caused it. So this answers
/// one direction only, whether a row opens something it never named, which the survey audits.
fn opened_by_input(record: &Recording) -> BTreeMap<&'static str, BTreeSet<Kind>> {
    let mut out: BTreeMap<&'static str, BTreeSet<Kind>> = BTreeMap::new();
    for applied in record.applied() {
        let direct = record.opened().iter().filter(|o| o.at.seq == applied.at.seq).map(|o| o.kind);
        out.entry(applied.input.name()).or_default().extend(direct);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A task that wakes itself forever never lets the clock move: the run stops the session as
    /// a livelock instead of spinning with it.
    #[test]
    fn a_task_spinning_at_one_instant_is_a_livelock() {
        let livelock = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Policy::default()).await;
            tokio::spawn(async {
                loop {
                    tokio::task::yield_now().await;
                }
            });
            run.apply(&Input::Advance { ms: Millis(20) }).await;
            run.record.livelock().map(str::to_owned)
        });
        assert!(livelock.is_some(), "a spinning task was not reported");
    }

    /// A joined task names the product line that spawned it, not the spawner's: without
    /// `#[track_caller]` through `Spawner::spawn` every stuck task would point at the harness.
    #[test]
    fn a_task_names_the_line_that_spawned_it() {
        let sites = simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Policy::default()).await;
            for input in [Input::Close { drain: false }, Input::AwaitExit {}] {
                run.apply(&input).await;
            }
            run.join_tasks().await;
            run.record.tasks().iter().map(|t| t.site.file().to_owned()).collect::<Vec<_>>()
        });
        assert!(!sites.is_empty(), "the session spawned no task to check");
        for site in sites {
            assert!(site.contains("lk-agents-voice/src/"), "a task is placed at {site}");
        }
    }

    /// Every contract's probe reaches its prerequisite: the run passes, and the contract made at
    /// least one expectation on it. Without this a probe can rot into a case that runs fine and
    /// never touches the rule it was written for.
    #[test]
    fn every_probe_reaches_its_contract() {
        {
            for rule in crate::rules::RULES {
                let contract = &rule.contract;
                let Some(probe) = contract.probe else { continue };
                let outcome = run(probe());
                assert!(outcome.ok(), "{} probe failed: {:?}", contract.id, outcome.failures);
                let made = outcome.expectations.get(contract.id).copied().unwrap_or(0);
                assert!(made > 0, "{} probe never reached its prerequisite", contract.id);
            }
        }
    }
}
