//! Releasing what a run owes, then reporting what it found.
//!
//! Beside the session it drives and not in `search`: a probe or a regression test needs exactly
//! this and nothing of the campaign, so changing how the search picks inputs cannot change how a
//! case ends. The rules run once, over the finished recording; the only check between inputs is
//! that the run is still moving.
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
        // a start without an end. The detector is speaking or not, so its last word stands: an
        // end with no start before it does not cancel the next start.
        for nth in self.record.live().of(Kind::Vad) {
            let still_speaking = self
                .record
                .applied()
                .iter()
                .filter(|a| a.accepted)
                .filter_map(|a| match &a.input {
                    Input::VadStart { vad, .. } if vad.nth == nth => Some(true),
                    Input::VadEnd { vad, .. } if vad.nth == nth => Some(false),
                    _ => None,
                })
                .last()
                .unwrap_or(false);
            if still_speaking {
                out.push(Input::VadEnd { vad: Ref::new(nth), silence_ms: Millis(500) });
            }
        }
        // The realtime server's VAD the same: a person it heard start is heard to stop, or the
        // driver holds the agent's voice for the end of their turn forever.
        if let Some(socket) = self.record.live().of(Kind::Realtime).first().copied() {
            let still_speaking = self
                .record
                .applied()
                .iter()
                .filter(|a| a.accepted)
                // This socket's: one a fallback adapter swapped away from took its person along.
                .filter_map(|a| match &a.input {
                    Input::RealtimeEvent { socket: s, event: RtEvent::SpeechStarted }
                        if s.nth == socket =>
                    {
                        Some(true)
                    }
                    Input::RealtimeEvent {
                        socket: s,
                        event: RtEvent::SpeechStopped | RtEvent::Disconnected,
                    } if s.nth == socket => Some(false),
                    _ => None,
                })
                .last()
                .unwrap_or(false);
            if still_speaking {
                out.push(Input::RealtimeEvent {
                    socket: Ref::new(socket),
                    event: RtEvent::SpeechStopped,
                });
            }
        }
        // A recogniser swapped out mid-utterance took that utterance with it, and the driver still
        // hears the person speaking: the new recogniser hears them out, an utterance the next
        // round ends.
        if self.recogniser_left_speaking()
            && self.record.live().of(Kind::Utterance).is_empty()
            && let Some(stt) = self.record.live().of(Kind::Stt).first().copied()
        {
            out.push(Input::SttStart { stt: Ref::new(stt) });
        }
        // A running tool is answered; one that detached is ended by the host's final update.
        for nth in self.record.live().of(Kind::Tool) {
            out.push(if self.tool_detached(nth) {
                Input::ToolUpdate { tool: Ref::new(nth), progress: None, end: ToolAnswer::Return }
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

    /// Whether the last round granted a floor hold the case asked for. The release phase never
    /// asks for one, and holds are granted one at a time, so such a round works down the case's
    /// own queue of `WaitForIdleAndHold` asks, a finite backlog, rather than producing new work.
    /// A grant beyond the asks is not the backlog.
    fn granted_queued_hold(&self, grants_before: usize) -> bool {
        let grants = self.hold_grants();
        let asks = self
            .record
            .sent(|i| matches!(i, Input::WaitForIdleAndHold {}).then_some(()))
            .filter(|(_, (), accepted)| *accepted)
            .count();
        grants > grants_before && grants <= asks
    }

    fn hold_grants(&self) -> usize {
        self.record
            .journal_where(|step| match step {
                Step::Speech(SpeechStep::IdleAnswered { hold: Some(_), .. }) => Some(()),
                _ => None,
            })
            .count()
    }

    /// Speech steps spoken and speeches created so far: a round that spoke a step and created no
    /// speech works down the live speeches' tool steps, a backlog `max_tool_steps` bounds, rather
    /// than producing new work.
    fn speech_progress(&self) -> (usize, usize) {
        let count = |want: fn(&Step) -> bool| {
            self.record.journal_where(|step| want(step).then_some(())).count()
        };
        (
            count(|s| matches!(s, Step::Speech(SpeechStep::StepSpoken { .. }))),
            count(|s| matches!(s, Step::Emitted(SessionEvent::SpeechCreated { .. }))),
        )
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
        let playing = now + self.room_audio_queued().as_millis() as u64;
        // A voice still owing its first audio answers on its own clock.
        let voice = now + self.registry.voice_due_in().map_or(0, |d| d.as_millis() as u64);
        // The avatar plays what it holds in real time before it reports it played.
        let avatar = now + self.avatar_due_in().map_or(0, |d| d.as_millis() as u64);
        let furthest = self
            .deadlines()
            .map(|(_, at)| at)
            .max()
            .unwrap_or(now)
            .max(playing)
            .max(voice)
            .max(avatar);
        Input::Advance { ms: Millis((furthest.saturating_sub(now) + 1).min(RELEASE_CAP_MS)) }
    }

    /// Whether the driver armed a deadline the clock has not reached: a speech queued behind
    /// `min_consecutive_speech_delay`, or a retry waiting out its backoff, waits on the driver's
    /// own clock, not on the case, so a round that leaves one ahead is not quiet. The recorder's
    /// flush re-arms for as long as the session lives and owes nobody anything.
    fn deadline_ahead(&self) -> bool {
        let now = self.record.now().at_ms;
        self.deadlines().any(|(periodic, at)| !periodic && at > now)
            || !self.room_audio_queued().is_zero()
            || self.registry.voice_due_in().is_some()
            || self.avatar_due_in().is_some()
    }

    /// The avatar's playout still to come, or the agent's sink waiting out its give-up timer.
    fn avatar_due_in(&self) -> Option<Duration> {
        let now = self.time.now();
        let timer = self
            .avatar_sink
            .as_ref()
            .and_then(|sink| sink.wake_at())
            .map(|at| at.saturating_duration_since(now));
        let playout = self.avatar.as_ref().and_then(|worker| worker.due_in());
        playout.into_iter().chain(timer).max()
    }

    /// The agent's voice still queued in the room's speaker or the console's: either plays on the
    /// clock and owes the driver its playout, so a round that leaves audio there is not quiet.
    fn room_audio_queued(&self) -> Duration {
        use lk_agents_test_support::rtc::AudioSource as _;
        let sources = self.room.local_participant().sources();
        let room = sources
            .iter()
            .map(|source| source.queued_duration().expect("fake source queue position"))
            .max()
            .unwrap_or_default();
        self.console.as_ref().map_or(room, |cli| room.max(cli.queued()))
    }

    /// Every deadline the driver journalled, in milliseconds, and whether it is the recorder's
    /// periodic flush: its timers, and the retries it waits out on its own.
    fn deadlines(&self) -> impl Iterator<Item = (bool, u64)> + '_ {
        use lk_agents_voice::{agent_session::effects::Timer, trace::TimerName};
        self.record
            .journal_where(|step| match step {
                Step::Core(CoreStep::TimerArmed { timer, at }) => {
                    Some((*timer == TimerName::Session(Timer::RecorderFlush), *at))
                }
                Step::Generation(GenerationStep::RetryArmed { at, .. }) => Some((false, *at)),
                // A playout taking a missing finish as lost once its margin passes.
                Step::Playout(PlayoutStep::FinishDue { at, .. }) => Some((false, *at)),
                _ => None,
            })
            .map(|(_, (periodic, at))| {
                (periodic, at.saturating_duration_since(Ts::ZERO).as_millis() as u64)
            })
    }

    /// Whether the person's speech the driver last heard of began on the recogniser's word and
    /// has not ended: the utterances since went with a recogniser that was swapped or failed
    /// under them. Once only: a recogniser the driver no longer takes the end of speech from
    /// (another turn detection) cannot end it, and asking again would never go quiet.
    fn recogniser_left_speaking(&self) -> bool {
        let Some((began, true)) = self
            .record
            .journal_where(|step| match step {
                Step::Emitted(SessionEvent::StartOfSpeech(ev)) => Some(!ev.from_vad),
                Step::Emitted(SessionEvent::EndOfSpeech(_)) => Some(false),
                _ => None,
            })
            .last()
        else {
            return false;
        };
        // A whole utterance since, started and ended, that did not end the speech: the driver
        // does not take the recogniser's word for it.
        let accepted = self.record.applied().iter().filter(|a| a.accepted && a.at > began);
        let restarted = accepted.clone().find(|a| matches!(a.input, Input::SttStart { .. }));
        !restarted.is_some_and(|start| {
            accepted.clone().any(|a| a.at > start.at && matches!(a.input, Input::SttEnd { .. }))
        })
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
/// How a run ended, with its evidence. A run stopped at a finite bound is `Budget` unless the
/// trail shows the cycle it was spinning in: a step limit alone proves no livelock, and a wait
/// that ran out proves no deadlock unless nothing could have moved it.
#[derive(Debug, Default, Clone, PartialEq)]
pub enum Ending {
    /// Ran to its close; every contract held.
    #[default]
    Completed,
    /// Ran to its close; a contract failed (`failures`).
    ContractViolated,
    /// Stopped at the step bound while dispatching the same cycle of steps, the whole trail.
    Livelock { cycle: Vec<&'static str> },
    /// A task the run awaited could not finish: nothing pending, no wake, no task runnable.
    Deadlock { evidence: String },
    /// Stopped at a finite bound (steps at one instant, release rounds) with no proof of either.
    Budget { bound: String, trail: String },
}

#[derive(Debug, Default)]
pub struct Outcome {
    pub ending: Ending,
    /// The run's dispatch order, folded (`Run::dispatch_order`).
    pub dispatch_order: u64,
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
    /// The most steps of the run's loop at one virtual instant.
    pub busiest_instant: u32,
}

impl Outcome {
    pub fn ok(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Run one case to completion in its own simulation.
///
/// One case, one runtime: the seeded scheduler is only reproducible from its start, so two cases
/// sharing a runtime would each begin at a different point in the stream and neither would
/// reproduce on its own.
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
    let mut backlog = 0;
    while quiet < QUIET_ROUNDS {
        // The backlogs are bounded too: a driver that never ran out of steps or audio is a finding.
        if rounds >= MAX_RELEASE_ROUNDS || backlog >= 10 * MAX_RELEASE_ROUNDS {
            let never_quiet = Failure {
                contract: "session_settles".into(),
                subject: Subject::Session,
                detail: format!(
                    "still producing work to release after {rounds} rounds ({backlog} more \
                     working down steps or audio) and {} inputs: the run never went quiet",
                    outcome.case.inputs.len()
                ),
            };
            outcome.stopped_early = true;
            // What the run reached still counts, and its rules still get their say.
            record_outcome(&run, &mut outcome);
            outcome.failures.push(never_quiet);
            outcome.ending = ending(&run, &outcome.failures);
            return outcome;
        }
        let grants_before = run.hold_grants();
        let (spoken_before, created_before) = run.speech_progress();
        // Audio already handed to the room plays out in real time, at most a release's advance
        // a round: a finite backlog too.
        let draining = !run.room_audio_queued().is_zero();
        // The avatar worker's playout reports are releases the run makes as they come due.
        let avatar_before = run.avatar.as_ref().map(|w| w.progress());
        let mut offered = run.releases();
        let releases = offered.len();
        // Advance regardless, so an armed deadline with nothing else to do still fires.
        offered.push(run.release_advance());
        for input in offered {
            outcome.case.inputs.push(input.clone());
            run.apply(&input).await;
            if let Some(found) = guard(&run, &mut outcome) {
                return found;
            }
        }
        // Once the driver has exited, a release it refuses is one nothing will ever take: what is
        // left of a dead session is `nothing_left_live`'s to judge, not work still to do.
        let applied = run.record.applied();
        let taken =
            applied[applied.len() - 1 - releases..applied.len() - 1].iter().any(|a| a.accepted);
        let avatar_reported = run.avatar.as_ref().map(|w| w.progress()) != avatar_before;
        if (releases == 0 && !run.deadline_ahead() && !avatar_reported)
            || (run.record.driver_finished() && !taken)
        {
            quiet += 1;
        } else {
            quiet = 0;
        }
        let (spoken, created) = run.speech_progress();
        let stepped = (spoken > spoken_before || draining) && created == created_before;
        if run.granted_queued_hold(grants_before) {
            // The case's own asks, as many as it made.
        } else if stepped {
            backlog += 1;
        } else {
            rounds += 1;
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
    if std::env::var_os("VOICE_JOURNAL").is_some() {
        dump_journal(&run.record);
    }
    let (failures, expectations) = check_all(&run.record);
    outcome.failures = failures;
    outcome.expectations = expectations;
    outcome.reached = run.record.journal_names().into_iter().collect();
    let names: Vec<String> = run.record.journal().iter().map(|(_, s)| step_name(s)).collect();
    outcome.edges = names.windows(2).map(|w| (w[0].clone(), w[1].clone())).collect();
    outcome.opened = opened_by_input(&run.record);
    outcome.cells = cells_by_input(&run.record);
    outcome.busiest_instant = run.busiest_instant;
    outcome.dispatch_order = run.dispatch_order();
    outcome.ending = ending(run, &outcome.failures);
}

/// How the run ended, from what it recorded: the bound it stopped at and the trail's cycle, the
/// deadlock evidence, or its findings.
fn ending(run: &Run, failures: &[Failure]) -> Ending {
    let trail = run.queue.trail();
    let stopped = run.record.livelock().map(str::to_owned).or_else(|| {
        failures.iter().find(|f| f.contract == "session_settles").map(|f| f.detail.clone())
    });
    if let Some(bound) = stopped {
        let trail = trail.lock().unwrap();
        return match trail.cycle() {
            Some(cycle) if run.record.livelock().is_some() => Ending::Livelock { cycle },
            _ => Ending::Budget { bound, trail: trail.dump() },
        };
    }
    if let Some(evidence) = &run.deadlock {
        return Ending::Deadlock { evidence: evidence.clone() };
    }
    if failures.is_empty() { Ending::Completed } else { Ending::ContractViolated }
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

    /// Runs `body` on a fresh run, then closes it and ends it as `run()` does: its outcome.
    fn ended(body: impl FnOnce(&Run) + 'static) -> Outcome {
        simulate(async {
            let mut run = Run::start(&Setup::default(), 0, Policy::default()).await;
            body(&run);
            let mut outcome = Outcome::default();
            for input in [Input::Advance { ms: Millis(20) }, Input::Close { drain: false }] {
                run.apply(&input).await;
                if let Some(stopped) = guard(&run, &mut outcome) {
                    return stopped;
                }
            }
            run.apply(&Input::AwaitExit {}).await;
            run.join_tasks().await;
            run.collect();
            record_outcome(&run, &mut outcome);
            outcome
        })
    }

    /// A task that wakes itself forever never lets the clock move: the run stops the session at
    /// its step bound, and the trail shows the cycle it spun in, which is the livelock's proof.
    #[test]
    fn a_task_spinning_at_one_instant_is_a_livelock() {
        let outcome = ended(|_| {
            tokio::spawn(async {
                loop {
                    tokio::task::yield_now().await;
                }
            });
        });
        assert!(outcome.stopped_early, "a spinning task was not stopped");
        assert!(
            matches!(&outcome.ending, Ending::Livelock { cycle } if cycle == &["Tasks"]),
            "{:?}",
            outcome.ending
        );
    }

    /// A quiet session that closes ends completed; one whose task panicked ends with the
    /// contract it broke; one whose task waits on nothing that can ever happen is a deadlock,
    /// with what was (not) enabled as evidence.
    #[test]
    fn each_ending_is_told_apart() {
        assert_eq!(ended(|_| {}).ending, Ending::Completed);
        let violated = ended(|run| {
            run.tasks.spawn(Box::pin(async { panic!("boom") }));
        });
        assert_eq!(violated.ending, Ending::ContractViolated, "{:?}", violated.failures);
        let deadlocked = ended(|run| {
            run.tasks.spawn(Box::pin(std::future::pending()));
        });
        assert!(
            matches!(&deadlocked.ending, Ending::Deadlock { evidence } if evidence.contains("nothing pending")),
            "{:?}",
            deadlocked.ending
        );
    }

    /// A joined task names the line that spawned it, the product's or the executor effect's, not
    /// the spawner's: without `#[track_caller]` through `Spawner::spawn` every stuck task would
    /// point at the harness's spawner.
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
            let placed = site.contains("lk-agents-voice/src/")
                || site.contains("lk-agents-runtime/src/room_io/")
                || site.ends_with("lk-agents-runtime/src/voice/mod.rs")
                || site.ends_with("lk-agents-runtime/src/voice/effects.rs")
                || site.ends_with("lk-agents-runtime/src/voice/console.rs")
                || site.ends_with("lk-agents-runtime/src/voice/legs.rs")
                || site.ends_with("lk-agents-runtime/src/voice/background_audio.rs")
                || site.ends_with("lk-agents-runtime/src/voice/audio.rs")
                || site.ends_with("lk-agents-runtime/src/voice/amd.rs")
                || site.ends_with("lk-agents-runtime/src/avatar_runner.rs")
                || site.ends_with("lk-agents-runtime/src/avatar/receiver.rs")
                || site.ends_with("lk-agents-runtime/src/avatar/output.rs");
            assert!(placed, "a task is placed at {site}");
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

/// `VOICE_JOURNAL=1`: every journal step, host event and applied input of the run, in sequence
/// order.
fn dump_journal(record: &crate::record::Recording) {
    let mut lines: Vec<(u64, String)> = record
        .journal()
        .iter()
        .map(|(at, step)| (at.seq, format!("{:>6} {:>7}ms   {step:?}", at.seq, at.at_ms)))
        .collect();
    lines.extend(
        record
            .host()
            .iter()
            .map(|(at, h)| (at.seq, format!("{:>6} {:>7}ms H {h:?}", at.seq, at.at_ms))),
    );
    lines.extend(
        record
            .avatar()
            .iter()
            .map(|(at, step)| (at.seq, format!("{:>6} {:>7}ms A {step:?}", at.seq, at.at_ms))),
    );
    lines.extend(record.applied().iter().map(|a| {
        let mark = if a.accepted { ">" } else { "x" };
        (a.at.seq, format!("{:>6} {:>7}ms {mark} {:?}", a.at.seq, a.at.at_ms, a.input))
    }));
    lines.sort_by_key(|(seq, _)| *seq);
    for (_, line) in lines {
        eprintln!("{line}");
    }
}
