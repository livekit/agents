use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechKind;
use lk_agents_voice::amd::{AmdObservation, AmdRunId};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{GateStep, SpeechStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Knob},
    record::{Handle, Observed, Recording},
};

// Spec §14 invariant 10: once the session reports
// `Detached`, the run observes nothing more, the gate neither holds authorization nor acts on a
// later turn, and replies are as interruptible as they were before the run attached.

pub struct GateCleansUp;

impl Rule for GateCleansUp {
    const CONTRACT: Contract = Contract {
        id: "gate_cleans_up",
        guarantee: "After Detached, AMD observes nothing, the gate neither holds nor acts, a \
                    later turn with nothing in its way reaches its model, and replies are as \
                    interruptible as before the run attached.",
        examples: &[clean, preemptive_adopted],
        counterexamples: &[
            observed_after,
            still_gated,
            interruptions_kept_forced,
            never_answered,
            adopted_in_the_trace_only,
        ],
        probe: Some(a_turn_after_completion),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        let attachments = amd.attachments();
        for (index, attachment) in attachments.iter().enumerate() {
            let Some(detached) = attachment.detached else { continue };
            let after = amd.observed(attachment.run).into_iter().filter(|(at, _)| *at > detached);
            let late: Vec<_> = after.map(|(at, o)| (at.at_ms, o.clone())).collect();
            check.expect_with(Subject::Session, late.is_empty(), || {
                format!("run {} observed {late:?} after it detached", attachment.run)
            });
            let next = attachments.get(index + 1).map_or(Observed::END, |a| a.attached);
            // No gate and no hold until the next attach (spec §14 invariant 10, owned by "a DST
            // rule after Detached: no gate, no hold"). Whether a later turn is answered depends
            // on the session's ordinary rules (hooks, interrupts, faults), not on the gate.
            let gated: Vec<_> = r
                .journal()
                .iter()
                .filter(|(at, _)| *at > detached && *at < next)
                .filter(|(_, step)| match step {
                    Step::Speech(SpeechStep::AuthorizationHeld(_) | SpeechStep::HeadHeld { .. }) => {
                        true
                    }
                    // A late answer of the finished run is dropped, which is cleanup working, and
                    // a refused attach installs no gate.
                    Step::Gate(GateStep::AnswerDropped { .. } | GateStep::AttachRefused { .. }) => {
                        false
                    }
                    Step::Gate(_) => true,
                    _ => false,
                })
                .map(|(at, step)| (at.at_ms, step.clone()))
                .collect();
            check.expect_with(Subject::Session, gated.is_empty(), || {
                format!("after run {} detached the gate still acted: {gated:?}", attachment.run)
            });
            answers_later_turns(r, attachment.run, detached, next, check);
            let reconfigured = r.applied().iter().any(|a| {
                matches!(&a.input, Input::Configure { knob: Knob::Interruptions, .. } | Input::Override { knob: Knob::Interruptions, .. } | Input::SetAllowInterruptions { .. })
            });
            if reconfigured {
                continue;
            }
            let replies = |from: Observed, to: Observed| -> Vec<bool> {
                r.host_view()
                    .speeches()
                    .into_iter()
                    .filter(|s| s.kind() == Some(SpeechKind::TurnReply))
                    .filter(|s| s.created_at().is_some_and(|at| at > from && at < to))
                    .filter_map(|s| r.handles().get(&s.id).map(|h: &Handle| h.allow_interruptions))
                    .collect()
            };
            let before = replies(Observed::at(0, 0), attachment.attached);
            let Some(setting) = before.last().copied() else { continue };
            for allowed in replies(detached, next) {
                check.expect(
                    Subject::Session,
                    allowed == setting,
                    format!(
                        "a reply after run {} detached is interruptible={allowed}; before it \
                         attached replies were {setting}",
                        attachment.run
                    ),
                );
            }
        }
    }
}

/// How long an ungated turn may take to reach its model, past its customer hook.
const BOUND_MS: u64 = 2_000;

/// A turn committed for a reply after `detached` and before the next attach reaches its model
/// within a bound: no decision holds it. Only an eligible turn with nothing in its way is
/// judged: past its customer hook, not stopped or dropped by it or behind a speech it could not
/// interrupt, with no newer turn, fault, refusal, handoff, drain, model swap or close before the
/// bound.
fn answers_later_turns(
    r: &Recording,
    run: u64,
    detached: Observed,
    next: Observed,
    check: &mut Check,
) {
    let amd = r.amd_view();
    let events = r.host_view().events();
    let commits: Vec<(Observed, u64, String)> = events
        .iter()
        .filter_map(|(at, e)| match e {
            SessionEvent::UserTurnCompleted(done) => {
                Some((*at, done.turn.0, done.new_transcript.clone()))
            }
            _ => None,
        })
        .collect();
    for (at, turn, words) in &commits {
        if *at < detached || *at > next || !amd.replies(*at) || amd.skips_reply(*turn) {
            continue;
        }
        let Some(hook) = amd.hook_returned_for(*turn, *at) else { continue };
        let due = (*at).max(hook).at_ms + BOUND_MS;
        if r.now().at_ms < due || r.close_begun().is_some_and(|c| c.at_ms <= due) {
            continue;
        }
        let window = |seen: Observed| seen > *at && seen.at_ms <= due;
        let newer = commits.iter().any(|(seen, t, _)| *t != *turn && window(*seen));
        let faulted = r.faults().iter().any(|(seen, _, _)| window(*seen));
        let refused = r.journal().iter().any(|(seen, step)| {
            window(*seen) && matches!(step, Step::Speech(SpeechStep::ScheduleRejected { .. }))
        });
        let interfered = r.applied().iter().filter(|a| a.accepted && window(a.at)).any(|a| {
            matches!(
                a.input,
                Input::Close { .. }
                    | Input::Drain { .. }
                    | Input::Handoff { .. }
                    | Input::UpdateModels { .. }
                    | Input::VoiceTask { .. }
                    | Input::AmdAttach { .. }
                    | Input::AmdEnter { .. }
            )
        });
        // Customer code running at any point until the bound (an agent's `on_enter`, a
        // handoff's hooks, an older turn's): the turn may wait on it.
        let hooked = hook_running(r, *at, due);
        if newer
            || faulted
            || refused
            || interfered
            || hooked
            || amd.stopped_turn(*turn)
            || amd.dropped_words(words, *at)
        {
            continue;
        }
        let item = events.iter().find_map(|(_, e)| match e {
            SessionEvent::ConversationItemAdded {
                item,
                source: lk_agents_voice::agent::Source::UserTurn { turn: t },
                ..
            } if t.0 == *turn => item.as_message().map(|m| m.id.clone()),
            _ => None,
        });
        let Some(item) = item else { continue };
        // A reply made for it, or a request answering it: what happens to the reply after that
        // is the session's ordinary rules (an interrupt, the store), not the gate.
        let replied = events.iter().any(|(seen, e)| {
            window(*seen)
                && matches!(e, SessionEvent::SpeechCreated { kind: SpeechKind::TurnReply, .. })
        }) || adopted(r, *at, due, words);
        let answered = replied
            || amd.requests().into_iter().any(|q| {
                q.opened.at_ms <= due
                    && q.last_user().is_some_and(|m| {
                        m.id == item || (amd.preemptive(&q) && m.text == *words)
                    })
            });
        check.expect_with(Subject::Session, answered, || {
            format!(
                "turn {turn} committed at {} ms, after run {run} detached, was not answered by \
                 {due} ms",
                at.at_ms
            )
        });
    }
}

/// Whether the turn the host heard commit at `at` adopted the preemptive generation by `due`
/// ms: that is the reply made for it (proposal, "Preemptive generation may run while
/// authorization is held"). The adoption is journalled in the commit's own transition, after its
/// committed end of turn and before the host hears it. It counts only for a preemptive speech
/// whose own request, opened before the adoption, answered the turn's `words`: a trace step alone
/// is the session's report, not evidence.
fn adopted(r: &Recording, at: Observed, due: u64, words: &str) -> bool {
    use lk_agents_voice::agent_session::machine::activity::scheduling::PreemptiveVerdict;
    use lk_agents_voice::trace::TurnStep;
    let committed =
        |step: &Step| matches!(step, Step::Turn(TurnStep::EndOfTurn { committed: true, .. }));
    let steps = r.journal();
    let Some(ended) = steps.iter().rposition(|(seen, step)| *seen < at && committed(step)) else {
        return false;
    };
    let amd = r.amd_view();
    let requests = amd.requests();
    let original = |speech: u64, reused: Observed| {
        requests.iter().any(|q| {
            q.speech == Some(speech)
                && q.opened <= reused
                && amd.preemptive(q)
                && q.last_user().is_some_and(|m| m.text == words)
        })
    };
    steps[ended + 1..]
        .iter()
        .take_while(|(seen, step)| seen.at_ms <= due && !committed(step))
        .any(|(seen, step)| match step {
            Step::Turn(TurnStep::PreemptiveReconciled {
                speech,
                verdict: PreemptiveVerdict::Reuse,
            }) => original(speech.0, *seen),
            _ => false,
        })
}

/// Whether a hook of any kind ran at some point between `from` and `to` ms.
fn hook_running(r: &Recording, from: Observed, to: u64) -> bool {
    use lk_agents_voice::trace::HandoffStep;
    let mut issued = std::collections::BTreeMap::new();
    let mut finished = std::collections::BTreeMap::new();
    for (at, step) in r.journal() {
        match step {
            Step::Handoff(HandoffStep::HookIssued { hook, .. }) => {
                issued.entry(*hook).or_insert(*at);
            }
            Step::Handoff(HandoffStep::HookFinished { hook, .. }) => {
                finished.entry(*hook).or_insert(*at);
            }
            _ => {}
        }
    }
    issued.iter().any(|(hook, start)| {
        start.at_ms <= to && finished.get(hook).is_none_or(|end: &Observed| *end > from)
    })
}

fn attached() -> Authored {
    Authored::new()
        .sent(Input::AmdAttach {})
        .sent(Input::AmdDetach { category: AmdVerdict::Allow, forbid: false })
        .amd(1, AmdObservation::Detached { run: AmdRunId(1) })
}

fn clean() -> Recording {
    attached().user_turn(None, 1, "hello").request(10, 1, "hello", &[], &[]).later(3_000).done()
}

fn observed_after() -> Recording {
    attached()
        .amd(1, AmdObservation::AgentBusy { busy: false, at: lk_agents_abi::Ts::ZERO })
        .done()
}

fn still_gated() -> Recording {
    attached()
        .user_turn(None, 1, "hello")
        .journal(Step::Speech(SpeechStep::AuthorizationHeld(lk_agents_voice::amd::Hold::Undecided {
            turn: lk_agents_voice::audio_recognition::TurnId(1),
        })))
        .later(3_000)
        .done()
}

/// A turn after the detach adopts its preemptive generation, made for its words: the turn had
/// its reply.
fn preemptive_adopted() -> Recording {
    use lk_agents_voice::agent_session::machine::activity::scheduling::{
        PreemptiveVerdict, SpeechId,
    };
    use lk_agents_voice::trace::TurnStep;
    attached()
        .speech_created(7, SpeechKind::Preemptive)
        .request(7, 1, "hello", &[], &[])
        .user_turn(None, 1, "hello")
        .journal(Step::Turn(TurnStep::PreemptiveReconciled {
            speech: SpeechId(7),
            verdict: PreemptiveVerdict::Reuse,
        }))
        .later(3_000)
        .done()
}

/// The session reports an adoption, and no preemptive request ever answered the turn: its own
/// report proves no reply.
fn adopted_in_the_trace_only() -> Recording {
    use lk_agents_voice::agent_session::machine::activity::scheduling::{
        PreemptiveVerdict, SpeechId,
    };
    use lk_agents_voice::trace::TurnStep;
    attached()
        .user_turn(None, 1, "hello")
        .journal(Step::Turn(TurnStep::PreemptiveReconciled {
            speech: SpeechId(7),
            verdict: PreemptiveVerdict::Reuse,
        }))
        .later(3_000)
        .done()
}

/// A turn after the detach, with nothing in its way, never reaches its model.
fn never_answered() -> Recording {
    attached().user_turn(None, 1, "hello").later(3_000).done()
}

fn interruptions_kept_forced() -> Recording {
    let reply = |interruptible| Handle { allow_interruptions: interruptible, ..Handle::live() };
    Authored::new()
        .speech_created(5, SpeechKind::TurnReply)
        .speech(5, reply(false))
        .sent(Input::AmdAttach {})
        .sent(Input::AmdDetach { category: AmdVerdict::Allow, forbid: false })
        .amd(1, AmdObservation::Detached { run: AmdRunId(1) })
        .speech_created(6, SpeechKind::TurnReply)
        .speech(6, reply(true))
        .done()
}

/// AMD completes, and the next turn is answered with no decision.
fn a_turn_after_completion() -> crate::drive::Case {
    use crate::fakes::amd::probe::*;
    let mut inputs = attached(None);
    inputs.extend(says(0, "hello"));
    inputs.push(decide(0, AmdVerdict::Allow));
    inputs.extend(answered(3));
    inputs.push(advance(3_000));
    inputs.push(Input::AmdDetach { category: AmdVerdict::Allow, forbid: false });
    inputs.extend(says(1, "who is this"));
    inputs.extend(answered(6));
    inputs.push(advance(3_000));
    call(inputs)
}
