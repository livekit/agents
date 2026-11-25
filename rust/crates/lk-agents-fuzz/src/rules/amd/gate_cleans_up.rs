use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechKind;
use lk_agents_voice::amd::{AmdObservation, AmdRunId};
use lk_agents_voice::trace::{GateStep, SpeechStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Knob},
    record::{Handle, Observed, Recording},
};

// Spec §14 invariant 10, the AMD proposal "Detach" steps 2 and 5: once the session reports
// `Detached`, the run observes nothing more, the gate neither holds authorization nor acts on a
// later turn, and replies are as interruptible as they were before the run attached.

pub struct GateCleansUp;

impl Rule for GateCleansUp {
    const CONTRACT: Contract = Contract {
        id: "gate_cleans_up",
        guarantee: "After Detached, AMD observes nothing, the gate neither holds nor acts, and \
                    replies are as interruptible as before the run attached.",
        examples: &[clean],
        counterexamples: &[observed_after, still_gated, interruptions_kept_forced],
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
                    // A late answer of the finished run is dropped, which is cleanup working.
                    Step::Gate(GateStep::AnswerDropped { .. }) => false,
                    Step::Gate(_) => true,
                    _ => false,
                })
                .map(|(at, step)| (at.at_ms, step.clone()))
                .collect();
            check.expect_with(Subject::Session, gated.is_empty(), || {
                format!("after run {} detached the gate still acted: {gated:?}", attachment.run)
            });
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
