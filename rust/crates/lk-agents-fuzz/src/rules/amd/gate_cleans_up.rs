use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechKind;
use lk_agents_voice::amd::{AmdObservation, AmdRunId};
use lk_agents_voice::events::SessionEvent;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Knob},
    record::{Handle, Observed, Recording},
};

// Spec §14 invariant 10, the AMD proposal "Detach" steps 2 and 5: once the session reports
// `Detached`, the run observes nothing more, a later turn is answered with no decision (no gate,
// no hold), and replies are as interruptible as they were before the run attached.

pub struct GateCleansUp;

/// How long an ungated turn may take to reach its model.
const BOUND_MS: u64 = 2_000;

impl Rule for GateCleansUp {
    const CONTRACT: Contract = Contract {
        id: "gate_cleans_up",
        guarantee: "After Detached, AMD observes nothing, the next turn replies with no \
                    decision, and replies are as interruptible as before the run attached.",
        examples: &[clean],
        counterexamples: &[observed_after, still_gated, interruptions_kept_forced],
        probe: None,
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
            let events = r.host_view().events();
            for (at, event) in &events {
                let SessionEvent::UserTurnCompleted(done) = event else { continue };
                if *at < detached || *at > next {
                    continue;
                }
                let due = at.at_ms + BOUND_MS;
                if r.now().at_ms < due || r.close_begun().is_some_and(|c| c.at_ms <= due) {
                    continue;
                }
                let item = events.iter().find_map(|(_, e)| match e {
                    SessionEvent::ConversationItemAdded {
                        item,
                        source: lk_agents_voice::agent::Source::UserTurn { turn },
                        ..
                    } if *turn == done.turn => item.as_message().map(|m| m.id.clone()),
                    _ => None,
                });
                let Some(item) = item else { continue };
                let answered = amd.requests().into_iter().any(|q| {
                    q.opened > *at && q.opened.at_ms <= due && q.last_user().is_some_and(|m| m.id == item)
                });
                check.expect_with(Subject::Session, answered, || {
                    format!(
                        "turn {:?} committed at {} ms, after run {} detached, was not answered \
                         by {due} ms",
                        done.turn, at.at_ms, attachment.run
                    )
                });
            }
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
