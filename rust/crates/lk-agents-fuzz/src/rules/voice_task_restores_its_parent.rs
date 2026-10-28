//! A `VoiceTask` gives the floor back to the agent it took it from, exactly once.
//!
//! The parent's tool awaits the task in memory; the parent itself is only suspended. Each task
//! that entered ends once, innermost first (an outer task's end ends the ones it started first),
//! and the agent installed next is the parent it suspended, read off the install itself rather
//! than the end's own claim. The close ends every task before anything else: none is open once a
//! close began, and none enters after it.
use lk_agents_voice::agent_activity::voice_task::{TaskEnd, VoiceTaskId};
use lk_agents_voice::trace::{HandoffStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};

pub struct VoiceTaskRestoresItsParent;

impl Rule for VoiceTaskRestoresItsParent {
    const CONTRACT: Contract = Contract {
        id: "voice_task_restores_its_parent",
        guarantee: "Every VoiceTask ends exactly once, innermost first, and the agent installed \
                    next is the parent it suspended; none is open once a close began.",
        examples: &[completed, nested],
        counterexamples: &[ended_twice, outer_first, wrong_agent, not_restored, open_at_close],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // The driver's own close: a task asked for before it reached the driver still enters.
        let close = r.internal_view().close_requested().map(|(at, _)| at);
        // The tasks open, innermost last, and the agent the next install must name.
        let mut open: Vec<(VoiceTaskId, String)> = Vec::new();
        let mut install: Option<(String, u64)> = None;
        for (at, step) in r.internal_view().steps() {
            let Step::Handoff(step) = step else { continue };
            // A task step owes an install before the next one.
            if let (
                HandoffStep::TaskEntered { .. } | HandoffStep::TaskEnded { .. },
                Some((want, since)),
            ) = (step, &install)
            {
                check.expect(
                    Subject::Session,
                    false,
                    format!("nothing installed {want} after the task step at seq {since}"),
                );
                install = None;
            }
            match step {
                HandoffStep::TaskEntered { task, parent, agent, .. } => {
                    check.expect(
                        Subject::Session,
                        close.is_none_or(|c| *at < c),
                        format!("{task:?} entered at seq {} after the close began", at.seq),
                    );
                    open.push((*task, parent.clone()));
                    install = Some((agent.clone(), at.seq));
                }
                HandoffStep::TaskEnded { task, restored, .. } => {
                    let top = open.pop();
                    let innermost = top.as_ref().is_some_and(|(t, _)| t == task);
                    check.expect(
                        Subject::Session,
                        innermost,
                        format!(
                            "{task:?} ended at seq {} but the innermost open task was {:?}",
                            at.seq,
                            top.as_ref().map(|(t, _)| *t)
                        ),
                    );
                    if let Some((_, parent)) = top.filter(|_| innermost) {
                        check.expect(
                            Subject::Session,
                            *restored == parent,
                            format!("{task:?} restored {restored}, not its parent {parent}"),
                        );
                        install = Some((parent, at.seq));
                    }
                }
                HandoffStep::AgentUpdated { agent, .. } => {
                    if let Some((want, since)) = install.take() {
                        check.expect(
                            Subject::Session,
                            *agent == want,
                            format!(
                                "the agent installed at seq {} for the task step at seq {since} \
                                 was {agent}, not {want}",
                                at.seq
                            ),
                        );
                    }
                }
                _ => {}
            }
        }
        if let Some((want, since)) = install {
            check.expect(
                Subject::Session,
                false,
                format!("nothing installed {want} after the task step at seq {since}"),
            );
        }
        if let Some(close) = close {
            check.expect(
                Subject::Session,
                open.is_empty(),
                format!(
                    "the close began at seq {} with tasks still open: {:?}",
                    close.seq,
                    open.iter().map(|(t, _)| *t).collect::<Vec<_>>()
                ),
            );
        }
    }
}

fn entered(task: u64, parent: &str, agent: &str) -> Step {
    Step::Handoff(HandoffStep::TaskEntered {
        task: VoiceTaskId(task),
        parent: parent.into(),
        agent: agent.into(),
        speech: lk_agents_voice::agent_activity::scheduling::SpeechId(1),
    })
}

fn ended(task: u64, restored: &str) -> Step {
    Step::Handoff(HandoffStep::TaskEnded {
        task: VoiceTaskId(task),
        how: TaskEnd::Completed,
        restored: restored.into(),
    })
}

fn installed(agent: &str) -> Step {
    Step::Handoff(HandoffStep::AgentUpdated {
        agent: agent.into(),
        tools: 0,
        chat_ctx: false,
        reopened: "",
        legs: Default::default(),
    })
}

/// The task's agent installed, then its parent once it completed.
fn completed() -> Recording {
    Authored::new()
        .journal(entered(1, "desk", "email"))
        .journal(installed("email"))
        .journal(ended(1, "desk"))
        .journal(installed("desk"))
        .done()
}

/// A task started from a task: the inner one ends first, back to the outer one's agent.
fn nested() -> Recording {
    Authored::new()
        .journal(entered(1, "desk", "email"))
        .journal(installed("email"))
        .journal(entered(2, "email", "spell"))
        .journal(installed("spell"))
        .journal(ended(2, "email"))
        .journal(installed("email"))
        .journal(ended(1, "desk"))
        .journal(installed("desk"))
        .done()
}

fn ended_twice() -> Recording {
    Authored::new()
        .journal(entered(1, "desk", "email"))
        .journal(installed("email"))
        .journal(ended(1, "desk"))
        .journal(installed("desk"))
        .journal(ended(1, "desk"))
        .journal(installed("desk"))
        .done()
}

fn outer_first() -> Recording {
    Authored::new()
        .journal(entered(1, "desk", "email"))
        .journal(installed("email"))
        .journal(entered(2, "email", "spell"))
        .journal(installed("spell"))
        .journal(ended(1, "desk"))
        .journal(installed("desk"))
        .done()
}

/// The end names the parent, but the agent installed is still the task's.
fn wrong_agent() -> Recording {
    Authored::new()
        .journal(entered(1, "desk", "email"))
        .journal(installed("email"))
        .journal(ended(1, "desk"))
        .journal(installed("email"))
        .done()
}

/// The task ended, and its parent was never installed again.
fn not_restored() -> Recording {
    Authored::new()
        .journal(entered(1, "desk", "email"))
        .journal(installed("email"))
        .journal(ended(1, "desk"))
        .done()
}

fn open_at_close() -> Recording {
    Authored::new()
        .journal(entered(1, "desk", "email"))
        .journal(installed("email"))
        .journal(Step::Close(lk_agents_voice::trace::CloseStep::Requested {
            reason: lk_agents_abi::voice::events::CloseReason::UserInitiated,
            drain: false,
        }))
        .done()
}
