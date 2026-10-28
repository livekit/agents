//! A `VoiceTask` suspends its parent and restores it without the parent's hooks.
//!
//! The parent's activation is the same one before and after the task: its `on_exit` is not run
//! when the task takes the floor, nor its `on_enter` when it comes back. From the task's entry
//! to its end the parent's `on_exit` is never issued (nothing may replace an agent a task holds);
//! its `on_enter` neither, through the restore up to the next command the driver takes. A close
//! after the restore may still exit the parent: that is the close's exit, not the task's.
use lk_agents_voice::agent_activity::voice_task::{TaskEnd, VoiceTaskId};
use lk_agents_voice::agent_session::{HookId, HookKind};
use lk_agents_voice::trace::{CoreStep, HandoffStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::Recording,
};

pub struct VoiceTaskSkipsParentHooks;

impl Rule for VoiceTaskSkipsParentHooks {
    const CONTRACT: Contract = Contract {
        id: "voice_task_skips_parent_hooks",
        guarantee: "Neither the parent's on_exit nor its on_enter is issued for a VoiceTask's \
                    entry or restore.",
        examples: &[suspended_quietly],
        counterexamples: &[exited_on_entry, entered_on_restore],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // Per open task, its parent and its agent; the parents just restored, until the next
        // command. A hook named after an open task's own agent is the task's.
        let mut open: Vec<(VoiceTaskId, String, String)> = Vec::new();
        let mut restored: Vec<String> = Vec::new();
        for (at, step) in r.internal_view().steps() {
            match step {
                Step::Handoff(HandoffStep::TaskEntered { task, parent, agent, .. }) => {
                    open.push((*task, parent.clone(), agent.clone()));
                }
                Step::Handoff(HandoffStep::TaskEnded { task, .. }) => {
                    if let Some(i) = open.iter().position(|(t, ..)| t == task) {
                        restored.push(open.remove(i).1);
                    }
                }
                Step::Core(CoreStep::CommandReceived { .. }) => restored.clear(),
                Step::Handoff(HandoffStep::HookIssued { kind, agent: Some(agent), .. }) => {
                    let own = open.iter().any(|(_, _, own)| own == agent);
                    let suspended = !own && open.iter().any(|(_, parent, _)| parent == agent);
                    let (bad, when) = match kind {
                        HookKind::OnExit => (suspended, "while its task held the floor"),
                        HookKind::OnEnter => (
                            suspended || (!own && restored.contains(agent)),
                            "for its task's entry or restore",
                        ),
                        _ => (false, ""),
                    };
                    check.expect(
                        Subject::Session,
                        !bad,
                        format!("{kind:?} of {agent} issued at seq {} {when}", at.seq),
                    );
                }
                _ => {}
            }
        }
    }
}

fn entered() -> Step {
    Step::Handoff(HandoffStep::TaskEntered {
        task: VoiceTaskId(1),
        parent: "desk".into(),
        agent: "email".into(),
        speech: lk_agents_voice::agent_activity::scheduling::SpeechId(1),
    })
}

fn ended() -> Step {
    Step::Handoff(HandoffStep::TaskEnded {
        task: VoiceTaskId(1),
        how: TaskEnd::Completed,
        restored: "desk".into(),
    })
}

fn hook(id: u64, kind: HookKind, agent: &str) -> Step {
    Step::Handoff(HandoffStep::HookIssued {
        hook: HookId(id),
        kind,
        agent: Some(agent.into()),
        speech: None,
        turn: None,
    })
}

/// The task's own `on_enter`, and none of the parent's.
fn suspended_quietly() -> Recording {
    Authored::new()
        .journal(entered())
        .journal(hook(1, HookKind::OnEnter, "email"))
        .journal(ended())
        .done()
}

fn exited_on_entry() -> Recording {
    Authored::new()
        .journal(entered())
        .journal(hook(1, HookKind::OnExit, "desk"))
        .journal(ended())
        .done()
}

fn entered_on_restore() -> Recording {
    Authored::new()
        .journal(entered())
        .journal(ended())
        .journal(hook(1, HookKind::OnEnter, "desk"))
        .done()
}
