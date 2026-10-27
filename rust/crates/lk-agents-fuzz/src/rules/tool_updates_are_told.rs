//! A background tool's later updates and its final result reach the conversation, whatever the
//! order in which its step's join hears it detach: `ctx.update()` reaches the driver directly,
//! while the detach is heard in the model's order, so an update can arrive first and must wait for
//! it rather than be dropped. Each progress update is told to the host (`ToolCallUpdated` under
//! `<call_id>_update_<n>`), and the final as `<call_id>_final`.
//!
//! The close may cut a tool before its updates land, and a cancelled tool says nothing more.
use std::collections::BTreeMap;

use lk_agents_voice::{
    events::{SessionEvent, ToolEndStatus, ToolLifecycle},
    trace::Step,
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AgentText, Input, Ref, ToolAnswer},
    record::{Detail, Kind, Recording},
};

pub struct ToolUpdatesAreTold;

impl Rule for ToolUpdatesAreTold {
    const CONTRACT: Contract = Contract {
        id: "tool_updates_are_told",
        guarantee: "Every update and the final result a background tool sends are told to the host, even when they arrive before its step's join heard it detach.",
        examples: &[an_update_told],
        counterexamples: &[an_update_dropped],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        if r.inputs_view().closing_asked_at().is_some() {
            return;
        }
        // Per call: updates sent, and finals sent.
        let mut sent: BTreeMap<String, (u32, u32, u32)> = BTreeMap::new();
        for applied in r.inputs_view().accepted() {
            let Input::ToolUpdate { tool, progress, end } = &applied.input else { continue };
            let Some(Detail::Tool { call_id, .. }) =
                r.resource(Kind::Tool, tool.nth).map(|t| t.detail().clone())
            else {
                continue;
            };
            let entry = sent.entry(call_id).or_insert((tool.nth, 0, 0));
            match (progress, end) {
                // A body that has returned sends nothing more; the case may still draw one.
                (Some(_), _) if entry.2 > 0 => {}
                (Some(_), _) => entry.1 += 1,
                (None, ToolAnswer::Detached) => {}
                (None, _) => entry.2 += 1,
            }
        }
        // Per call: updates and finals told, cancelled, and whether it was heard to detach.
        let mut told: BTreeMap<String, (u32, u32, bool, bool)> = BTreeMap::new();
        for (_, step) in r.journal() {
            let Step::Emitted(SessionEvent::ToolExecutionUpdated(update)) = step else { continue };
            match update {
                ToolLifecycle::Updated { id, call_id, .. } if id != call_id => {
                    told.entry(call_id.clone()).or_default().0 += 1;
                }
                ToolLifecycle::Updated { call_id, .. } => {
                    told.entry(call_id.clone()).or_default().3 = true;
                }
                ToolLifecycle::Ended { id, call_id, status, .. } => {
                    let entry = told.entry(call_id.clone()).or_default();
                    if *status == ToolEndStatus::Cancelled {
                        entry.2 = true;
                    } else if *id == format!("{call_id}_final") {
                        entry.1 += 1;
                    }
                }
                _ => {}
            }
        }
        for (call_id, (nth, updates, finals)) in sent {
            let (got_updates, got_finals, cancelled, detached) =
                told.get(&call_id).copied().unwrap_or_default();
            // A body only updates once it has detached (its first update is the detach); one
            // never heard to detach was cut first, or ended without detaching.
            if cancelled || !detached {
                continue;
            }
            let subject = Subject::Resource(Kind::Tool, nth);
            check.expect_with(subject.clone(), got_updates >= updates, || {
                format!("{call_id} sent {updates} updates, the host was told {got_updates}")
            });
            check.expect_with(subject, got_finals >= finals.min(1), || {
                format!("{call_id} sent its final, the host was never told it ended")
            });
        }
    }
}

fn update(id: &str) -> Step {
    Step::Emitted(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Updated {
        id: id.into(),
        call_id: "c1".into(),
        message: "half way".into(),
    }))
}

fn sent() -> Authored {
    Authored::new()
        .tool_call("c1", 1)
        .sent(Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Detached,
            text: AgentText("started".into()),
        })
        .sent(Input::ToolUpdate {
            tool: Ref::new(0),
            progress: Some(AgentText("half way".into())),
            end: ToolAnswer::Return,
        })
}

/// The update arrives before the detach is heard, and is told after it.
fn an_update_told() -> Recording {
    sent().journal(update("c1")).journal(update("c1_update_1")).done()
}

/// The update arrives before the detach is heard, and is lost.
fn an_update_dropped() -> Recording {
    sent().journal(update("c1")).done()
}
