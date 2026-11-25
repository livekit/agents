use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input},
    record::Recording,
};

use lk_agents_voice::amd::AmdObservation;

// the AMD proposal, "Detach" and "Completion": the session applies `Detach`, removes the
// gate and sends `Detached` at once, so AMD's completion never waits for customer code. A
// session that never acknowledges fails here.

pub struct DetachedIsPrompt;

/// How long the acknowledgement may take: one scheduling instant, never a hook's bound.
const BOUND_MS: u64 = 50;

impl Rule for DetachedIsPrompt {
    const CONTRACT: Contract = Contract {
        id: "detached_is_prompt",
        guarantee: "A Detach the session accepted is acknowledged with Detached within one \
                    instant, whatever customer hook is still running.",
        examples: &[acknowledged],
        counterexamples: &[never_acknowledged, acknowledged_late],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        for attachment in amd.attachments() {
            let Some(asked) = attachment.detach_asked else { continue };
            let due = asked.at_ms + BOUND_MS;
            // A close that began first refuses what the loop had not read yet.
            if r.close_begun().is_some_and(|c| c.at_ms <= due) || r.now().at_ms < due {
                continue;
            }
            check.expect_with(
                Subject::Session,
                attachment.detached.is_some_and(|d| d > asked && d.at_ms <= due),
                || {
                    format!(
                        "run {} was asked to detach at {} ms; Detached came at {:?}",
                        attachment.run, asked.at_ms, attachment.detached
                    )
                },
            );
        }
    }
}

fn detach() -> Input {
    Input::AmdDetach { category: AmdVerdict::Allow, forbid: false }
}

fn acknowledged() -> Recording {
    Authored::new()
        .sent(Input::AmdAttach {})
        .sent(detach())
        .amd(1, AmdObservation::Detached { run: lk_agents_voice::amd::AmdRunId(1) })
        .later(1_000)
        .done()
}

fn never_acknowledged() -> Recording {
    Authored::new().sent(Input::AmdAttach {}).sent(detach()).later(1_000).done()
}

fn acknowledged_late() -> Recording {
    Authored::new()
        .sent(Input::AmdAttach {})
        .sent(detach())
        .later(1_000)
        .amd(1, AmdObservation::Detached { run: lk_agents_voice::amd::AmdRunId(1) })
        .later(1_000)
        .done()
}
