//! `handle.allow_interruptions = v` sticks: once the loop has had a turn, the handle reads `v`.
//!
//! The setter on a running speech asks the loop, which publishes the value back; on an ended
//! speech the value is the handle's own, as in 1.x. Either way the host reads what it set. A
//! request the loop never applied (asked for a speech it had already let go) leaves the old value
//! (ef4a20f5c). An interrupted speech refuses `False`, and a tool's interruption hold restores its
//! own value when released, so those speeches are not judged.
use lk_agents_voice::trace::{CoreStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{Input, Ref},
    record::{Handle, Id, Kind, Recording},
};

pub struct SpeechKeepsTheHostsSetting;

impl Rule for SpeechKeepsTheHostsSetting {
    const CONTRACT: Contract = Contract {
        id: "speech_keeps_the_hosts_setting",
        guarantee: "A speech handle reads the allow_interruptions the host last set on it, once the loop has had a turn.",
        examples: &[set_on_a_running_speech, set_on_an_ended_speech],
        counterexamples: &[setting_lost],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let mut last = std::collections::BTreeMap::new();
        for sent in inputs.accepted() {
            if let Input::SetAllowInterruptions { speech, allow } = &sent.input
                && let Some(Id::Speech(id)) = r.resource(Kind::Speech, speech.nth).map(|s| s.id())
            {
                last.insert(id.0, (sent.at, *allow));
            }
        }
        let held = r.journal().iter().filter_map(|(at, step)| match step {
            Step::Core(CoreStep::CommandReceived { kind, .. })
                if *kind == "SpeechCommand::HoldInterruptions" =>
            {
                Some(*at)
            }
            _ => None,
        });
        let held: Vec<_> = held.collect();
        for (speech, (at, allow)) in last {
            let Some(handle) = r.handles().get(&speech) else { continue };
            if handle.interrupted || !inputs.driver_ran_after(at) || held.iter().any(|h| *h > at) {
                continue;
            }
            check.expect(
                Subject::Speech(speech),
                handle.allow_interruptions == allow,
                format!(
                    "the host set allow_interruptions={allow} on speech {speech}, the handle reads {}",
                    handle.allow_interruptions
                ),
            );
        }
    }
}

fn setter(allow: bool) -> Input {
    Input::SetAllowInterruptions { speech: Ref::new(0), allow }
}

fn said() -> Authored {
    use lk_agents_voice::agent_activity::scheduling::SpeechKind;
    Authored::new().speech_created(1, SpeechKind::Say)
}

/// Set on a running speech: the loop published it back.
fn set_on_a_running_speech() -> Recording {
    said()
        .sent(setter(false))
        .then_act()
        .speech(1, Handle { allow_interruptions: false, ..Handle::live() })
        .done()
}

/// Set after the speech ended: the handle keeps it.
fn set_on_an_ended_speech() -> Recording {
    said()
        .speech(1, Handle::finished())
        .sent(setter(false))
        .then_act()
        .speech(1, Handle { allow_interruptions: false, ..Handle::finished() })
        .done()
}

/// Asked of a speech the loop had let go: the request was dropped and the value lost.
fn setting_lost() -> Recording {
    said()
        .speech(1, Handle::finished())
        .sent(setter(false))
        .then_act()
        .speech(1, Handle::finished())
        .done()
}
