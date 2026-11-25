use crate::{
    authored::{Authored, amd_outcome},
    check::{Check, Contract, Rule, Subject},
    record::{Handle, Recording},
};
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
use lk_agents_voice::amd::AmdRunId;
use lk_agents_voice::amd::events::{AmdEvent, AmdReason, Stage};
use lk_agents_voice::trace::AmdStep;

// Spec §14 invariant 6: one voicemail message at most. A run journals `VoicemailPlayed` once at
// most, only for a speech the sink played and that ended neither interrupted nor failed, and the
// outcome reports `voicemail_message_played` exactly when it did (spec §10.4, §12.1).

pub struct OneVoicemailMessage;

impl Rule for OneVoicemailMessage {
    const CONTRACT: Contract = Contract {
        id: "one_voicemail_message",
        guarantee: "AMD leaves at most one voicemail message, counts it only once it played to its \
                    end, and says so when it completes.",
        examples: &[one_message_reported],
        counterexamples: &[two_messages, an_unheard_message, a_message_not_reported],
        probe: Some(a_voicemail_greeting),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let view = r.amd_run();
        let amd = r.amd_view();
        for (_, run) in view.entered() {
            let played: Vec<_> = (view.steps().into_iter())
                .filter_map(|(at, s)| match s {
                    AmdStep::VoicemailPlayed { run: r, speech } if r == run => Some((at, speech)),
                    _ => None,
                })
                .collect();
            if !played.is_empty() {
                check.expect_with(Subject::Session, played.len() == 1, || {
                    format!("run {} played {} voicemail messages", run.0, played.len())
                });
            }
            for (at, speech) in &played {
                let heard = amd.first_audio(speech.0).is_some_and(|heard| heard < *at);
                let handle = r.host_view().speech(speech.0);
                let whole = handle.is_done() && !handle.is_interrupted();
                check.expect_with(Subject::Speech(speech.0), heard && whole, || {
                    format!(
                        "speech {} counted as the voicemail message: heard {heard}, played to \
                         its end {whole}",
                        speech.0
                    )
                });
            }
            for (at, outcome) in view.completed(run) {
                check.expect_with(
                    Subject::Session,
                    outcome.voicemail_message_played == !played.is_empty(),
                    || {
                        format!(
                            "run {} completed at {} ms saying voicemail_message_played={} after \
                             {} played messages",
                            run.0,
                            at.at_ms,
                            outcome.voicemail_message_played,
                            played.len()
                        )
                    },
                );
            }
        }
    }
}

fn played(speech: u64) -> AmdStep {
    AmdStep::VoicemailPlayed { run: AmdRunId(1), speech: SpeechId(speech) }
}

fn finished() -> Handle {
    Handle { done: true, interrupted: false, ..Handle::unknown() }
}

fn completed(reported: bool) -> AmdEvent {
    AmdEvent::Completed(amd_outcome(Stage::MachineVm, AmdReason::IdleTimeout, reported))
}

fn a_message(run: Authored, speech: u64) -> Authored {
    run.pushed(speech).speech(speech, finished()).amd_step(played(speech))
}

fn one_message_reported() -> Recording {
    a_message(Authored::new().amd_entered(1), 7).amd_event(1, completed(true)).done()
}

fn two_messages() -> Recording {
    let run = a_message(Authored::new().amd_entered(1), 7);
    a_message(run, 8).amd_event(1, completed(true)).done()
}

fn an_unheard_message() -> Recording {
    Authored::new().amd_entered(1).amd_step(played(7)).amd_event(1, completed(true)).done()
}

fn a_message_not_reported() -> Recording {
    a_message(Authored::new().amd_entered(1), 7).amd_event(1, completed(false)).done()
}

/// A voicemail greeting, the message, and the box going quiet.
fn a_voicemail_greeting() -> crate::drive::Case {
    use crate::fakes::amd::{AmdSetup, executor_probe::call, probe};
    use crate::input::{AgentText, AmdSays, Input, Ref};
    let options = lk_agents_voice::amd::options::AmdOptions {
        voicemail_idle_timeout: std::time::Duration::from_secs(3),
        ..crate::fakes::amd::options()
    };
    let mut inputs = probe::says(0, "please leave a message after the tone");
    inputs.push(Input::AmdReply { ask: Ref::new(0), says: AmdSays::Voicemail });
    inputs.push(probe::advance(2_000));
    let text = AgentText("Hello, this is the clinic. Please call us back at your convenience.".into());
    inputs.push(Input::LlmText { chat: Ref::new(0), text });
    inputs.push(Input::LlmEnd { chat: Ref::new(0) });
    inputs.push(probe::advance(12_000));
    call(AmdSetup { options, ..AmdSetup::default() }, inputs)
}
