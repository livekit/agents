use crate::{
    authored::{Authored, amd_outcome},
    check::{Check, Contract, Rule, Subject},
    fakes::amd::VOICEMAIL,
    record::{Handle, Observed, Recording},
};
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
use lk_agents_voice::amd::AmdRunId;
use lk_agents_voice::amd::events::{AmdEvent, AmdReason, Stage};
use lk_agents_voice::trace::AmdStep;

// Spec §14 invariant 6: one voicemail message at most. A run journals `VoicemailPlayed` once at
// most, only for a speech the sink played and that ended neither interrupted nor failed, and the
// outcome reports `voicemail_message_played` exactly when it did (spec §10.4, §12.1). At the
// sink: of the replies made under the voicemail stage's instructions, at most one per run is
// heard and plays to its end, whatever the run journalled.

pub struct OneVoicemailMessage;

impl Rule for OneVoicemailMessage {
    const CONTRACT: Contract = Contract {
        id: "one_voicemail_message",
        guarantee: "AMD leaves at most one voicemail message, counts it only once it played to its \
                    end, and says so when it completes.",
        examples: &[one_message_reported, an_interrupted_attempt_then_the_message],
        counterexamples: &[
            two_messages,
            an_unheard_message,
            a_message_not_reported,
            two_messages_heard_one_journalled,
        ],
        probe: Some(a_voicemail_greeting),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let view = r.amd_run();
        let amd = r.amd_view();
        let entered = view.entered();
        for (index, (from, run)) in entered.iter().copied().enumerate() {
            let until = entered.get(index + 1).map_or(Observed::END, |(at, _)| *at);
            let whole_messages: Vec<u64> = (amd.requests().into_iter())
                .filter(|q| q.opened >= from && q.opened < until && q.carries(VOICEMAIL))
                .filter_map(|q| q.speech)
                .filter(|speech| {
                    let handle = r.host_view().speech(*speech);
                    amd.first_audio(*speech).is_some()
                        && handle.is_done()
                        && !handle.is_interrupted()
                })
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            check.expect_with(Subject::Session, whole_messages.len() <= 1, || {
                format!(
                    "run {} had {} voicemail-stage replies heard to their end: {whole_messages:?}",
                    run.0,
                    whole_messages.len()
                )
            });
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
    voicemail_reply(run, speech).speech(speech, finished()).amd_step(played(speech))
}

/// A reply made under the voicemail stage's instructions, heard at the sink.
fn voicemail_reply(run: Authored, speech: u64) -> Authored {
    run.request(speech, speech, "please leave a message", &[VOICEMAIL], &[]).pushed(speech)
}

/// The first attempt is interrupted, so it is not the message; the retry is.
fn an_interrupted_attempt_then_the_message() -> Recording {
    let interrupted = Handle { done: true, interrupted: true, ..Handle::unknown() };
    let run = voicemail_reply(Authored::new().amd_entered(1), 7).speech(7, interrupted);
    a_message(run, 8).amd_event(1, completed(true)).done()
}

/// Two voicemail-stage replies are heard to their end; the run journals one.
fn two_messages_heard_one_journalled() -> Recording {
    let run = a_message(Authored::new().amd_entered(1), 7);
    voicemail_reply(run, 8).speech(8, finished()).amd_event(1, completed(true)).done()
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
