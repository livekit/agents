use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
use lk_agents_voice::trace::{Step, ToolStep};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Handle, Recording},
};

// The executor's end and the run's settlement of a tool step meet once, and their results are
// handled once: a step left owed once everything was answered is a reply waiting on nobody, a silent
// agent.

pub struct ReplyStepIsHandled;

impl Rule for ReplyStepIsHandled {
    const CONTRACT: Contract = Contract {
        id: "reply_step_is_handled",
        guarantee: "Every tool step the executor ended has its results handled once, or its speech is done, once everything was answered.",
        examples: &[handled, released],
        counterexamples: &[owed, unended],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let Some(answered) = r.inputs_view().everything_answered_at() else { return };
        let mut owed = std::collections::BTreeMap::new();
        for (at, step) in r.internal_view().steps().iter().filter(|(at, _)| *at <= answered) {
            match step {
                Step::Tool(ToolStep::StepEnded { speech, step }) => {
                    owed.insert(speech.0, (*at, *step));
                }
                Step::Tool(ToolStep::Finished { speech, .. }) => {
                    let ended = owed.remove(&speech.0).is_some();
                    check.expect(Subject::Speech(speech.0), ended, "results handled for no step");
                }
                _ => {}
            }
        }
        for (speech, (at, step)) in owed {
            let done = r.host_view().speech(speech).done_at().is_some_and(|done| done <= answered);
            check.expect(
                Subject::Speech(speech),
                done,
                format!("step {step} ended at {at:?}, its results were never handled"),
            );
        }
    }
}

fn ended() -> Step {
    Step::Tool(ToolStep::StepEnded { speech: SpeechId(7), step: 1 })
}

/// Ended, then handled.
fn handled() -> Recording {
    Authored::new()
        .speech(7, Handle::live())
        .journal(ended())
        .journal(Step::Tool(ToolStep::Finished { speech: SpeechId(7), committed: 2 }))
        .everything_answered()
        .done()
}

/// Ended, and the speech let go with it.
fn released() -> Recording {
    Authored::new().journal(ended()).speech(7, Handle::finished()).everything_answered().done()
}

/// Ended, and owed still once everything was answered.
fn owed() -> Recording {
    Authored::new().speech(7, Handle::live()).journal(ended()).everything_answered().done()
}

/// Handled with no step ended.
fn unended() -> Recording {
    Authored::new()
        .speech(7, Handle::live())
        .journal(Step::Tool(ToolStep::Finished { speech: SpeechId(7), committed: 2 }))
        .everything_answered()
        .done()
}
