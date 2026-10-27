//! An io switch takes effect when the driver receives it, behind what the case sent before it.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Policy, Run, Setup, simulate_with};
use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};

/// A schedule on which the driver runs the `say` through before it takes the attach.
const SCHEDULE: u64 = 1;

// Catches: the io paths stamped by when the case sent each switch, while the driver handles
// commands in order behind the ones before them: a transcription output attached right after a
// `say` read as live for the speech the driver had already run detached, and
// `text_conservation` asked it for words it had no path to forward (survey seed 3028).
#[test]
fn an_attach_sent_after_a_say_is_not_the_says_path() {
    assert!(failures(SCHEDULE).is_empty(), "{:?}", failures(SCHEDULE));
}

fn failures(schedule: u64) -> Vec<String> {
    let setup = Setup { tts: false, ..Setup::default() };
    let run = simulate_with(schedule, async move {
        let mut run = Run::start(&setup, schedule, Policy::BasicPos).await;
        let inputs = [
            Input::SetTranscriptionOutput { attach: false },
            Input::SetAudioOutput { attach: false },
            Input::Advance { ms: Millis(20) },
            Input::Say {
                text: AgentText("Your balance is due on the fourteenth.".into()),
                input: SayInput::Words,
                allow: None,
                to_ctx: false,
            },
            Input::SetTranscriptionOutput { attach: true },
            Input::Advance { ms: Millis(500) },
        ];
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let (failures, _) = check_all(&run.record);
    failures
        .into_iter()
        .filter(|f| f.contract == "text_conservation")
        .map(|f| f.detail)
        .collect()
}
