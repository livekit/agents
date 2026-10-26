//! A speech held at its launch by a person whose end the driver can no longer hear is waiting on
//! them, not stuck.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{
    AgentText, Count, Input, Knob, Millis, Ref, SayInput, TurnMode, UserText,
};

// Catches: `speech_settles` reporting a say held at launch while the person speaks, after the
// turn detection left `stt` mid-utterance so the recogniser's end of speech is no longer taken
// and nothing can end it (the reference's reply waits on the user's silence the same way; survey
// seed 4050).
#[test]
fn a_launch_held_by_an_unheard_speaker_is_not_stuck() {
    let setup = Setup { stt: true, turn: TurnMode::Stt, tools: Vec::new(), ..Setup::default() };
    let utterance = |n| Input::SttEnd { utterance: Ref::new(n) };
    let inputs = vec![
        Input::Advance { ms: Millis(20) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        Input::Configure {
            knob: Knob::Detection,
            mode: TurnMode::Manual,
            ms: Millis(500),
            on: true,
            words: Count(1),
        },
        Input::Advance { ms: Millis(100) },
        Input::SttFinal { utterance: Ref::new(0), text: UserText("hello".into()), speaker: None },
        utterance(0),
        Input::Advance { ms: Millis(100) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        utterance(1),
        Input::Advance { ms: Millis(100) },
        Input::Say {
            text: AgentText("Hi there.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(1000) },
    ];
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_settles"), "{failed:?}");
}
