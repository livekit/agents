//! The release phase hears out a person whose recogniser was swapped mid-utterance.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Models, Ref};

// Catches: a recogniser swapped while the person spoke taking their utterance with it, so no end
// of speech was ever delivered and a reply held at launch behind them waited until the close
// (`speech_settles`, survey seed 2337).
#[test]
fn a_person_left_speaking_by_a_swapped_recogniser_is_heard_out() {
    let inputs = vec![
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::UpdateModels { models: Models::Stt },
        Input::Advance { ms: Millis(20) },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmText { chat: Ref::new(0), text: AgentText("Hello there.".into()) },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
    ];
    let setup = Setup { stt: true, ..Setup::default() };
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"speech_settles"), "{failed:?}");
    assert!(outcome.expectations.get("speech_settles").is_some_and(|n| *n > 0));
}
