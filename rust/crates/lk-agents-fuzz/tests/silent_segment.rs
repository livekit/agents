//! A speech authorized after its audio output was detached plays a silent segment, and a cut
//! still ends it.
use lk_agents_fuzz::drive::{Case, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};

// Catches: the no-output path of `perform_audio_forwarding` awaiting the next frame without
// racing the interruption. Synthesis opened while the output was attached; once the speech is
// cut the producer stops without ending the segment, so the reply task never finished, its
// synthesis stream was never closed, and the run never went quiet (survey seed 25750 at 6000
// seeds from 21000, where joining the task parked the campaign for good).
#[test]
fn a_cut_silent_segment_ends() {
    let inputs = vec![
        Input::PauseAuthorization {},
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        // The words start a synthesis, which never ends: no `LlmEnd`.
        Input::LlmText { chat: Ref::new(0), text: AgentText("Hello there.".into()) },
        Input::Advance { ms: Millis(20) },
        Input::SetAudioOutput { attach: false },
        Input::ResumeAuthorization {},
        Input::Advance { ms: Millis(20) },
        Input::Close { drain: false },
        Input::Advance { ms: Millis(20) },
    ];
    let outcome = run(Case { setup: Setup::default(), inputs, ..Case::default() });
    assert!(outcome.ok(), "{:#?}", outcome.failures);
}
