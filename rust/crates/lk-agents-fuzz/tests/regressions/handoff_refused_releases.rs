//! The providers a refused handoff hands over are released.
use lk_agents_fuzz::drive::{Case, NodeSetup, Setup, run};
use lk_agents_fuzz::input::{AgentText, Input};

// Catches: `update_agent` registering the frame's providers before asking for its answer, so a
// handoff refused because the session had ended left them in the table for good
// (`every_resource_released`, survey seed 66). The host gives every leg on every handoff
// (`_frame`), so each late handoff leaked three.
#[test]
fn a_handoff_refused_after_the_end_releases_its_providers() {
    let setup = Setup {
        stt: true,
        nodes: Some(NodeSetup {
            llm: true,
            tts: true,
            stt: true,
            llm_text: None,
            transcription: false,
            realtime_audio: false,
        }),
        ..Setup::default()
    };
    let inputs = vec![
        Input::Close { drain: false },
        Input::AwaitExit {},
        Input::Handoff {
            agent: "late".into(),
            instructions: AgentText("Hi.".into()),
            tools: Vec::new(),
            turn: None,
        },
    ];
    let outcome = run(Case { setup, inputs, ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"every_resource_released"), "{failed:?}");
}
