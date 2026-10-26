//! Once AMD has called the line a machine, the turns it hears are its own, not the agent's.
use lk_agents_fuzz::drive::{AmdAnswer, AmdSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AmdInput, Input, Millis, Ref, ToolArgs, TurnMode, UserText};
use lk_agents_voice::events::SessionEvent;

/// One spoken turn on utterance `nth`: onset, a final, the end of speech.
fn turn(nth: u32, text: &str) -> Vec<Input> {
    vec![
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(300) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(300) },
        Input::SttFinal { utterance: Ref::new(nth), text: UserText(text.into()), speaker: None },
        Input::SttEnd { utterance: Ref::new(nth) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(500) },
        Input::Advance { ms: Millis(20) },
    ]
}

/// Whether AMD's verdict was a machine, and the speeches the session created after it.
fn after_verdict(interrupt_on_machine: bool, close: bool) -> (bool, usize) {
    let setup = Setup {
        vad: true,
        stt: true,
        turn: TurnMode::Stt,
        tools: Vec::new(),
        amd: Some(AmdSetup {
            interrupt_on_machine,
            answer: AmdAnswer::Track,
            ..AmdSetup::default()
        }),
        ..Setup::default()
    };
    let mut inputs = turn(0, "hi you have reached the voicemail of theo");
    // The classifier's request, answered as a model answers it.
    inputs.extend([
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "save_prediction".into(),
            args: ToolArgs(r#"{"label":"machine-vm"}"#.into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(3000) },
        Input::Amd { input: AmdInput::Execute },
    ]);
    if close {
        inputs.push(Input::Amd { input: AmdInput::Close });
    }
    inputs.extend(turn(1, "please leave a message after the tone"));
    inputs.push(Input::Advance { ms: Millis(3000) });
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let events = run.record.host_view().events();
    let verdict = events
        .iter()
        .position(|(_, ev)| matches!(ev, SessionEvent::AmdResult { .. }))
        .expect("AMD gave a verdict");
    let machine =
        matches!(events[verdict].1, SessionEvent::AmdResult { verdict } if verdict.is_machine());
    let created = events[verdict..]
        .iter()
        .filter(|(_, ev)| matches!(ev, SessionEvent::SpeechCreated { .. }))
        .count();
    (machine, created)
}

// Catches: a turn heard after a machine verdict answered by the agent. `AMD._on_end_of_turn`
// has the activity skip the reply (`info.skip_reply`) while the detector holds a machine verdict
// under `interrupt_on_machine`: the developer is leaving a voicemail, and an auto reply to the
// greeting's tail would talk over it.
#[test]
fn a_machine_verdict_skips_the_replies() {
    assert_eq!(after_verdict(true, false), (true, 0));
}

// The skip is the detector's: without `interrupt_on_machine`, or once it is closed, turns are
// answered again.
#[test]
fn replies_resume_without_the_detector() {
    assert_eq!(after_verdict(false, false), (true, 1));
    assert_eq!(after_verdict(true, true), (true, 1));
}
