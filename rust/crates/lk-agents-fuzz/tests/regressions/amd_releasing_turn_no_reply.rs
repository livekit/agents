//! The turn whose end releases AMD's machine verdict gets no reply, not one created and then cut.
use lk_agents_fuzz::drive::{AmdAnswer, AmdSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AmdInput, Input, Millis, Ref, ToolArgs, TurnMode, UserText};
use lk_agents_voice::events::SessionEvent;

// Catches: the session decided the turn before AMD heard its end, so the turn that released a
// machine verdict under `interrupt_on_machine` still created a reply, which AMD's interrupt then
// cut. The reference's `on_end_of_turn` asks `amd._on_end_of_turn(info)` first, which forwards the
// end of turn to the classifier and, the verdict now out, skips the reply.
#[test]
fn the_releasing_turn_gets_no_reply() {
    let setup = Setup {
        vad: true,
        stt: true,
        turn: TurnMode::Stt,
        tools: Vec::new(),
        amd: Some(AmdSetup {
            interrupt_on_machine: true,
            answer: AmdAnswer::Track,
            ..AmdSetup::default()
        }),
        ..Setup::default()
    };
    let inputs = [
        // A long greeting: a machine's silence threshold applies.
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(3000) },
        Input::SttStart { stt: Ref::new(0) },
        Input::Advance { ms: Millis(300) },
        Input::SttFinal {
            utterance: Ref::new(0),
            text: UserText("hi you have reached the voicemail of theo".into()),
            speaker: None,
        },
        Input::Advance { ms: Millis(20) },
        // The classifier answers while the greeting's turn is still open.
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "save_prediction".into(),
            args: ToolArgs(r#"{"label":"machine-vm"}"#.into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) },
        // Past the machine silence threshold: the verdict now waits on the end of turn alone.
        Input::Advance { ms: Millis(2000) },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::Advance { ms: Millis(3000) },
        Input::Amd { input: AmdInput::Execute },
        Input::Advance { ms: Millis(1000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let events = run.record.host_view().events();
    let machine = events
        .iter()
        .any(|(_, ev)| matches!(ev, SessionEvent::AmdResult { verdict } if verdict.is_machine()));
    let created =
        events.iter().filter(|(_, ev)| matches!(ev, SessionEvent::SpeechCreated { .. })).count();
    assert_eq!((machine, created), (true, 0));
}
