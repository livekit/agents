//! A leg's path is open only while its enable and its attachment both are.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};
use lk_agents_voice::agent_session::IoKind;

// Catches: the harness reading a leg's two independent switches as one, so enabling the
// transcription leg after detaching its output read as the path reopening. A speech spoken with
// no text output then looked like it lost its words (`text_conservation`, and the same misread of
// the audio input behind most `forwarded_while_enabled` findings).
#[test]
fn enabling_a_detached_leg_does_not_reopen_it() {
    let inputs = [
        Input::SetTranscriptionOutput { attach: false },
        Input::SetIoEnabled { leg: IoKind::Transcription, enabled: true },
        Input::SetAudioOutput { attach: false },
        Input::Advance { ms: Millis(20) },
        Input::Say {
            text: AgentText("Hello.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::Advance { ms: Millis(100) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let r = &run.record;
    assert!(!r.inputs_view().transcription_output().on_at(r.now()));
    let speeches = r.host_view().speeches();
    let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
    assert!(speech.completed_normally());
    let (failures, _) = check_all(r);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"text_conservation"), "{failed:?}");
}
