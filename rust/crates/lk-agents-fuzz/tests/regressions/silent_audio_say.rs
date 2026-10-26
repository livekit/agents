//! A `say(audio=)` with no audio path speaks its text alone and leaves the frames unread.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, SayInput};

// Catches: `no_premature_completion` judging an audio say spoken with the audio output detached.
// With no sink the reference's `_tts_task` forwards the text only and never reads `audio`, so the
// speech completes with the caller's stream still open (180 of the rule's 222 failures over seeds
// 0 to 1023, e.g. seed 2).
#[test]
fn an_audio_say_with_no_output_completes_on_its_text() {
    let inputs = [
        Input::SetAudioOutput { attach: false },
        Input::Advance { ms: Millis(20) },
        Input::Say {
            text: AgentText("Hello.".into()),
            input: SayInput::Audio,
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
    let host = run.record.host_view();
    let speeches = host.speeches();
    let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
    assert!(speech.completed_normally(), "the text was spoken");
    let (failures, _) = check_all(&run.record);
    let failed: Vec<_> = failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"no_premature_completion"), "{failed:?}");
}
