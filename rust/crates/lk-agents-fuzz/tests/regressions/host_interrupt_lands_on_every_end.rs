//! A host `interrupt()` lands on the speech it was asked of however that speech ends, and one
//! asked of a speech that never ran lands nowhere: a done handle reads the core's verdict.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput};

// Catches: an `interrupt()` asked of a `say()` handle before its speech existed was held by the
// handle and sent once the loop created the speech, which the loop had created done (`say()` with
// no TTS is refused): the host read the speech interrupted forever while the core had journalled
// no interruption (`interrupted_handle_is_journalled`, survey seed 453 once the DST held a
// call's handle from its return).
#[test]
fn an_interrupt_of_a_speech_created_done_lands_nowhere() {
    let inputs = [
        Input::Advance { ms: Millis(100) },
        Input::Say {
            text: AgentText("Hello there.".into()),
            input: SayInput::Words,
            allow: None,
            to_ctx: true,
        },
        Input::InterruptSpeech { speech: Ref::new(0), force: true },
        Input::Advance { ms: Millis(5_000) },
    ];
    let run = simulate(async {
        let mut run =
            Run::start(&Setup { tts: false, ..Setup::default() }, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    assert!(run.record.applied()[2].accepted, "the scenario: the uncreated handle interrupted");
    let speeches = run.record.host_view().speeches();
    let [speech] = &speeches[..] else { panic!("one speech, got {}", speeches.len()) };
    assert!(speech.is_done() && speech.error().is_some(), "the scenario: the say was refused");
    assert!(!speech.is_interrupted(), "{speech:?}");
    let (failures, _) = check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
}
