//! A recogniser, a detector or a microphone the session replaced is closed, so nothing a case says
//! into it can look like it was heard.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Models};
use lk_agents_fuzz::record::Kind;

// Catches: the fakes ignoring `end_input` and the harness keeping a replaced microphone open. A
// real stream ends once its input does; the fake never did, so the conversation kept speaking
// into the first recogniser, whose audio had moved to the new one, and every final it held back
// for want of audio left the caller unheard (soak seeds 0, 1, 3 and 6 heard 0 to 4 turns of 40).
#[test]
fn replaced_inputs_close() {
    let inputs = [
        Input::UpdateModels { models: Models::All },
        Input::SetAudioInput { attach: true },
        Input::Advance { ms: Millis(20) },
    ];
    let run = simulate(async {
        let mut run =
            Run::start(&Setup { stt: true, vad: true, ..Setup::default() }, 0, Default::default())
                .await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    for kind in [Kind::Stt, Kind::Vad, Kind::Mic] {
        let live = run.record.live().of(kind);
        assert_eq!(live.len(), 1, "one {kind:?} live, the newest: {live:?}");
    }
}
