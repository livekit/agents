//! Replacing an output detaches the one it replaces, as `AgentOutput`'s setters do.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis};
use lk_agents_fuzz::record::Sink;

// Catches: `SetTranscriptionOutput` swapping the sink without `on_detached` on the old one
// (`io.py`'s setter). The synchroniser between the outputs never heard its text leg go, stayed
// enabled, and kept pacing an in-flight speech's words into the replaced leaf (survey seed 57).
// Detached, it closes its segment on the leaf at once, as `_on_attachment_changed` rotates it.
#[test]
fn a_replaced_synchronised_text_leg_is_detached() {
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    let inputs =
        [Input::SetTranscriptionOutput { attach: true }, Input::Advance { ms: Millis(20) }];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let flushed: Vec<_> = run
        .record
        .sink()
        .iter()
        .filter_map(|(_, s)| match s {
            Sink::TextFlush { output } => Some(*output),
            _ => None,
        })
        .collect();
    assert_eq!(flushed, [1], "the old leaf's segment closed when its leg was detached");
}
