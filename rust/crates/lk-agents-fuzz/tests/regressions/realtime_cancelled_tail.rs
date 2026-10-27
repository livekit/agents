//! The tail of a cancelled response, which the server sends after the next reply was created,
//! belongs to neither: the next reply generates on and plays only its own audio.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis};
use lk_agents_fuzz::record::Tag;

// Catches: the native session taking the cancelled response's tail for the reply asked in the
// same breath. The real server frees the conversation at a cancel and creates the new reply
// before the cancelled one's last audio delta and `response.done`
// (`corpus/probes/probe-cancel_then_create.json`). Its late deltas were played as the new
// reply's, and its late `response.done` ended the new reply's generation, whose audio was then
// dropped: the agent fell silent.
#[test]
fn a_cancelled_response_tail_leaves_the_next_reply_alone() {
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
        ..Setup::default()
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(1_000) },
        Input::Interrupt { force: true, wait: false },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let r = &run.record;
    let asked_again = r.applied().iter().filter(|a| matches!(a.input, Input::GenerateReply { .. })).nth(1);
    let since = asked_again.expect("the second reply was asked").at;
    let pushed = r.host_view().audio_pushed();
    let generation = |n: u32| {
        pushed
            .iter()
            .filter(|p| p.at > since)
            .any(|p| p.tags.iter().any(|t| matches!(t, Tag::Generated { generation, .. } if *generation == n)))
    };
    assert!(generation(1), "the new reply's audio is played");
    assert!(!generation(0), "none of the cancelled response's tail is");
}
