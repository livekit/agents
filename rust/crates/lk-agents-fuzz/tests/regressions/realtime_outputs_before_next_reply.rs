//! A realtime tool's output reaches the server with its commit, before anything the commit lets
//! go asks the server for a reply.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
use lk_agents_voice::trace::{RealtimeStep, Step, ToolStep};

// Catches: the outputs pushed to the server only on the step's later wake, after the commit had
// already let a waiting reply go: a background tool's deferred reply, held while another
// speech's tools ran, was asked for (`response.create`) before that speech's tool output reached
// the server, and read the call without its result (`realtime_reply_reads_tool_output`, campaign
// seed 1978). The outputs now go with the commit, ahead of everything it releases.
#[test]
fn the_output_goes_with_the_commit() {
    let setup = Setup {
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(100),
            ..RealtimeSetup::default()
        }),
        tools: vec!["lookup".into()],
        ..Setup::default()
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(10_000) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Value,
            text: AgentText("shipped".into()),
        },
        Input::Advance { ms: Millis(10_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            super::playout::apply_with_playout(&mut run, input).await;
        }
        run
    });
    let journal = run.record.journal();
    let position = |pick: &dyn Fn(&Step) -> bool| journal.iter().position(|(_, s)| pick(s));
    let pushed = position(&|s| {
        matches!(
            s,
            Step::Realtime(RealtimeStep::Sent {
                kind: lk_agents_abi::RealtimeCommandKind::PushItem,
                ..
            })
        )
    });
    let finished = position(&|s| matches!(s, Step::Tool(ToolStep::Finished { .. })));
    let (Some(pushed), Some(finished)) = (pushed, finished) else {
        panic!("the tool's output pushed ({pushed:?}) and committed ({finished:?})")
    };
    assert!(pushed < finished, "pushed at step {pushed}, after the commit at step {finished}");
}
