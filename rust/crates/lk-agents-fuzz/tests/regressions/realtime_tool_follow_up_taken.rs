//! A realtime tool follow-up is taken by the server even while another reply generates.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
use lk_agents_fuzz::record::Breach;

// Catches: the tools' follow-up asked while the server was generating another reply (one the
// host asked for while the tool ran): the server refused it (`conversation_already_has_active_
// response`) and the tool's result was never spoken. The reference interrupts the session first
// (`self._rt_session.interrupt()` before the tool reply), and the server takes a reply asked
// right after a cancel (`corpus/probes/probe-cancel_then_create.json`).
#[test]
fn the_follow_up_is_taken_over_a_reply_generating() {
    let setup = Setup {
        tools: vec!["lookup_order".into()],
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(100),
            ..RealtimeSetup::default()
        }),
        ..Setup::default()
    };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(5_000) },
        // The reply's words play out; the tool still runs, and the host's reply may go.
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(600) },
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Value,
            text: AgentText("shipped".into()),
        },
        Input::Advance { ms: Millis(5_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let breaches = run.record.inference_view().realtime_breaches();
    let refused: Vec<_> = breaches.iter().filter(|(_, _, b)| matches!(b, Breach::Refused { .. })).collect();
    assert!(refused.is_empty(), "{refused:?}");
}
