//! A realtime reply the server answers while the session closes is cancelled, not left generating.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis};

// Catches: a turn committed while the session was closing asked the server for a reply, and the
// answer was discarded (nothing plays it) without telling the server to stop: still generating,
// it refused the next reply asked meanwhile (`conversation_already_has_active_response`,
// `realtime_server_accepts_requests`, survey seed 8349). The discarded answer is now cancelled.
#[test]
fn a_reply_answered_while_closing_frees_the_server() {
    let setup = Setup {
        hook_bound_ms: Some(2_500),
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(0),
            ..RealtimeSetup::default()
        }),
        ..Setup::default()
    };
    let inputs = [
        // The agent's `on_enter` is still running, so the close takes its bound.
        Input::Close { drain: false },
        Input::Advance { ms: Millis(100) },
        Input::CommitTurn { skip_reply: false },
        Input::Advance { ms: Millis(500) },
        Input::CommitTurn { skip_reply: false },
        Input::Advance { ms: Millis(2_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let (failures, _) = check_all(&run.record);
    let refused: Vec<_> = failures
        .iter()
        .filter(|f| f.contract == "realtime_server_accepts_requests")
        .map(|f| f.detail.as_str())
        .collect();
    assert!(refused.is_empty(), "{refused:?}");
}
