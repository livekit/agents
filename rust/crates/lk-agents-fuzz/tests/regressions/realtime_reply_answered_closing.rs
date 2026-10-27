//! A realtime reply the server answers while the session closes is cancelled, not left generating.
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};

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

// Catches: a reply the server opened on its own (its idle timeout) while the session closed was
// discarded and left generating, as only a reply the client had asked for was cancelled: the
// turn committed next asked over it and was refused (`conversation_already_has_active_response`,
// `realtime_server_accepts_requests`, survey seed 104301).
#[test]
fn a_server_reply_opened_while_closing_frees_the_server() {
    let setup = Setup {
        hook_bound_ms: Some(2_500),
        realtime: Some(RealtimeSetup {
            turn_detection: true,
            tool_calls: Some(0),
            ..RealtimeSetup::default()
        }),
        ..Setup::default()
    };
    let inputs = [
        Input::Close { drain: false },
        Input::Advance { ms: Millis(100) },
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SecondGeneration },
        Input::Advance { ms: Millis(100) },
        Input::CommitUserTurn { transcript_timeout: Millis(5_000), stt_flush: Millis(120), skip_reply: false },
        Input::Advance { ms: Millis(2_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let accepted = run.record.applied().iter().filter(|a| a.accepted).count();
    assert_eq!(accepted, inputs.len(), "every input reached the session");
    let (failures, _) = check_all(&run.record);
    let refused: Vec<_> = failures
        .iter()
        .filter(|f| f.contract == "realtime_server_accepts_requests")
        .map(|f| f.detail.as_str())
        .collect();
    assert!(refused.is_empty(), "{refused:?}");
}
