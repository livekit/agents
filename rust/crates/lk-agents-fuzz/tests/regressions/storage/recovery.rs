//! Cancellation during restoration must end the acquired run, including its durable journal row.

use lk_agents_fuzz::drive::{Outcome, Run, Setup, StorageSetup, finish, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref};

/// A first reply stores immutable content. The next reply takes ownership but is interrupted
/// while reading that content; the interruption watchdog cancels its unfinished load.
#[test]
fn interrupting_snapshot_loading_ends_the_acquired_run() {
    let outcome = simulate(async {
        let setup = Setup {
            tts: false,
            storage: Some(StorageSetup {
                session_read_answer_ms: Some(20_000),
                ..StorageSetup::default()
            }),
            ..Setup::default()
        };
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(1_000) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("Hello.".into()) },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
        ] {
            run.apply(&input).await;
        }
        for _ in 0..100 {
            if run.session_read_unanswered() {
                break;
            }
            run.apply(&Input::Advance { ms: Millis(1) }).await;
        }
        assert!(run.session_read_unanswered(), "the next reply did not reach snapshot loading");
        run.apply(&Input::Interrupt { force: true, wait: false }).await;
        run.apply(&Input::Advance { ms: Millis(5_500) }).await;
        assert!(!run.session_read_unanswered(), "the watchdog did not cancel the load");
        finish(run, Outcome::default()).await
    });
    assert_eq!(outcome.expectations.get("storage_partition_released_at_close"), Some(&1));
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}
