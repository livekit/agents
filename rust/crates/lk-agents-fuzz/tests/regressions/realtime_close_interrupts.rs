//! A close tells the realtime socket to stop even when no generation is open.
use lk_agents_abi::RealtimeCommandKind;
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};
use lk_agents_voice::trace::{RealtimeStep, Step};

// Catches: the session-level interrupt (which a close without drain runs) telling the socket only
// about generations it had open. The reference calls `rt_session.interrupt()` unconditionally, so
// a reply that had ended, or a response the server started and had not announced, is still told.
#[test]
fn a_close_interrupts_the_socket_with_no_generation_open() {
    let setup =
        Setup { realtime: Some(RealtimeSetup { turn_detection: false }), ..Setup::default() };
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
        Input::Advance { ms: Millis(2_000) },
        Input::Close { drain: false },
        Input::Advance { ms: Millis(20) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let interrupts = run
        .record
        .journal()
        .iter()
        .filter(|(_, step)| {
            matches!(
                step,
                Step::Realtime(RealtimeStep::Sent { kind: RealtimeCommandKind::Interrupt, .. })
            )
        })
        .count();
    assert_eq!(interrupts, 1);
}
