//! A realtime request settled before the server answered ends its handle through the journal.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};
use lk_agents_fuzz::record::Recording;
use lk_agents_voice::trace::{InterruptCause, InterruptStep, Step};

fn realtime(inputs: &[Input]) -> Recording {
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
        ..Setup::default()
    };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in inputs {
            run.apply(input).await;
        }
        run
    });
    run.record
}

fn interrupted(r: &Recording) -> Vec<(u64, InterruptCause)> {
    r.journal()
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Interrupt(InterruptStep::Interrupted { speech, cause, .. }) => {
                Some((speech.0, *cause))
            }
            _ => None,
        })
        .collect()
}

// Catches: a `generate_reply()` still waiting on the server when an interruption cancels the
// server's turn, answered with a handle that reports itself interrupted while the journal never
// said so (`interrupted_handle_is_journalled`, survey seed 14).
#[test]
fn an_interrupted_request_is_journalled_with_its_cause() {
    let r = realtime(&[
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Interrupt { force: true, wait: false },
        Input::Advance { ms: Millis(20) },
    ]);
    let speeches = r.host_view().speeches();
    let [speech] = &speeches[..] else { panic!("one handle, got {}", speeches.len()) };
    assert!(speech.is_interrupted());
    assert_eq!(interrupted(&r), [(speech.id, InterruptCause::Host)]);
}

// Catches: a request made after the socket ended answered with an interrupted handle no
// interruption journalled. The reference's pending `generate_reply()` fails with a
// `RealtimeError` there, which `_realtime_reply_task` logs, as it does a timeout. A real session
// reconnects; a server that never takes the connection again ends it with the session's error
// close, and the request with it, journalled.
#[test]
fn a_request_after_the_socket_ended_fails() {
    let r = realtime(&[
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::Disconnected },
        Input::Advance { ms: Millis(20) },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(30_000) },
    ]);
    let speeches = r.host_view().speeches();
    let [speech] = &speeches[..] else { panic!("one handle, got {}", speeches.len()) };
    assert!(speech.is_interrupted());
    assert_eq!(interrupted(&r), [(speech.id, InterruptCause::Closing)]);
}
