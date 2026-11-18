use super::*;
use lk_agents_fuzz::drive::simulate;
use lk_agents_fuzz::fakes::a2a_peer::{PeerScript, question, result};

/// The frontend delegates `request`, then lets `ms` pass.
async fn delegate(run: &mut Run, call: &mut Call, request: &str, ms: u64) {
    call.next_call = Some(request.into());
    call.prompt(run).await;
    call.advance(run, ms).await;
}

/// The person answers with `words`, and the frontend forwards them through the remote tool.
async fn answer(run: &mut Run, call: &mut Call, words: &str, ms: u64) {
    call.say(run, words).await;
    call.next_call = Some(words.into());
    call.advance(run, ms).await;
}

fn count(steps: &[RemoteStep], want: impl Fn(&RemoteStep) -> bool) -> usize {
    steps.iter().filter(|step| want(step)).count()
}

fn print(steps: &[RemoteStep]) {
    for step in steps {
        println!("{step:?}");
    }
}

// Catches: a remote question handed to the session again when a later progress burst, an answer
// or the next question replays the task's state, so the person is asked twice.
#[test]
fn each_remote_question_is_presented_once_through_progress_bursts() {
    simulate(async {
        let script = PeerScript { asks: 2, progress: 4, work_ms: 150, ..PeerScript::default() };
        let mut run = Run::start(&frontend(script, true), 0, Default::default()).await;
        let mut call = Call::new();
        delegate(&mut run, &mut call, "Book a table for two", 3_000).await;
        answer(&mut run, &mut call, "Friday", 3_000).await;
        answer(&mut run, &mut call, "Eight in the evening", 3_000).await;
        let failures = call.hang_up(&mut run).await;
        let steps = steps(&run);
        print(&steps);
        for k in 1..=2 {
            let asked = question("task-1", k);
            assert_eq!(
                count(&steps, |s| matches!(s, RemoteStep::Presented { text, .. } if *text == asked)),
                1,
                "question {k} was not presented exactly once"
            );
        }
        assert!(remote_failures(&failures).is_empty(), "{failures:?}");
    });
}

// Catches: an answer whose acknowledgement was lost after the peer applied it being sent again,
// so the remote booking receives the person's reply twice.
#[test]
fn a_reply_whose_acknowledgement_was_lost_reaches_the_peer_once() {
    simulate(async {
        let script = PeerScript { asks: 1, lost_ack: true, ..PeerScript::default() };
        let mut run = Run::start(&frontend(script, true), 0, Default::default()).await;
        let mut call = Call::new();
        delegate(&mut run, &mut call, "Book a table for two", 3_000).await;
        answer(&mut run, &mut call, "Friday", 3_000).await;
        // The frontend tries again with the same words; the peer already has them.
        call.next_call = Some("Friday".into());
        call.prompt(&mut run).await;
        call.advance(&mut run, 3_000).await;
        let failures = call.hang_up(&mut run).await;
        let steps = steps(&run);
        print(&steps);
        assert!(steps.iter().any(|s| matches!(s, RemoteStep::AckLost { .. })));
        assert_eq!(
            count(&steps, |s| matches!(s, RemoteStep::Received { text, .. } if text == "Friday")),
            1,
            "the person's reply reached the peer more than once"
        );
        assert!(remote_failures(&failures).is_empty(), "{failures:?}");
    });
}

// Catches: the task's result resolving the call twice, or reaching the later call that only
// forwarded the person's answer.
#[test]
fn the_result_resolves_its_original_call_once() {
    simulate(async {
        let script = PeerScript { asks: 1, progress: 1, work_ms: 200, ..PeerScript::default() };
        let mut run = Run::start(&frontend(script, true), 0, Default::default()).await;
        let mut call = Call::new();
        delegate(&mut run, &mut call, "Book a table for two", 3_000).await;
        answer(&mut run, &mut call, "Friday", 5_000).await;
        let failures = call.hang_up(&mut run).await;
        let steps = steps(&run);
        print(&steps);
        let original = steps
            .iter()
            .find_map(|s| match s {
                RemoteStep::BodyStarted { call } => Some(call.clone()),
                _ => None,
            })
            .unwrap();
        let booked = result("task-1");
        let returned: Vec<_> = steps
            .iter()
            .filter_map(|s| match s {
                RemoteStep::Returned { call, outcome: Some(o) } if o.contains(&booked) => {
                    Some(call.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(returned, [original], "the result did not return once, to its own call");
        assert!(remote_failures(&failures).is_empty(), "{failures:?}");
    });
}

// Catches: hanging up while the desk works cancelling the remote task: ending the call stops the
// local wait only.
#[test]
fn hanging_up_leaves_the_remote_task_running() {
    simulate(async {
        let script = PeerScript { asks: 0, progress: 2, work_ms: 20_000, ..PeerScript::default() };
        let mut run = Run::start(&frontend(script, false), 0, Default::default()).await;
        let mut call = Call::new();
        delegate(&mut run, &mut call, "Book a table for two", 3_000).await;
        let failures = call.hang_up(&mut run).await;
        let steps = steps(&run);
        print(&steps);
        assert!(steps.iter().any(|s| matches!(s, RemoteStep::Returned { outcome: None, .. })));
        assert_eq!(count(&steps, |s| matches!(s, RemoteStep::CancelReceived { .. })), 0);
        assert!(remote_failures(&failures).is_empty(), "{failures:?}");
    });
}

// Catches: a delegation still waiting on its response stream after the session exited: every
// body settles and every stream it opened at the peer is closed.
#[test]
fn hanging_up_settles_the_delegation_and_closes_its_stream() {
    simulate(async {
        let script = PeerScript { asks: 1, progress: 3, work_ms: 5_000, ..PeerScript::default() };
        let mut run = Run::start(&frontend(script, false), 0, Default::default()).await;
        let mut call = Call::new();
        delegate(&mut run, &mut call, "Book a table for two", 1_000).await;
        let failures = call.hang_up(&mut run).await;
        let steps = steps(&run);
        print(&steps);
        let opened = count(&steps, |s| matches!(s, RemoteStep::StreamOpened { .. }));
        assert!(opened > 0);
        assert_eq!(count(&steps, |s| matches!(s, RemoteStep::StreamClosed { .. })), opened);
        assert_eq!(
            count(&steps, |s| matches!(s, RemoteStep::BodySettled { .. })),
            count(&steps, |s| matches!(s, RemoteStep::BodyStarted { .. }))
        );
        assert!(remote_failures(&failures).is_empty(), "{failures:?}");
    });
}
