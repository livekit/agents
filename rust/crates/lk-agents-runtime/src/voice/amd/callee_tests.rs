use super::*;
use parking_lot::Mutex;

// Catches: a session without room I/O never starting to listen, as Python starts at once there.
#[tokio::test(start_paused = true)]
async fn without_a_room_the_run_listens_at_once() {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let report: Report = {
        let heard = heard.clone();
        Box::new(move |callee| {
            heard.lock().push(callee);
            true
        })
    };
    let setup =
        CalleeSetup { room: None, participant: None, linked: None, wait_until_answered: true };
    let time = Timebase::from_virtual(Arc::new(lk_agents_abi::VirtualClock::new()));
    watch(setup, time, CancellationToken::new(), report).await;
    assert_eq!(*heard.lock(), [Callee::Answered]);
}
