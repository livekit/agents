use super::*;
use lk_agents_net::clock::RealClock;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

type Run = dyn Fn(
        Arc<dyn HandlerPort>,
        Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'static, Result<HandlerOutput, ProviderError>>
    + Send
    + Sync;

struct Script(Box<Run>);

impl TurnHost for Script {
    fn handle<'a>(
        &'a self,
        _cx: &crate::telemetry::Cx,
        _ctx: &'a lk_agents_abi::a2a::turn::TurnContext,
        turn: Arc<dyn HandlerPort>,
        cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'a, Result<HandlerOutput, ProviderError>> {
        (self.0)(turn, cancel)
    }
}

fn open(
    run: impl Fn(
        Arc<dyn HandlerPort>,
        Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'static, Result<HandlerOutput, ProviderError>>
    + Send
    + Sync
    + 'static,
) -> Handler {
    let ctx = lk_agents_abi::a2a::turn::TurnContext {
        thread_id: "AT_1".into(),
        lane: "desk".into(),
        message: lk_agents_abi::ChatItem::Message(lk_agents_abi::ChatMessage::text(
            "message",
            lk_agents_abi::Role::User,
            "",
            std::time::SystemTime::UNIX_EPOCH,
        )),
        batch: Vec::new(),
        delegated: false,
        history: Vec::new(),
        state: None,
        context_scope_id: String::new(),
        context_scopes: Vec::new(),
        read_only: false,
        task_id: "task".into(),
        tasks: Vec::new(),
        caller_history: None,
        answer_tool: None,
        host_context: None,
    };
    Handler::open(Arc::new(Script(Box::new(run))), crate::telemetry::Cx::NONE, ctx)
}

fn progress(text: &str) -> TurnEvent {
    TurnEvent::Progress { task_id: "task".into(), text: text.into() }
}

// Catches: a report overtaken by the output, or a verdict not reaching the checkpoint.
#[tokio::test(start_paused = true)]
async fn reports_arrive_in_order_and_the_output_after_them() {
    let mut handler = open(|turn, _| {
        Box::pin(async move {
            turn.progress("working".into()).await;
            turn.token("reply".into()).await;
            let interrupted = turn.checkpoint().await;
            turn.progress("after".into()).await;
            Ok(HandlerOutput { interrupted, ..Default::default() })
        })
    });
    assert!(matches!(handler.next().await, Some(Read::Event(e)) if e == progress("working")));
    let token = TurnEvent::Token { task_id: "task".into(), delta: "reply".into() };
    assert!(matches!(handler.next().await, Some(Read::Event(e)) if e == token));
    assert!(matches!(handler.next().await, Some(Read::Checkpoint)));
    handler.answer(true);
    assert!(matches!(handler.next().await, Some(Read::Event(e)) if e == progress("after")));
    assert!(
        matches!(handler.next().await, Some(Read::Output(out)) if matches!(*out, Ok(ref out) if out.interrupted))
    );
    assert!(handler.next().await.is_none(), "the output is read once");
    assert!(handler.close(&RealClock::new(), Ts::from_nanos(u64::MAX)).await);
}

// Catches: an unbounded queue letting a handler run ahead of the turn.
#[tokio::test(start_paused = true)]
async fn a_full_queue_holds_the_handler_until_the_turn_reads() {
    let sent = Arc::new(AtomicUsize::new(0));
    let counted = sent.clone();
    let mut handler = open(move |turn, _| {
        let sent = counted.clone();
        Box::pin(async move {
            for n in 0..100 {
                turn.progress(format!("{n}")).await;
                sent.fetch_add(1, Ordering::Relaxed);
            }
            Ok(HandlerOutput::default())
        })
    });
    assert!(matches!(handler.next().await, Some(Read::Event(e)) if e == progress("0")));
    assert_eq!(sent.load(Ordering::Relaxed), REPORT_QUEUE);
    for n in 1..100 {
        let expected = progress(&format!("{n}"));
        assert!(matches!(handler.next().await, Some(Read::Event(e)) if e == expected));
    }
    assert!(matches!(handler.next().await, Some(Read::Output(out)) if out.is_ok()));
}

// Catches: a close that abandons the handler's cleanup, or waits past the teardown bound.
#[tokio::test(start_paused = true)]
async fn closing_cancels_the_handler_and_waits_for_its_cleanup_within_the_deadline() {
    for (cleanup, settles) in [(Duration::from_secs(1), true), (Duration::from_secs(10), false)] {
        let cleaned = Arc::new(AtomicBool::new(false));
        let flag = cleaned.clone();
        let mut handler = open(move |turn, cancel| {
            let cleaned = flag.clone();
            Box::pin(async move {
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => {
                        tokio::time::sleep(cleanup).await;
                        cleaned.store(true, Ordering::Relaxed);
                    }
                    _ = turn.checkpoint() => panic!("a closed checkpoint resumed"),
                }
                Ok(HandlerOutput::default())
            })
        });
        assert!(matches!(handler.next().await, Some(Read::Checkpoint)));
        let clock = RealClock::new();
        let deadline = clock.now().saturating_add(Duration::from_secs(5));
        assert_eq!(handler.close(&clock, deadline).await, settles);
        assert_eq!(cleaned.load(Ordering::Relaxed), settles);
    }
}

// Catches: a report from a task the handler left behind blocking, or a checkpoint after the
// turn ended letting the work go on.
#[tokio::test(start_paused = true)]
async fn reports_after_the_turn_closed_are_dropped_and_checkpoints_yield() {
    let kept: Arc<Mutex<Option<Arc<dyn HandlerPort>>>> = Arc::default();
    let slot = kept.clone();
    let mut handler = open(move |turn, _| {
        *slot.lock() = Some(turn);
        Box::pin(async { Ok(HandlerOutput::default()) })
    });
    assert!(matches!(handler.next().await, Some(Read::Output(out)) if out.is_ok()));
    assert!(handler.close(&RealClock::new(), Ts::from_nanos(u64::MAX)).await);
    let turn = kept.lock().take().expect("the handler ran");
    for n in 0..2 * REPORT_QUEUE {
        turn.progress(format!("{n}")).await;
    }
    assert!(turn.checkpoint().await);
}
