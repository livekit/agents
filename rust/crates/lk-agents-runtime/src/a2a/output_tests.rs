use super::super::{Read, tests::open};
use lk_agents_abi::a2a::turn::HandlerOutput;
use lk_agents_abi::{ChatMessage, Role, Ts};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn words(index: usize) -> ChatMessage {
    ChatMessage::text(
        format!("exact-{index}"),
        Role::Assistant,
        format!("Reference {index}."),
        std::time::UNIX_EPOCH,
    )
}

#[tokio::test]
async fn an_exact_output_receipt_waits_for_handoff_instead_of_handler_queue_admission() {
    let confirmed = Arc::new(AtomicBool::new(false));
    let observed = confirmed.clone();
    let mut handler = open(move |turn, _| {
        let confirmed = confirmed.clone();
        Box::pin(async move {
            turn.say(words(1)).await.unwrap();
            confirmed.store(true, Ordering::SeqCst);
            Ok(HandlerOutput::default())
        })
    });
    let Some(Read::Exact(output)) = handler.next().await else { panic!("exact report expected") };
    assert_eq!(output.task_id, "task");
    assert_eq!(*output.message, words(1));
    assert!(!observed.load(Ordering::SeqCst), "admission was incorrectly counted as handoff");
    output.delivered.resolve(Ok(()));
    assert!(matches!(handler.next().await, Some(Read::Output(_))));
    assert!(observed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn unawaited_exact_outputs_keep_order_and_precede_their_handlers_final_result() {
    let mut handler = open(|turn, _| {
        Box::pin(async move {
            for index in 0..100 {
                drop(turn.say(words(index)));
            }
            Ok(HandlerOutput::default())
        })
    });
    for index in 0..100 {
        let Some(Read::Exact(output)) = handler.next().await else {
            panic!("an unawaited exact output was discarded or overtaken by its final result")
        };
        assert_eq!(*output.message, words(index));
        output.delivered.resolve(Ok(()));
    }
    assert!(matches!(handler.next().await, Some(Read::Output(_))));
    assert!(handler.next().await.is_none());
}

#[tokio::test]
async fn closing_the_handler_settles_unread_exact_receipts_before_awaiting_its_host() {
    let pending = Arc::new(Mutex::new(None));
    let held = pending.clone();
    let mut handler = open(move |turn, cancel| {
        *pending.lock() = Some(turn.say(words(1)));
        Box::pin(async move {
            cancel.cancelled().await;
            Ok(HandlerOutput::default())
        })
    });
    {
        let next = handler.next();
        tokio::pin!(next);
        assert!(futures_util::poll!(next.as_mut()).is_pending());
    }
    let receipt = held.lock().take().expect("the host immediately enqueued its exact output");
    assert!(handler.close(&lk_agents_net::clock::RealClock::new(), Ts::from_nanos(u64::MAX)).await);
    let error = receipt.await.unwrap_err();
    assert_ne!(
        error.kind,
        lk_agents_abi::errors::HostErrorKind::Cancelled,
        "closing presentation cannot manufacture a business-task cancellation"
    );
}
