use super::*;
use lk_agents_abi::llm::realtime::{GenerateReply, UpdateChatCtx};
use lk_agents_abi::{ChatMessage, RealClock, Role, TokioSpawner};
use lk_agents_inference::realtime::OPENAI;
use lk_agents_inference::testkit::{FakeWs, NoImages};

fn pump() -> (Pump, Arc<Shared>, super::super::queue::EventReceiver) {
    let clock = Arc::new(RealClock::new());
    let options = SessionOptions::from_dialect(&OPENAI, false);
    let caps = OPENAI.capabilities(options.turn_detection.as_ref(), false, true, true);
    let mut machine = RealtimeMachine::new(OPENAI, options, caps, "cleanup_".into());
    machine.open(clock.now());
    machine.command(
        clock.now(),
        Command::UpdateChatCtx(UpdateChatCtx {
            chat_ctx: ChatContext::new(
                (0..10_000)
                    .map(|id| {
                        ChatMessage::text(
                            id.to_string(),
                            Role::User,
                            "retained history",
                            std::time::UNIX_EPOCH,
                        )
                        .into()
                    })
                    .collect(),
            ),
        }),
    );
    machine.command(
        clock.now(),
        Command::GenerateReply(GenerateReply {
            req_id: "closing".into(),
            instructions: None,
            tool_choice: None,
        }),
    );
    let shared = Arc::new(Shared {
        machine: Mutex::new(machine),
        caps: std::sync::Mutex::new(caps),
        closing: tokio::sync::Notify::new(),
        cleanup: std::sync::Mutex::new(None),
        retirement: std::sync::Mutex::new(None),
        retirement_wake: Arc::new(tokio::sync::Notify::new()),
        retirement_closing: std::sync::atomic::AtomicBool::new(false),
    });
    let (events, receiver) = super::super::queue::events();
    let pump = Pump {
        shared: shared.clone(),
        url: "https://unused.test".into(),
        connect: Arc::new(Connect::openai("test", "https://unused.test")),
        shape: WireShape::default(),
        transport: Arc::new(FakeWs::new(vec![])),
        clock,
        provider_label: "openai",
        label: "cleanup".into(),
        events,
        images: Arc::new(NoImages),
        spawner: Arc::new(TokioSpawner::current()),
    };
    (pump, shared, receiver)
}

#[tokio::test]
async fn cancelling_close_keeps_the_runtime_owner_until_context_retirement_settles() {
    let (pump, shared, mut receiver) = pump();
    let closing = tokio::spawn(async move {
        pump.close().await;
        std::future::pending::<()>().await;
    });
    assert!(matches!(
        receiver.next().await,
        Some(Event::GenerateReplyFailed(failed)) if failed.req_id == "closing"
    ));
    closing.abort();
    let _ = closing.await;
    let mut done = shared.retirement.lock().unwrap().clone().expect("runtime cleanup owner");
    tokio::time::timeout(Duration::from_secs(5), done.wait_for(|settled| *settled))
        .await
        .expect("cancelling close abandoned the remaining context retirement")
        .expect("runtime cleanup lost its completion receipt");
    assert!(!shared.machine.lock().await.has_context_retirement());
}

#[tokio::test]
async fn refused_admission_retains_all_owners_after_the_last_caller_leaves() {
    use crate::media_workers::{MediaWorkerClass, MediaWorkerLimits, MediaWorkers};
    let workers = MediaWorkers::new(
        tokio::runtime::Handle::current(),
        MediaWorkerLimits { formatting: 1, ..Default::default() },
    )
    .unwrap();
    let held = workers.try_admit(MediaWorkerClass::Formatting).unwrap();
    let (pump, shared, _) = pump();
    shared.machine.lock().await.close(pump.clock.now());
    shared.retirement_closing.store(true, std::sync::atomic::Ordering::Release);
    let weak = Arc::downgrade(&shared);
    let (settled, mut done) = tokio::sync::watch::channel(false);
    *shared.retirement.lock().unwrap() = Some(done.clone());
    let cleanup = tokio::spawn(super::retirement::run(
        weak.clone(),
        shared.retirement_wake.clone(),
        pump.clock.clone(),
        pump.label.clone(),
        settled,
        Some(workers),
    ));
    for _ in 0..16 {
        tokio::task::yield_now().await;
    }
    assert!(
        shared.machine.lock().await.has_context_retirement(),
        "refused admission transferred destructor-bearing state out of the core"
    );
    drop(pump);
    drop(shared);
    assert!(weak.upgrade().is_some(), "the runtime abandoned the blocked cleanup owner");
    drop(held);
    tokio::time::timeout(Duration::from_secs(5), done.wait_for(|settled| *settled))
        .await
        .unwrap()
        .unwrap();
    cleanup.await.unwrap();
    assert!(weak.upgrade().is_none(), "settled cleanup retained the session");
}
