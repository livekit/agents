use super::*;
use crate::realtime::OPENAI;
use crate::testkit::{FakeWs, NoImages};
use lk_agents_abi::llm::realtime::{GenerateReply, UpdateChatCtx};
use lk_agents_abi::{ChatMessage, RealClock, Role};
use lk_agents_net::task::TokioSpawner;

fn pump() -> (Pump, Arc<Shared>, super::super::queue::EventReceiver) {
    let clock = Arc::new(RealClock::new());
    let options = SessionOptions::from_dialect(&OPENAI, false);
    let caps = OPENAI.capabilities(options.turn_detection.as_ref(), false, true, true);
    let mut machine = RealtimeMachine::new(OPENAI, options, caps, "cleanup_".into())
        .with_response_timeout(Some(Duration::from_millis(50)));
    machine.open(clock.now());
    // A previously established conversation: cleanup tests begin while the next live socket
    // already has a substantial acknowledged history.
    for id in 0..10_000usize {
        machine.incoming_on(
            machine.connection_generation(),
            clock.now(),
            crate::realtime::Incoming::ItemAdded {
                previous_item_id: id.checked_sub(1).map(|id: usize| id.to_string()),
                item: ChatMessage::text(
                    id.to_string(),
                    Role::User,
                    "retained history",
                    std::time::UNIX_EPOCH,
                )
                .into(),
            },
        );
    }
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
        context_ready: tokio::sync::Notify::new(),
        retirement_closing: std::sync::atomic::AtomicBool::new(false),
        media_pool: crate::testkit::media_pool,
        disposal: Arc::default(),
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
    use lk_agents_media::pool::{JobClass, MediaPool, MediaPoolLimits};
    let media_pool =
        MediaPool::new(MediaPoolLimits { formatting: 1, ..Default::default() }).unwrap();
    let held = media_pool.try_admit(JobClass::Formatting).unwrap();
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
        pump.events.clone(),
        settled,
        Ok(media_pool),
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

#[tokio::test]
async fn permanent_worker_startup_failure_settles_cleanup_without_retaining_the_session() {
    let (pump, shared, mut receiver) = pump();
    for _ in 0..2 {
        shared.machine.lock().await.command(
            pump.clock.now(),
            Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: ChatContext::empty() }),
        );
    }
    let weak = Arc::downgrade(&shared);
    let (settled, mut done) = tokio::sync::watch::channel(false);
    *shared.retirement.lock().unwrap() = Some(done.clone());
    let cleanup = tokio::spawn(super::retirement::run(
        weak.clone(),
        shared.retirement_wake.clone(),
        pump.clock.clone(),
        pump.label.clone(),
        pump.events.clone(),
        settled,
        Err("forked".to_owned()),
    ));
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if shared.cleanup.lock().unwrap().is_some() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the owner did not observe the injected startup failure");
    assert!(
        matches!(tokio::time::timeout(Duration::from_secs(5), receiver.next()).await.unwrap(),
        Some(Event::GenerateReplyFailed(failed)) if failed.req_id == "closing"),
        "worker startup failure must refuse the session's pending caller"
    );
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(5), receiver.next())
            .await
            .expect("terminal worker startup failure did not explain the session close"),
        Some(Event::Error(error))
            if !error.recoverable && error.message.contains("cleanup service is unavailable")
    ));
    drop(pump);
    drop(shared);
    if tokio::time::timeout(Duration::from_millis(250), done.wait_for(|settled| *settled))
        .await
        .is_err()
    {
        cleanup.abort();
        let _ = cleanup.await;
        panic!("permanent startup failure kept retrying cleanup instead of settling");
    }
    cleanup.await.unwrap();
    assert!(weak.upgrade().is_none(), "permanent startup failure retained the session owner");
}

#[tokio::test]
async fn reconnect_and_live_commands_survive_asynchronous_cleanup_backpressure() {
    use lk_agents_abi::llm::realtime::{CancelReply, Interrupt, PushAudio};
    use lk_agents_media::pool::{JobClass, MediaPool, MediaPoolLimits};
    use lk_agents_net::transport::WsMessage;

    let media_pool =
        MediaPool::new(MediaPoolLimits { formatting: 1, ..Default::default() }).unwrap();
    let held = media_pool.try_admit(JobClass::Formatting).unwrap();
    let (mut pump, shared, mut receiver) = pump();
    shared
        .machine
        .lock()
        .await
        .command(pump.clock.now(), Command::CancelReply(CancelReply { req_id: "closing".into() }));
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    pump.transport = ws.clone();
    let (settled, done) = tokio::sync::watch::channel(false);
    *shared.retirement.lock().unwrap() = Some(done);
    let cleanup = tokio::spawn(super::retirement::run(
        Arc::downgrade(&shared),
        shared.retirement_wake.clone(),
        pump.clock.clone(),
        pump.label.clone(),
        pump.events.clone(),
        settled,
        Ok(media_pool.clone()),
    ));
    let (commands, incoming) = mpsc::channel(128);
    let running = tokio::spawn(pump.run(incoming));
    wait_for_wire(&ws, "session update", |frames| {
        frames.iter().any(|frame| frame["type"] == "session.update")
    })
    .await;
    ws.feed(Ok(WsMessage::Closed));
    tokio::time::timeout(Duration::from_secs(5), async {
        while ws.connect_count() < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    ws.feed(crate::testkit::text(serde_json::json!({
        "type": "input_audio_buffer.speech_started", "item_id": "speaker",
    })));
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match receiver.next().await {
                Some(Event::InputSpeechStarted(_)) => break,
                Some(Event::Error(error)) => assert!(
                    error.recoverable,
                    "ordinary cleanup admission closed the session: {}",
                    error.message
                ),
                Some(_) => {}
                None => panic!("cleanup backpressure ended the session before its control event"),
            }
        }
    })
    .await
    .expect("incoming controls were blocked by cleanup admission");
    assert_eq!(ws.connect_count(), 2, "cleanup admission caused a reconnect storm");
    assert!(shared.machine.lock().await.context_pages_pending());
    assert!(shared.machine.lock().await.has_context_retirement());
    drop(held);

    wait_for_wire(&ws, "10k reconnect replay", |frames| {
        frames.iter().filter(|frame| frame["type"] == "conversation.item.create").count() == 10_000
    })
    .await;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let machine = shared.machine.lock().await;
            if !machine.context_pages_pending() && !machine.has_context_retirement() {
                break;
            }
            drop(machine);
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("cleanup completion failed to wake the parked context continuation");

    let held = loop {
        match media_pool.try_admit(JobClass::Formatting) {
            Ok(permit) => break permit,
            Err(_) => media_pool.wait_for_capacity(JobClass::Formatting).await,
        }
    };
    let target = ChatContext::new(
        (0..10_000usize)
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
    );
    commands
        .send(Cmd::Core(Command::UpdateChatCtx(UpdateChatCtx { chat_ctx: target })))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let machine = shared.machine.lock().await;
            if !machine.context_pages_pending() && machine.has_context_retirement() {
                break;
            }
            drop(machine);
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("an identical update did not finish preparation while cleanup was withheld");

    commands
        .send(Cmd::Core(Command::GenerateReply(GenerateReply {
            req_id: "live".into(),
            instructions: None,
            tool_choice: None,
        })))
        .await
        .unwrap();
    wait_for_wire(&ws, "live reply", |frames| {
        frames.iter().any(|frame| {
            frame["type"] == "response.create"
                && frame["response"]["metadata"]["client_event_id"] == "live"
        })
    })
    .await;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match receiver.next().await {
                Some(Event::GenerateReplyFailed(failed)) if failed.req_id == "live" => break,
                Some(Event::Error(error)) => {
                    assert!(error.recoverable, "cleanup blocked a core timer: {}", error.message)
                }
                Some(_) => {}
                None => panic!("cleanup ended the live response stream"),
            }
        }
    })
    .await
    .expect("a written reply's deadline was blocked by cleanup admission");
    ws.feed(crate::testkit::text(serde_json::json!({
        "type": "response.created", "response": {"id": "control-response"},
    })));
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if matches!(receiver.next().await, Some(Event::GenerationCreated(created)) if created.response_id == "control-response") { break; }
        }
    }).await.unwrap();
    commands.send(Cmd::Core(Command::Interrupt(Interrupt { response_id: None }))).await.unwrap();
    wait_for_wire(&ws, "interrupt", |frames| {
        frames.iter().any(|frame| frame["type"] == "response.cancel")
    })
    .await;
    // Keep each burst below the independent 32-batch socket write backlog. The writer
    // aggregates five 20 ms frames into one 100 ms append; all 140 commands must proceed
    // while cleanup is withheld, without accumulating in the core's deferred-media queue.
    for burst in 1..=7 {
        for _ in 0..20 {
            commands
                .send(Cmd::Core(Command::PushAudio(PushAudio {
                    frame: lk_agents_abi::AudioFrame {
                        data: vec![0u8; 960].into(),
                        sample_rate: 24_000,
                        num_channels: 1,
                        samples_per_channel: 480,
                    },
                })))
                .await
                .unwrap();
        }
        wait_for_wire(&ws, "audio", |frames| {
            frames.iter().filter(|frame| frame["type"] == "input_audio_buffer.append").count()
                == burst * 4
        })
        .await;
    }
    assert_eq!(ws.connect_count(), 2, "ordinary cleanup forced another socket acquisition");
    drop(held);
    commands.send(Cmd::Close).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), running).await.unwrap().unwrap();
    cleanup.await.unwrap();
}

async fn wait_for_wire(ws: &FakeWs, stage: &str, predicate: impl Fn(&[serde_json::Value]) -> bool) {
    let mut observed = Vec::new();
    let result = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let frames = ws.sent();
            observed.extend(
                frames[observed.len()..]
                    .iter()
                    .map(|frame| serde_json::from_str(frame).expect("a JSON frame")),
            );
            if predicate(&observed) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    if result.is_err() {
        let frames = ws.sent_json();
        let creates = frames.iter().filter(|f| f["type"] == "conversation.item.create").count();
        panic!(
            "missing {stage}: {} frames, {creates} creates; last: {:?}",
            frames.len(),
            frames
                .iter()
                .rev()
                .take(3)
                .map(|frame| (&frame["type"], &frame["item"]["id"]))
                .collect::<Vec<_>>()
        );
    }
}
