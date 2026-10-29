use super::*;
use lk_agents_abi::llm::realtime::{GenerationCreated, MessageStarted};

/// A scripted session: events to hand out, then parked; every command recorded.
struct Scripted {
    events: Mutex<VecDeque<Event>>,
    sent: Mutex<Vec<String>>,
    closed: std::sync::atomic::AtomicBool,
}

impl RealtimeSession for Scripted {
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities { audio_output: true, ..Default::default() }
    }
    fn send(&self, cmd: RealtimeCommand) {
        let label = match &cmd {
            RealtimeCommand::GenerateReply(g) => format!("generate_reply:{}", g.req_id),
            other => format!("{other:?}").split('(').next().unwrap_or_default().to_owned(),
        };
        self.sent.lock().unwrap().push(label);
    }
    fn next_event(&self) -> lk_agents_abi::BoxFuture<'_, Option<Event>> {
        Box::pin(async move {
            let next = self.events.lock().unwrap().pop_front();
            match next {
                Some(ev) => Some(ev),
                None => std::future::pending().await,
            }
        })
    }
    fn chat_ctx(&self) -> lk_agents_abi::BoxFuture<'_, ChatContext> {
        Box::pin(async move { ChatContext::default() })
    }
    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            self.closed.store(true, std::sync::atomic::Ordering::Release);
        })
    }
}

struct Model(Arc<Scripted>);

impl Realtime for Model {
    fn label(&self) -> String {
        "scripted".into()
    }
    fn capabilities(&self) -> RealtimeCapabilities {
        RealtimeCapabilities { audio_output: true, ..Default::default() }
    }
    fn session(
        &self,
        _td: bool,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
        Box::pin(async move { Ok(Arc::clone(&self.0) as Arc<dyn RealtimeSession>) })
    }
}

fn scripted(events: Vec<Event>) -> Arc<Scripted> {
    Arc::new(Scripted {
        events: Mutex::new(events.into()),
        sent: Mutex::default(),
        closed: Default::default(),
    })
}

fn fatal() -> Event {
    Event::Error(RealtimeModelError {
        timestamp: lk_agents_abi::Ts::ZERO,
        label: "scripted".into(),
        message: "boom".into(),
        recoverable: false,
        req_id: None,
        code: None,
        provider_error: None,
    })
}

fn kind(ev: &Event) -> String {
    format!("{ev:?}")
        .split('(')
        .next()
        .unwrap_or_default()
        .split(' ')
        .next()
        .unwrap_or_default()
        .to_owned()
}

// Catches: a fatal child error passed on as fatal while a fallback exists, the dead child's
// open generation left open, a swap reported as a reconnect (the driver then gives the new
// model nothing), or a request reaching the new model before its frame.
#[tokio::test]
async fn a_fatal_error_mid_reply_swaps_to_the_next_model_and_asks_for_the_reply_again() {
    let primary = scripted(vec![
        Event::GenerationCreated(GenerationCreated { response_id: "r1".into(), req_id: None }),
        Event::MessageStarted(MessageStarted {
            message_id: "m1".into(),
            response_id: Some("r1".into()),
        }),
        fatal(),
    ]);
    let backup = scripted(vec![]);
    let changes = Arc::new(Mutex::new(Vec::new()));
    struct Record(Arc<Mutex<Vec<(usize, bool)>>>);
    impl ChainObserver for Record {
        fn availability_changed(&self, ev: AvailabilityChanged) {
            self.0.lock().unwrap().push((ev.index, ev.available));
        }
    }
    let model = RealtimeFallbackModel::new(
        vec![
            ChainModel::Native(Arc::new(Model(Arc::clone(&primary)))),
            ChainModel::Native(Arc::new(Model(Arc::clone(&backup)))),
        ],
        10.0,
        true,
        Arc::new(Record(Arc::clone(&changes))),
        Arc::new(lk_agents_abi::RealClock::default()),
    )
    .unwrap();
    let session = model.session(false).await.unwrap();

    let mut kinds = Vec::new();
    for _ in 0..7 {
        let ev = session.next_event().await.unwrap();
        if let Event::Error(e) = &ev {
            assert!(e.recoverable, "a fallback remains: the error is rewritten recoverable");
        }
        kinds.push(kind(&ev));
    }
    assert_eq!(
        kinds,
        [
            "GenerationCreated",
            "MessageStarted",
            "Error",
            "MessageModalities",
            "MessageDone",
            "GenerationDone",
            "SessionSwapped"
        ]
    );
    assert!(primary.closed.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(*changes.lock().unwrap(), [(0, false)]);
    assert_eq!(model.active_index(), 1);

    // A reply asked before the driver gave the frame waits for it; the tools end the frame.
    let ask = |id: &str| {
        RealtimeCommand::GenerateReply(lk_agents_abi::GenerateReply {
            req_id: id.into(),
            instructions: None,
            tool_choice: None,
        })
    };
    session.send(ask("early"));
    session.send(RealtimeCommand::Interrupt(lk_agents_abi::llm::realtime::Interrupt {
        response_id: Some("r1".into()),
    }));
    assert!(
        backup.sent.lock().unwrap().is_empty(),
        "nothing reaches the new model before its frame"
    );
    session.send(RealtimeCommand::UpdateInstructions(lk_agents_abi::UpdateInstructions {
        instructions: "hi".into(),
    }));
    session.send(RealtimeCommand::UpdateTools(lk_agents_abi::UpdateTools { tools: vec![] }));
    session.send(ask("late"));
    let sent = backup.sent.lock().unwrap().clone();
    assert_eq!(
        sent,
        ["UpdateInstructions", "UpdateTools", "generate_reply:early", "generate_reply:late"]
    );
}
#[derive(Default)]
struct AvailabilityLog(Mutex<Vec<(usize, bool)>>);
impl ChainObserver for AvailabilityLog {
    fn availability_changed(&self, ev: AvailabilityChanged) {
        self.0.lock().unwrap().push((ev.index, ev.available));
    }
}

async fn cancellation_case(failed_open: bool) {
    struct CancelledOpen;
    impl Realtime for CancelledOpen {
        fn label(&self) -> String {
            "cancelled-open".into()
        }
        fn capabilities(&self) -> RealtimeCapabilities {
            RealtimeCapabilities { audio_output: true, ..Default::default() }
        }
        fn session(
            &self,
            _: bool,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
            Box::pin(async {
                Err(lk_agents_abi::errors::ApiError::status("provider cancelled", 499, None, None)
                    .into())
            })
        }
    }
    let mut error = fatal();
    if !failed_open && let Event::Error(e) = &mut error {
        e.provider_error = Some(
            lk_agents_abi::errors::ApiError::status("provider cancelled", 499, None, None).into(),
        );
    }
    let primary = scripted(vec![error]);
    let backup: Arc<dyn Realtime> =
        if failed_open { Arc::new(CancelledOpen) } else { Arc::new(Model(scripted(vec![]))) };
    let changes = Arc::new(AvailabilityLog::default());
    let model = RealtimeFallbackModel::new(
        vec![
            ChainModel::Native(Arc::new(Model(primary.clone()))),
            ChainModel::Native(backup),
        ],
        10.0,
        false,
        changes.clone(),
        Arc::new(lk_agents_abi::RealClock::default()),
    )
    .unwrap();
    let session = model.session(false).await.unwrap();
    let first = session.next_event().await.unwrap();
    let terminal = if failed_open {
        assert!(matches!(first, Event::Error(e) if e.recoverable));
        session.next_event().await.unwrap()
    } else {
        first
    };
    assert_eq!(
        *changes.0.lock().unwrap(),
        if failed_open { vec![(0, false)] } else { vec![] },
        "cancellation must not change the cancelled backend's health"
    );
    let Event::Error(error) = terminal else { panic!("cancellation was retried") };
    assert!(!error.recoverable);
    assert!(
        error
            .provider_error
            .as_ref()
            .is_some_and(|e| e.kind == lk_agents_abi::errors::ApiErrorKind::Cancelled),
        "the cancellation's class and diagnostic cause must survive fallback"
    );
    assert_eq!(model.active_index(), 0, "cancellation must not swap providers");
    if !failed_open {
        assert!(!primary.closed.load(std::sync::atomic::Ordering::Acquire));
    }
    session.aclose().await;
}

#[tokio::test]
async fn a_cancelled_realtime_request_does_not_swap_or_penalize_its_provider() {
    cancellation_case(false).await;
}

#[tokio::test]
async fn a_cancelled_realtime_open_preserves_its_cause_without_a_health_penalty() {
    cancellation_case(true).await;
}

#[tokio::test]
async fn closing_during_a_failed_swap_does_not_publish_a_fatal_cancellation() {
    struct BlockedOpen(Arc<tokio::sync::Notify>, Arc<tokio::sync::Notify>);
    impl Realtime for BlockedOpen {
        fn label(&self) -> String {
            "blocked-open".into()
        }
        fn capabilities(&self) -> RealtimeCapabilities {
            RealtimeCapabilities { audio_output: true, ..Default::default() }
        }
        fn session(
            &self,
            _: bool,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
            Box::pin(async move {
                self.0.notify_one();
                self.1.notified().await;
                Err(lk_agents_abi::errors::ApiError::connection(None).into())
            })
        }
    }
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let changes = Arc::new(AvailabilityLog::default());
    let model = RealtimeFallbackModel::new(
        vec![
            ChainModel::Native(Arc::new(Model(scripted(vec![fatal()])))),
            ChainModel::Native(Arc::new(BlockedOpen(entered.clone(), release.clone()))),
            ChainModel::Native(Arc::new(Model(scripted(vec![])))),
        ],
        10.0,
        false,
        changes.clone(),
        Arc::new(lk_agents_abi::RealClock::default()),
    )
    .unwrap();
    let session = model.session(false).await.unwrap();
    let reading = tokio::spawn({
        let session = session.clone();
        async move { session.next_event().await }
    });
    entered.notified().await;
    session.aclose().await;
    release.notify_one();
    let _ = reading.await.unwrap();
    assert_eq!(
        *changes.0.lock().unwrap(),
        vec![(0, false)],
        "a late failure after close must not announce a health penalty"
    );
    assert!(
        session.next_event().await.is_none(),
        "session close fabricated a fatal cancellation error"
    );
}

#[tokio::test]
async fn repeating_the_same_backend_does_not_create_a_second_fallback_identity() {
    let child: Arc<dyn Realtime> = Arc::new(Model(scripted(vec![fatal()])));
    let model = RealtimeFallbackModel::new(
        vec![ChainModel::Native(child.clone()), ChainModel::Native(child)],
        10.0,
        false,
        Arc::new(crate::fallback::IgnoreEvents),
        Arc::new(lk_agents_abi::RealClock::default()),
    )
    .unwrap();
    let session = model.session(false).await.unwrap();
    assert!(
        matches!(session.next_event().await, Some(Event::Error(e)) if !e.recoverable),
        "one failed backend cannot fall back to another occurrence of itself"
    );
    session.aclose().await;
}

// A failed open is attempted once in one failover, even when it is immediately eligible
// for a later request. The cap turns an accidental infinite swap loop into a test failure.
#[tokio::test]
async fn zero_cooldown_exhausts_one_fallback_pass_without_reopening_failed_children() {
    struct FailsToOpen(std::sync::atomic::AtomicUsize);
    impl Realtime for FailsToOpen {
        fn label(&self) -> String {
            "failed-open".into()
        }
        fn capabilities(&self) -> RealtimeCapabilities {
            RealtimeCapabilities { audio_output: true, ..Default::default() }
        }
        fn session(
            &self,
            _: bool,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Arc<dyn RealtimeSession>, ProviderError>> {
            Box::pin(async move {
                assert_eq!(
                    self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                    0,
                    "the same failover must not reopen a failed child"
                );
                Err(lk_agents_abi::errors::ApiError::connection(Some("open failed".into())).into())
            })
        }
    }
    let primary = scripted(vec![fatal()]);
    let backup = Arc::new(FailsToOpen(Default::default()));
    let model = RealtimeFallbackModel::new(
        vec![
            ChainModel::Native(Arc::new(Model(primary))),
            ChainModel::Native(backup.clone()),
        ],
        0.0,
        false,
        Arc::new(crate::fallback::IgnoreEvents),
        Arc::new(lk_agents_abi::RealClock::default()),
    )
    .unwrap();
    let session = model.session(false).await.unwrap();
    assert!(matches!(session.next_event().await, Some(Event::Error(e)) if e.recoverable));
    assert!(matches!(session.next_event().await, Some(Event::Error(e)) if !e.recoverable));
    assert_eq!(backup.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    session.aclose().await;
}

#[tokio::test]
async fn realtime_fallback_backlog_refuses_excess_requests_before_forwarding() {
    let primary = scripted(vec![fatal()]);
    let backup = scripted(vec![]);
    let model = RealtimeFallbackModel::new(
        vec![
            ChainModel::Native(Arc::new(Model(primary))),
            ChainModel::Native(Arc::new(Model(backup.clone()))),
        ],
        10.0,
        false,
        Arc::new(crate::fallback::IgnoreEvents),
        Arc::new(lk_agents_abi::RealClock::default()),
    )
    .unwrap();
    let session = model.session(false).await.unwrap();
    assert!(matches!(session.next_event().await, Some(Event::Error(_))));
    assert!(matches!(session.next_event().await, Some(Event::SessionSwapped { .. })));
    let (polling, polled) = tokio::sync::oneshot::channel();
    let rejected_reader = tokio::spawn({
        let session = session.clone();
        async move {
            let _ = polling.send(());
            session.next_event().await
        }
    });
    polled.await.unwrap();
    for index in 0..33 {
        session.send(RealtimeCommand::GenerateReply(lk_agents_abi::GenerateReply {
            req_id: format!("queued_{index}"),
            instructions: None,
            tool_choice: None,
        }));
    }
    session.send(RealtimeCommand::CancelReply(lk_agents_abi::CancelReply {
        req_id: "queued_0".into(),
    }));
    session.send(RealtimeCommand::GenerateReply(lk_agents_abi::GenerateReply {
        req_id: "oversized".into(),
        instructions: Some("x".repeat(1024 * 1024)),
        tool_choice: None,
    }));
    session.send(RealtimeCommand::GenerateReply(lk_agents_abi::GenerateReply {
        req_id: "replacement".into(),
        instructions: None,
        tool_choice: None,
    }));
    session.send(RealtimeCommand::UpdateTools(lk_agents_abi::UpdateTools { tools: vec![] }));
    assert!(
        !backup.sent.lock().unwrap().iter().any(|command| command == "generate_reply:queued_32"),
        "a request beyond the bounded replay backlog must never reach the replacement model"
    );
    let rejected = tokio::time::timeout(std::time::Duration::from_secs(1), rejected_reader)
        .await
        .expect("a held-backlog rejection must wake an already parked reader")
        .unwrap();
    assert!(matches!(rejected, Some(Event::GenerateReplyFailed(failed))
        if failed.req_id == "queued_32" && failed.error.code.as_deref() == Some("realtime_fallback_backlog_exceeded")));
    assert!(
        backup.sent.lock().unwrap().iter().any(|command| command == "generate_reply:replacement"),
        "cancellation releases admission"
    );
    assert!(
        !backup.sent.lock().unwrap().iter().any(|command| command == "generate_reply:oversized")
    );
    let rejected = tokio::time::timeout(std::time::Duration::from_secs(1), session.next_event())
        .await
        .unwrap();
    assert!(
        matches!(rejected, Some(Event::GenerateReplyFailed(failed)) if failed.req_id == "oversized" && failed.error.code.as_deref() == Some("realtime_fallback_backlog_exceeded"))
    );
    session.aclose().await;
}
