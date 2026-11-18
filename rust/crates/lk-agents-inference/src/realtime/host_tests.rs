use super::*;
use lk_agents_abi::llm::realtime::{GenerateReply, RealtimeError};
use lk_agents_abi::{BoxFuture, ChatItem, HostRef, RefRegistry};

struct Registry;

impl RefRegistry for Registry {
    fn release(&self, _id: u64) {}
}

fn host_ref(id: u64) -> HostRef {
    let registry: Arc<dyn RefRegistry> = Arc::new(Registry);
    HostRef::new(id, &registry)
}

/// A plugin session whose generation's streams raise as it is drained; it sends no event of its
/// own.
struct RaisingDrain {
    fail_drain: bool,
    events: tokio::sync::mpsc::UnboundedSender<Option<Event>>,
    receiver: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<Option<Event>>>,
}

impl RaisingDrain {
    fn new(fail_drain: bool) -> Self {
        let (events, receiver) = tokio::sync::mpsc::unbounded_channel();
        Self { fail_drain, events, receiver: tokio::sync::Mutex::new(receiver) }
    }
}

impl HostRealtimeSession for RaisingDrain {
    fn send(&self, _: RealtimeCommand) {}
    fn update_instructions(
        &self,
        _: &telemetry::Cx,
        instructions: String,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            if instructions == "reject" { Err(HostError::new("options rejected")) } else { Ok(()) }
        })
    }
    fn update_chat_ctx(
        &self,
        _: &telemetry::Cx,
        _: Vec<ChatItem>,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn push_item(&self, _: &telemetry::Cx, _: ChatItem) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn update_tools(
        &self,
        _: &telemetry::Cx,
        _: Vec<String>,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn request(
        &self,
        _: &telemetry::Cx,
        _: RealtimeCommand,
    ) -> BoxFuture<'_, Result<HostRef, RealtimeError>> {
        Box::pin(async { Ok(host_ref(1)) })
    }
    fn generation(
        &self,
        _: &telemetry::Cx,
        _: HostRef,
    ) -> BoxFuture<'_, Result<HostRef, RealtimeError>> {
        Box::pin(async { Ok(host_ref(2)) })
    }
    fn cancel(&self, _: &telemetry::Cx, _: HostRef) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn drain(
        &self,
        _: &telemetry::Cx,
        _: HostRef,
        req_id: Option<String>,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            if self.fail_drain {
                return Err(HostError::new("message_stream raised"));
            }
            self.events
                .send(Some(Event::GenerationCreated(
                    lk_agents_abi::llm::realtime::GenerationCreated {
                        response_id: "healthy-response".into(),
                        req_id,
                    },
                )))
                .unwrap();
            self.events
                .send(Some(Event::GenerationDone(lk_agents_abi::llm::realtime::GenerationDone {
                    response_id: "healthy-response".into(),
                    status: lk_agents_abi::llm::realtime::GenerationStatus::Completed,
                    usage: Default::default(),
                })))
                .unwrap();
            Ok(())
        })
    }
    fn next_event(&self, _: &telemetry::Cx) -> BoxFuture<'_, Result<Option<Event>, HostError>> {
        Box::pin(async { Ok(self.receiver.lock().await.recv().await.flatten()) })
    }
    fn aclose(&self, _: &telemetry::Cx) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async {
            self.events.send(None).unwrap();
            Err(HostError::new("aclose raised"))
        })
    }
}

// Catches: a drain whose streams raised swallowed in the binding, so the core never learned the
// reply failed; and a raising close hanging or panicking the leg.
#[tokio::test]
async fn a_drain_that_raises_reaches_the_core_as_an_error_for_its_request() {
    let spawner: Arc<dyn lk_agents_abi::Spawner> =
        Arc::new(lk_agents_net::task::TokioSpawner::current());
    let clock = lk_agents_abi::VirtualClock::new();
    let leg = HostRealtimeLeg::open(
        Arc::new(RaisingDrain::new(true)),
        telemetry::Cx::NONE,
        RealtimeCapabilities::default(),
        WallAnchor::of(&clock),
        &spawner,
    );
    let reply = GenerateReply { req_id: "r1".into(), instructions: None, tool_choice: None };
    leg.send(RealtimeCommand::GenerateReply(reply));
    let event = leg.next_event().await.expect("the failure is an event");
    let Event::Error(error) = event else { panic!("an error event: {event:?}") };
    assert_eq!((error.req_id.as_deref(), error.recoverable), (Some("r1"), true));
    assert!(error.message.contains("message_stream raised"), "{}", error.message);
    leg.aclose().await;
}

#[tokio::test]
async fn a_rejected_host_update_is_reported_without_poisoning_the_next_request() {
    let spawner: Arc<dyn lk_agents_abi::Spawner> =
        Arc::new(lk_agents_net::task::TokioSpawner::current());
    let clock = lk_agents_abi::VirtualClock::new();
    let leg = HostRealtimeLeg::open(
        Arc::new(RaisingDrain::new(false)),
        telemetry::Cx::NONE,
        RealtimeCapabilities::default(),
        WallAnchor::of(&clock),
        &spawner,
    );
    for instructions in ["reject", "accepted"] {
        leg.send(RealtimeCommand::UpdateInstructions(
            lk_agents_abi::llm::realtime::UpdateInstructions { instructions: instructions.into() },
        ));
    }
    leg.send(RealtimeCommand::GenerateReply(GenerateReply {
        req_id: "next-request".into(),
        instructions: None,
        tool_choice: None,
    }));
    let first = leg.next_event().await.unwrap();
    assert!(
        matches!(first, Event::Error(ref e) if e.recoverable && e.req_id.is_none()),
        "{first:?}"
    );
    assert!(
        matches!(leg.next_event().await.unwrap(), Event::GenerationCreated(c) if c.req_id.as_deref() == Some("next-request"))
    );
    assert!(
        matches!(leg.next_event().await.unwrap(), Event::GenerationDone(d) if d.status == lk_agents_abi::llm::realtime::GenerationStatus::Completed)
    );
    leg.aclose().await;
}
