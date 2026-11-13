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
struct RaisingDrain;

impl HostRealtimeSession for RaisingDrain {
    fn send(&self, _: RealtimeCommand) {}
    fn update_instructions(
        &self,
        _: &telemetry::Cx,
        _: String,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
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
        _: Option<String>,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Err(HostError::new("message_stream raised")) })
    }
    fn next_event(&self, _: &telemetry::Cx) -> BoxFuture<'_, Result<Option<Event>, HostError>> {
        Box::pin(std::future::pending())
    }
    fn aclose(&self, _: &telemetry::Cx) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Err(HostError::new("aclose raised")) })
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
        Arc::new(RaisingDrain),
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
