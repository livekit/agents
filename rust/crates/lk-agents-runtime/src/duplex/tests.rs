//! Duplex control outcomes must wake a parked reader and remain bounded without host output.

use super::*;
use lk_agents_abi::VirtualClock;
use lk_agents_abi::llm::realtime::GenerateReply;

struct SilentHost;

#[async_trait::async_trait]
impl DuplexHost for SilentHost {
    fn label(&self) -> String {
        "silent".into()
    }
    fn capabilities(&self) -> DuplexCapabilities {
        DuplexCapabilities::default()
    }
    async fn open(
        &self,
    ) -> Result<(Arc<dyn DuplexHostSession>, Option<Box<dyn AudioGate>>), ProviderError> {
        Ok((Arc::new(SilentHost), None))
    }
}

#[async_trait::async_trait]
impl DuplexHostSession for SilentHost {
    fn send(&self, _: DuplexCommand) {}
    async fn next_event(&self) -> Option<DuplexEvent> {
        std::future::pending().await
    }
    async fn aclose(&self) {}
}

async fn session(clock: Arc<VirtualClock>) -> Arc<dyn RealtimeSession> {
    DuplexModel::new(Arc::new(SilentHost), lk_agents_inference::duplex::AUDIO_TIMEOUT, clock)
        .session(false)
        .await
        .unwrap()
}

fn reply(req_id: &str) -> RealtimeCommand {
    RealtimeCommand::GenerateReply(GenerateReply {
        req_id: req_id.into(),
        instructions: None,
        tool_choice: None,
    })
}

#[tokio::test]
async fn duplex_command_outcome_wakes_a_reader_parked_without_host_output() {
    for supersede in [true, false] {
        let clock = Arc::new(VirtualClock::new());
        let session = session(clock.clone()).await;
        let next = session.next_event();
        tokio::pin!(next);
        tokio::select! {
            biased;
            _ = &mut next => panic!("silent host unexpectedly produced output"),
            () = tokio::task::yield_now() => {}
        }
        session.send(reply("first"));
        if supersede {
            session.send(reply("second"));
        } else {
            clock.advance(lk_agents_inference::duplex::segmenter::REPLY_TIMEOUT);
        }
        let event = tokio::time::timeout(Duration::from_secs(1), &mut next)
            .await
            .expect("command outcome did not wake the parked reader");
        assert!(
            matches!(event, Some(Event::GenerateReplyFailed(failed)) if failed.req_id == "first")
        );
        session.aclose().await;
    }
}

#[tokio::test]
async fn duplex_command_backlog_preserves_a_terminal_overload_outcome() {
    let session = session(Arc::new(VirtualClock::new())).await;
    let capacity = crate::realtime::queue::EVENT_BACKLOG;
    for index in 0..capacity + 2 {
        session.send(reply(&format!("request_{index}")));
    }
    let mut terminal = None;
    for _ in 0..=capacity {
        let event = tokio::time::timeout(Duration::from_secs(1), session.next_event())
            .await
            .expect("admitted outcomes must end or expose the reserved overload failure");
        if let Some(Event::Error(error)) = event {
            terminal = Some(error);
            break;
        }
    }
    assert!(
        terminal.is_some_and(|error| !error.recoverable && error.provider_error.is_some()),
        "event saturation silently retained an unbounded backlog"
    );
    assert!(session.next_event().await.is_none());
    session.aclose().await;
}

struct EndingHost;
struct EndingSession(Mutex<Option<Fragment>>);

#[async_trait::async_trait]
impl DuplexHost for EndingHost {
    fn label(&self) -> String {
        "ending".into()
    }
    fn capabilities(&self) -> DuplexCapabilities {
        DuplexCapabilities::default()
    }
    async fn open(
        &self,
    ) -> Result<(Arc<dyn DuplexHostSession>, Option<Box<dyn AudioGate>>), ProviderError> {
        Ok((
            Arc::new(EndingSession(Mutex::new(Some(Fragment {
                text: "Final text".into(),
                start_ms: Some(10_000),
                end_ms: Some(10_200),
            })))),
            None,
        ))
    }
}

#[async_trait::async_trait]
impl DuplexHostSession for EndingSession {
    fn send(&self, _: DuplexCommand) {}
    async fn next_event(&self) -> Option<DuplexEvent> {
        self.0.lock().unwrap().take().map(DuplexEvent::Transcript)
    }
    async fn aclose(&self) {}
}

#[tokio::test]
async fn duplex_host_eof_delivers_unclaimed_final_text_before_stream_end() {
    let session = DuplexModel::new(
        Arc::new(EndingHost),
        lk_agents_inference::duplex::AUDIO_TIMEOUT,
        Arc::new(VirtualClock::new()),
    )
    .session(false)
    .await
    .unwrap();
    let mut text = String::new();
    let mut completed = false;
    while let Some(event) = session.next_event().await {
        match event {
            Event::TextDelta(delta) => text.push_str(&delta.delta),
            Event::GenerationDone(_) => completed = true,
            _ => {}
        }
    }
    assert_eq!(text, "Final text");
    assert!(completed);
    session.aclose().await;
}
