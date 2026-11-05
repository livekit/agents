//! Duplex control outcomes must wake a parked reader and remain bounded without host output.

use super::*;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::llm::duplex::{self as abi, DuplexAudio, DuplexTranscript};
use lk_agents_abi::llm::realtime::GenerateReply;
use lk_agents_abi::{AudioFrame, BoxFuture, ToolChoice, VirtualClock};

/// A plugin's model and session over the declared views: `None` stays silent and open; a script
/// is its transcript, then the end of the session.
struct FakeHost(Arc<FakeSession>);
struct FakeSession(Mutex<Option<Vec<DuplexTranscript>>>, Arc<dyn abi::DuplexInputQueue>);

impl DuplexHost for FakeHost {
    fn label(&self) -> Result<String, HostError> {
        Ok("fake".into())
    }
    fn capabilities(&self) -> Result<DuplexCapabilities, HostError> {
        Ok(DuplexCapabilities::default())
    }
    fn gate(&self) -> Result<Option<Arc<dyn abi::AudioGate>>, HostError> {
        Ok(None)
    }
    fn session(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> BoxFuture<'_, Result<Arc<dyn abi::DuplexSession>, HostError>> {
        Box::pin(async { Ok(self.0.clone() as Arc<dyn abi::DuplexSession>) })
    }
}

fn applied<'a>() -> BoxFuture<'a, Result<(), HostError>> {
    Box::pin(async { Ok(()) })
}

impl abi::DuplexSession for FakeSession {
    fn input(&self) -> &Arc<dyn abi::DuplexInputQueue> {
        &self.1
    }
    fn configure(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Option<String>,
        _: Option<Vec<ChatItem>>,
        _: Vec<String>,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        applied()
    }
    fn update_instructions(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: String,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        applied()
    }
    fn append_items(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Vec<ChatItem>,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        applied()
    }
    fn update_tools(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Vec<String>,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        applied()
    }
    fn generate(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Option<String>,
        _: Option<ToolChoice>,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        applied()
    }
    fn next_audio(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> BoxFuture<'_, Result<Option<DuplexAudio>, HostError>> {
        Box::pin(std::future::pending())
    }
    fn next_event(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> BoxFuture<'_, Result<Option<abi::DuplexEvent>, HostError>> {
        Box::pin(async move {
            let next = self.0.lock().as_mut().map(Vec::pop);
            match next {
                Some(transcript) => Ok(transcript.map(abi::DuplexEvent::Transcript)),
                None => std::future::pending().await,
            }
        })
    }
    fn close_plugin(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
    fn aclose(&self, _cx: &lk_agents_abi::telemetry::Cx) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

async fn session(
    script: Option<Vec<DuplexTranscript>>,
    clock: Arc<VirtualClock>,
) -> Arc<dyn RealtimeSession> {
    let input = Arc::new(crate::duplex::DuplexInput::new(clock.clone()));
    session_on(script, clock, input).await
}

async fn session_on(
    script: Option<Vec<DuplexTranscript>>,
    clock: Arc<VirtualClock>,
    input: Arc<crate::duplex::DuplexInput>,
) -> Arc<dyn RealtimeSession> {
    let model = DuplexAdapterModel {
        host: Arc::new(FakeHost(Arc::new(FakeSession(Mutex::new(script), input)))),
        label: "fake".into(),
        caps: crate::duplex::realtime_capabilities(DuplexCapabilities::default()),
        audio_timeout: crate::duplex::AUDIO_TIMEOUT,
        anchor: WallAnchor::of(&*clock),
        clock,
        spawner: Arc::new(lk_agents_net::task::TokioSpawner::current()),
    };
    model.session(lk_agents_abi::telemetry::Cx::NONE, false).await.unwrap()
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
        let session = session(None, clock.clone()).await;
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
            clock.advance(crate::duplex::segmenter::REPLY_TIMEOUT);
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
    let session = session(None, Arc::new(VirtualClock::new())).await;
    let capacity = crate::realtime::openai::queue::EVENT_BACKLOG;
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

#[tokio::test]
async fn duplex_host_eof_delivers_unclaimed_final_text_before_stream_end() {
    let fragment = DuplexTranscript {
        text: "Final text".into(),
        start_ms: Some(10_000),
        end_ms: Some(10_200),
    };
    let session = session(Some(vec![fragment]), Arc::new(VirtualClock::new())).await;
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

// Catches: the caller's audio going anywhere but the plugin's input, or the pre-connect head
// held to the live audio's bound there.
#[tokio::test]
async fn the_callers_audio_reaches_the_plugins_input_head_first_and_in_full() {
    let clock = Arc::new(VirtualClock::new());
    let input = Arc::new(crate::duplex::DuplexInput::new(clock.clone()));
    let session = session_on(None, clock, input.clone()).await;
    let frame = |tag: i16| AudioFrame::new(tag.to_le_bytes().repeat(160), 16_000, 1, 160);
    for _ in 0..100 {
        session.head(frame(1));
    }
    session.send(RealtimeCommand::PushAudio(lk_agents_abi::PushAudio { frame: frame(2) }));
    let (mut head, mut live) = (0.0, 0.0);
    while let Ok(Some(item)) = tokio::time::timeout(Duration::from_millis(10), input.recv()).await {
        let abi::DuplexInputItem::Audio(f) = item else { panic!("unexpected {item:?}") };
        *if f.data[0] == 1 { &mut head } else { &mut live } += f.duration();
    }
    assert!((head - 1.0f64).abs() < 1e-9 && (live - 0.01f64).abs() < 1e-9, "{head} {live}");
    session.aclose().await;
}
