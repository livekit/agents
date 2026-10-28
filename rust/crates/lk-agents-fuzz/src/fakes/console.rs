//! The console boundary: typed messages and audio legs, with CLI playout on virtual time. What the
//! legs account for is the session's own [`ConsolePlayback`], on the simulated clock.

use lk_agents_abi::errors::HostError;
use lk_agents_abi::voice::console::{ConsolePort, ConsoleSink, SessionTransport};
use lk_agents_abi::voice::io::{
    AudioInput, AudioOutput, AudioOutputCapabilities, IoError, PlaybackTracker,
};
use lk_agents_abi::{AudioFrame, BoxFuture, Clock, Spawner};
use lk_agents_voice::remote_session::console::{
    CONSOLE_SAMPLE_RATE, ConsolePlayback, microphone_frame,
};
use lk_agents_voice::remote_session::{self, proto};
use proto::agent_session_message::{Message, console_io};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The session's end of the simulated console; its stream has no TCP execution.
pub struct FakeConsole {
    incoming: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<proto::AgentSessionMessage>>,
    outgoing: Mutex<Option<tokio::sync::mpsc::UnboundedSender<proto::AgentSessionMessage>>>,
    mic: Arc<super::output::FakeAudioInput>,
    microphone: Mutex<Option<tokio::sync::mpsc::UnboundedSender<AudioFrame>>>,
    tracker: Arc<PlaybackTracker>,
    playback: Mutex<ConsolePlayback>,
    clock: Arc<dyn Clock>,
    writer: Mutex<
        Option<(
            tokio::sync::mpsc::UnboundedReceiver<proto::AgentSessionMessage>,
            tokio::sync::mpsc::UnboundedSender<proto::AgentSessionMessage>,
        )>,
    >,
}

/// The console as the session's transport, writing on the session's spawner.
struct FakeTransport {
    console: Arc<FakeConsole>,
    spawner: Arc<dyn Spawner>,
}

impl FakeConsole {
    /// The simulated wall clock, as the session's remote host reads it.
    fn now(&self) -> f64 {
        lk_agents_abi::unix_secs(self.clock.wall())
    }
    fn send(&self, message: Message) {
        if let Some(outgoing) = self.outgoing.lock().unwrap().as_ref() {
            let _ = outgoing.send(proto::AgentSessionMessage {
                message: Some(message),
            });
        }
    }
    fn finish(&self) {
        if let Some((segment, finished)) = self.playback.lock().unwrap().clear() {
            self.tracker.playback_finished(segment, finished);
        }
    }
}

impl ConsolePort for FakeConsole {
    fn as_transport(self: Arc<Self>, spawner: Arc<dyn Spawner>) -> Arc<dyn SessionTransport> {
        Arc::new(FakeTransport {
            console: self,
            spawner,
        })
    }
    fn as_console(self: Arc<Self>) -> Arc<dyn ConsoleSink> {
        self
    }
    fn as_audio_input(self: Arc<Self>) -> Arc<dyn AudioInput> {
        self
    }
    fn as_audio_output(self: Arc<Self>) -> Arc<dyn AudioOutput> {
        self
    }
}

impl SessionTransport for FakeTransport {
    fn start(&self) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            let Some((mut messages, to_cli)) = self.console.writer.lock().unwrap().take() else {
                return Ok(());
            };
            self.spawner.spawn(Box::pin(async move {
                while let Some(message) = messages.recv().await {
                    if to_cli.send(message).is_err() {
                        return;
                    }
                }
            }));
            Ok(())
        })
    }
    fn send_message<'a>(
        &'a self,
        message: &'a proto::AgentSessionMessage,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        self.send_message_nowait(message);
        Box::pin(async { Ok(()) })
    }
    fn send_message_nowait(&self, message: &proto::AgentSessionMessage) {
        if let Some(outgoing) = self.console.outgoing.lock().unwrap().as_ref() {
            let _ = outgoing.send(message.clone());
        }
    }
    fn recv(&self) -> BoxFuture<'_, Result<Option<proto::AgentSessionMessage>, HostError>> {
        Box::pin(async move {
            let message = self.console.incoming.lock().await.recv().await;
            if message.is_none() {
                self.console.microphone.lock().unwrap().take();
            }
            Ok(message)
        })
    }
    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.console.outgoing.lock().unwrap().take();
            self.console.microphone.lock().unwrap().take();
        })
    }
}

impl ConsoleSink for FakeConsole {
    fn push_input_frame(&self, frame: &console_io::AudioFrame) {
        if let Some(mic) = self.microphone.lock().unwrap().as_ref() {
            let _ = mic.send(microphone_frame(frame));
        }
    }
    fn playout_finished(&self) {
        // This raw CLI acknowledgement has no operation key and cannot settle a local buffer.
    }
}

impl AudioInput for FakeConsole {
    fn label(&self) -> &str {
        "Console"
    }
    fn next_frame(&self) -> BoxFuture<'_, Result<Option<AudioFrame>, IoError>> {
        self.mic.next_frame()
    }
}

impl AudioOutput for FakeConsole {
    fn label(&self) -> &str {
        "Console"
    }
    fn tracker(&self) -> &Arc<PlaybackTracker> {
        &self.tracker
    }
    fn capabilities(&self) -> AudioOutputCapabilities {
        AudioOutputCapabilities { pause: false }
    }
    fn sample_rate(&self) -> Option<u32> {
        Some(CONSOLE_SAMPLE_RATE)
    }
    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> BoxFuture<'a, Result<(), IoError>> {
        Box::pin(async move {
            let now = self.now();
            let segment = self
                .tracker
                .current_segment()
                .ok_or(IoError::PlaybackSegmentUnavailable)?;
            let (message, opened) = self.playback.lock().unwrap().capture(frame, segment)?;
            if opened {
                self.tracker.report_started(segment, now);
            }
            self.send(message);
            Ok(())
        })
    }
    fn flush_segment(&self) {
        self.send(Message::AudioPlaybackFlush(
            console_io::AudioPlaybackFlush {},
        ));
    }
    fn clear_buffer(&self) {
        self.send(Message::AudioPlaybackClear(
            console_io::AudioPlaybackClear {},
        ));
        self.finish();
    }
}

/// `lk agent console`'s simulated peer: what it sends the agent (the microphone,
/// requests, playout acks) goes out in order, and every segment the agent flushes is acked once
/// its audio has played on the clock, unless a clear cut it first. Its tasks are the CLI's, not
/// the session's, so nothing judges them.
#[derive(Clone)]
pub struct FakeCli {
    to_agent: tokio::sync::mpsc::UnboundedSender<proto::AgentSessionMessage>,
    /// When the last flushed segment finishes playing, on the clock: its ack is owed until then.
    playing_until: Arc<std::sync::Mutex<Option<tokio::time::Instant>>>,
}

impl FakeCli {
    /// The typed console boundary and the CLI's scheduled playout acknowledgements.
    /// Each direction keeps its writer task: a send is queued before its peer receives it.
    pub fn connect(world: Arc<dyn Spawner>, clock: Arc<dyn Clock>) -> (Arc<FakeConsole>, FakeCli) {
        let (to_agent, mut sent) = tokio::sync::mpsc::unbounded_channel();
        let (to_console, incoming) = tokio::sync::mpsc::unbounded_channel();
        world.spawn(Box::pin(async move {
            while let Some(message) = sent.recv().await {
                if to_console.send(message).is_err() {
                    return;
                }
            }
        }));
        let (outgoing, messages) = tokio::sync::mpsc::unbounded_channel();
        let (to_cli, mut received) = tokio::sync::mpsc::unbounded_channel();
        let (mic, microphone) = super::output::FakeAudioInput::new();
        let io = Arc::new(FakeConsole {
            writer: Mutex::new(Some((messages, to_cli))),
            incoming: tokio::sync::Mutex::new(incoming),
            outgoing: Mutex::new(Some(outgoing)),
            mic,
            microphone: Mutex::new(Some(microphone)),
            tracker: Arc::new(PlaybackTracker::new()),
            playback: Mutex::default(),
            clock,
        });
        let cli = FakeCli {
            to_agent,
            playing_until: Arc::default(),
        };
        let speaker = cli.clone();
        let world_inner = world.clone();
        world.spawn(Box::pin(async move {
            // The audio of the segment being received, and which segment it is: a clear moves
            // on, so an ack still waiting for a cut segment's audio is dropped.
            let (mut pending, mut segment) = (Duration::ZERO, 0u64);
            let played = Arc::new(std::sync::atomic::AtomicU64::new(0));
            while let Some(message) = received.recv().await {
                match message.message {
                    Some(Message::AudioOutput(f)) => {
                        pending += Duration::from_secs_f64(
                            f64::from(f.samples_per_channel) / f64::from(f.sample_rate.max(1)),
                        );
                    }
                    Some(Message::AudioPlaybackFlush(_)) => {
                        let (speaker, played, this) = (speaker.clone(), played.clone(), segment);
                        let audio = std::mem::take(&mut pending);
                        let until = tokio::time::Instant::now() + audio;
                        let mut playing = speaker.playing_until.lock().unwrap();
                        *playing = Some(playing.map_or(until, |at| at.max(until)));
                        drop(playing);
                        world_inner.spawn(Box::pin(async move {
                            tokio::time::sleep(audio).await;
                            if played.load(std::sync::atomic::Ordering::Relaxed) == this {
                                speaker.send(Message::AudioPlaybackFinished(
                                    console_io::AudioPlaybackFinished {},
                                ));
                            }
                        }));
                    }
                    Some(Message::AudioPlaybackClear(_)) => {
                        pending = Duration::ZERO;
                        segment += 1;
                        played.store(segment, std::sync::atomic::Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
        }));
        (io, cli)
    }

    /// The agent's voice the speaker has yet to play: an ack still owed on the clock.
    pub fn queued(&self) -> Duration {
        let until = *self.playing_until.lock().unwrap();
        until.map_or(Duration::ZERO, |at| {
            at.saturating_duration_since(tokio::time::Instant::now())
        })
    }

    /// One message to the agent; `false` once the CLI's socket is gone.
    pub fn send(&self, message: Message) -> bool {
        let message = proto::AgentSessionMessage {
            message: Some(message),
        };
        self.to_agent.send(message).is_ok()
    }
}

/// What the CLI asks of the session (`ConsoleCall`'s `_core_*` port), answered as the host would
/// from its own state. Toggling the legs stays with the case's own inputs, so `update_io` is
/// acknowledged and changes nothing.
pub struct FakeConsolePort;

#[async_trait::async_trait]
impl remote_session::SessionPort for FakeConsolePort {
    async fn chat_history(&self) -> Result<Vec<lk_agents_abi::ChatItem>, HostError> {
        Ok(Vec::new())
    }
    async fn agent_info(&self) -> Result<remote_session::AgentInfo, HostError> {
        Ok((
            "agent".into(),
            Some("You are a test agent.".into()),
            Vec::new(),
            Vec::new(),
        ))
    }
    async fn run_input(&self, text: &str) -> Result<remote_session::RunOutcome, HostError> {
        Ok(if text.is_empty() {
            (None, Some("empty run_input text".into()))
        } else {
            (Some(Vec::new()), None)
        })
    }
    async fn session_state(&self) -> Result<remote_session::SessionState, HostError> {
        let options = remote_session::STATE_OPTION_FIELDS
            .iter()
            .map(|(key, _)| ((*key).to_owned(), "None".to_owned()))
            .collect();
        Ok((
            "listening".into(),
            "listening".into(),
            "agent".into(),
            options,
            Some(0.0),
        ))
    }
    async fn rtc_stats(&self) -> Result<Option<remote_session::RtcStats>, HostError> {
        Ok(None)
    }
    async fn session_usage(
        &self,
    ) -> Result<Vec<lk_agents_abi::metrics::usage::ModelUsage>, HostError> {
        Ok(Vec::new())
    }
    fn framework_info(&self) -> (String, String) {
        ("rust".into(), "fuzz".into())
    }
    async fn update_io(&self, _update: remote_session::IoUpdate) -> Result<(), HostError> {
        Ok(())
    }
    async fn finalize_simulation(
        &self,
        _provisional: remote_session::SimulationVerdict,
    ) -> Result<remote_session::VerdictOutcome, HostError> {
        Ok((None, None))
    }
}
