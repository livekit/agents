//! The console boundary: typed messages and audio legs, with CLI playout on virtual time.

use lk_agents_abi::errors::HostError;
use lk_agents_abi::voice::console::{ConsolePort, ConsoleSink, SessionTransport};
use lk_agents_abi::voice::io::{
    AudioInput, AudioOutput, AudioOutputCapabilities, IoError, PlaybackFinished, PlaybackTracker,
};
use lk_agents_abi::{AudioFrame, BoxFuture};
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
    segment: Mutex<(Option<f64>, Duration)>,
    spawner: Arc<dyn lk_agents_abi::Spawner>,
    writer: Mutex<
        Option<(
            tokio::sync::mpsc::UnboundedReceiver<proto::AgentSessionMessage>,
            tokio::sync::mpsc::UnboundedSender<proto::AgentSessionMessage>,
        )>,
    >,
}

impl FakeConsole {
    pub fn as_audio_input(self: &Arc<Self>) -> Arc<dyn AudioInput> {
        self.clone()
    }
    pub fn as_audio_output(self: &Arc<Self>) -> Arc<dyn AudioOutput> {
        self.clone()
    }
    fn send(&self, message: Message) {
        if let Some(outgoing) = self.outgoing.lock().unwrap().as_ref() {
            let _ = outgoing.send(proto::AgentSessionMessage { message: Some(message) });
        }
    }
    fn finish(&self, interrupted: bool) {
        let (started, pushed) = std::mem::take(&mut *self.segment.lock().unwrap());
        if let Some(started) = started {
            let played = Duration::try_from_secs_f64(lk_agents_abi::unix_now() - started)
                .unwrap_or_default();
            self.tracker.playback_finished(PlaybackFinished {
                playback_position: if interrupted { played.min(pushed) } else { pushed },
                interrupted,
                synchronized_transcript: None,
            });
        }
    }
}

impl ConsolePort for FakeConsole {
    fn as_transport(self: Arc<Self>) -> Arc<dyn SessionTransport> {
        self
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

impl SessionTransport for FakeConsole {
    fn start(&self) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            let Some((mut messages, to_cli)) = self.writer.lock().unwrap().take() else {
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
        if let Some(outgoing) = self.outgoing.lock().unwrap().as_ref() {
            let _ = outgoing.send(message.clone());
        }
    }
    fn recv(&self) -> BoxFuture<'_, Result<Option<proto::AgentSessionMessage>, HostError>> {
        Box::pin(async move {
            let message = self.incoming.lock().await.recv().await;
            if message.is_none() {
                self.microphone.lock().unwrap().take();
            }
            Ok(message)
        })
    }
    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.outgoing.lock().unwrap().take();
            self.microphone.lock().unwrap().take();
        })
    }
}

impl ConsoleSink for FakeConsole {
    fn push_input_frame(&self, frame: &console_io::AudioFrame) {
        if let Some(mic) = self.microphone.lock().unwrap().as_ref() {
            let _ = mic.send(AudioFrame::new(
                frame.data.clone(),
                frame.sample_rate,
                frame.num_channels as u16,
                frame.samples_per_channel,
            ));
        }
    }
    fn playout_finished(&self) {
        self.finish(false);
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
        Some(48_000)
    }
    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> BoxFuture<'a, Result<(), IoError>> {
        Box::pin(async move {
            let now = lk_agents_abi::unix_now();
            let first = {
                let mut segment = self.segment.lock().unwrap();
                segment.1 += Duration::from_secs_f64(frame.duration());
                let first = segment.0.is_none();
                segment.0.get_or_insert(now);
                first
            };
            if first {
                self.tracker.report_started(now);
            }
            self.send(Message::AudioOutput(console_io::AudioFrame {
                data: frame.data.to_vec(),
                sample_rate: frame.sample_rate,
                num_channels: u32::from(frame.num_channels),
                samples_per_channel: frame.samples_per_channel,
            }));
            Ok(())
        })
    }
    fn flush_segment(&self) {
        self.send(Message::AudioPlaybackFlush(console_io::AudioPlaybackFlush {}));
    }
    fn clear_buffer(&self) {
        self.send(Message::AudioPlaybackClear(console_io::AudioPlaybackClear {}));
        self.finish(true);
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
    pub fn connect(
        spawner: Arc<dyn lk_agents_abi::Spawner>,
        world: Arc<dyn lk_agents_abi::Spawner>,
    ) -> (Arc<FakeConsole>, FakeCli) {
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
            spawner,
            writer: Mutex::new(Some((messages, to_cli))),
            incoming: tokio::sync::Mutex::new(incoming),
            outgoing: Mutex::new(Some(outgoing)),
            mic,
            microphone: Mutex::new(Some(microphone)),
            tracker: Arc::new(PlaybackTracker::new()),
            segment: Mutex::default(),
        });
        let cli = FakeCli { to_agent, playing_until: Arc::default() };
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
        until.map_or(Duration::ZERO, |at| at.saturating_duration_since(tokio::time::Instant::now()))
    }

    /// One message to the agent; `false` once the CLI's socket is gone.
    pub fn send(&self, message: Message) -> bool {
        let message = proto::AgentSessionMessage { message: Some(message) };
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
        Ok(("agent".into(), Some("You are a test agent.".into()), Vec::new(), Vec::new()))
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
        Ok(("listening".into(), "listening".into(), "agent".into(), options, Some(0.0)))
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
