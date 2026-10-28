//! `lk agent console`'s local call (`cli/tcp_console.py`): the CLI owns the microphone, the speakers
//! and the TUI, and reaches the agent over one TCP socket carrying length-prefixed
//! `AgentSessionMessage`s. [`ConsoleIo`] is that socket as the session sees it: the
//! [`SessionTransport`] and [`ConsoleSink`] the session's remote session host
//! ([`HostRuntime`](super::driver::HostRuntime)) runs on, and the session's audio input and output legs.

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::voice::io::{
    AudioInput, AudioOutput, AudioOutputCapabilities, IoError, PlaybackFinished, PlaybackTracker,
};
use lk_agents_abi::{AudioFrame, Spawner, SpawnerExt};
use prost::Message as _;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;
#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
use tokio::sync::{Mutex as AsyncMutex, mpsc};

use super::{ConsoleSink, FrameDecoder, SessionTransport, frame, proto};
use proto::agent_session_message::{Message, console_io};

/// The rate the CLI captures and plays at.
const WIRE_SAMPLE_RATE: u32 = 48_000;

/// The CLI's socket: transport, console sink and both audio legs. Connected by `start()`; what is
/// sent before then is queued.
pub struct ConsoleIo {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    socket: Mutex<Option<Socket>>,
    spawner: Arc<dyn Spawner>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    outgoing: Mutex<Option<mpsc::UnboundedSender<Vec<u8>>>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    outgoing_rx: Mutex<Option<mpsc::UnboundedReceiver<Vec<u8>>>>,
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    reader: AsyncMutex<Option<(Box<dyn AsyncRead + Send + Unpin>, FrameDecoder)>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    mic: Mutex<Option<mpsc::UnboundedSender<AudioFrame>>>,
    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    mic_rx: AsyncMutex<mpsc::UnboundedReceiver<AudioFrame>>,
    tracker: Arc<PlaybackTracker>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    playback: Mutex<Playback>,
}

/// Where the CLI is: its `host:port`, or a stream already open to it.
enum Socket {
    Addr(String),
    Open(Box<dyn Stream>),
}

trait Stream: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin> Stream for T {}

/// The segment being played: when its first frame went out (Unix seconds) and how much audio it
/// holds.
#[derive(Default)]
struct Playback {
    started: Option<f64>,
    pushed: Duration,
}

impl Playback {
    /// Close the segment at `now` (Unix seconds); `Some(position)` when it held audio.
    fn take(&mut self, interrupted: bool, now: f64) -> Option<Duration> {
        let started = self.started.take()?;
        let pushed = std::mem::take(&mut self.pushed);
        let played = Duration::try_from_secs_f64(now - started).unwrap_or_default();
        Some(if interrupted { played.min(pushed) } else { pushed })
    }
}

/// The `lk agent console` socket: the session takes it as its audio input and output and as its
/// remote session's transport and console.
#[lk_agents_abi::host(object(
    name = "ConsoleIo",
    cell = "arc",
    as_dyn(
        SessionTransport = as_transport,
        ConsoleSink = as_console,
        AudioInput = as_audio_input,
        AudioOutput = as_audio_output
    )
))]
impl ConsoleIo {
    /// `addr` is `host:port`, the CLI's `--connect-addr`.
    #[host(new)]
    pub fn new(addr: String, #[host(ambient)] spawner: Arc<dyn Spawner>) -> Self {
        Self::on(Socket::Addr(addr), spawner)
    }
}

impl ConsoleIo {
    /// Over a stream already open to the CLI (an in-process one, say).
    pub fn over(
        stream: impl AsyncRead + AsyncWrite + Send + Unpin + 'static,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        Self::on(Socket::Open(Box::new(stream)), spawner)
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn on(socket: Socket, spawner: Arc<dyn Spawner>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let (mic, mic_rx) = mpsc::unbounded_channel();
        ConsoleIo {
            socket: Mutex::new(Some(socket)),
            spawner,
            outgoing: Mutex::new(Some(tx)),
            outgoing_rx: Mutex::new(Some(rx)),
            reader: AsyncMutex::new(None),
            mic: Mutex::new(Some(mic)),
            mic_rx: AsyncMutex::new(mic_rx),
            tracker: Arc::new(PlaybackTracker::new()),
            playback: Mutex::new(Playback::default()),
        }
    }

    pub fn as_transport(self: &Arc<Self>) -> Arc<dyn SessionTransport> {
        self.clone()
    }

    pub fn as_console(self: &Arc<Self>) -> Arc<dyn ConsoleSink> {
        self.clone()
    }

    pub fn as_audio_input(self: &Arc<Self>) -> Arc<dyn AudioInput> {
        self.clone()
    }

    pub fn as_audio_output(self: &Arc<Self>) -> Arc<dyn AudioOutput> {
        self.clone()
    }

    fn send(&self, message: Message) {
        let msg = proto::AgentSessionMessage { message: Some(message) };
        if let Some(tx) = self.outgoing.lock().unwrap().as_ref() {
            let _ = tx.send(frame(&msg.encode_to_vec()));
        }
    }

    fn finish(&self, interrupted: bool) {
        let position = self.playback.lock().unwrap().take(interrupted, lk_agents_abi::unix_now());
        if let Some(position) = position {
            self.tracker.playback_finished(PlaybackFinished {
                playback_position: position,
                interrupted,
                synchronized_transcript: None,
            });
        }
    }
}

#[async_trait]
impl SessionTransport for ConsoleIo {
    async fn start(&self) -> Result<(), HostError> {
        let Some(mut rx) = self.outgoing_rx.lock().unwrap().take() else { return Ok(()) };
        let socket = self.socket.lock().unwrap().take();
        let stream: Box<dyn Stream> = match socket {
            Some(Socket::Open(stream)) => stream,
            Some(Socket::Addr(addr)) => {
                let stream = TcpStream::connect(&addr)
                    .await
                    .map_err(|e| HostError::new(format!("console: connecting to {addr}: {e}")))?;
                let _ = stream.set_nodelay(true);
                Box::new(stream)
            }
            None => return Ok(()),
        };
        let (read, mut write) = tokio::io::split(stream);
        *self.reader.lock().await = Some((Box::new(read), FrameDecoder::new()));
        self.spawner.spawn_unboxed(async move {
            while let Some(bytes) = rx.recv().await {
                if write.write_all(&bytes).await.is_err() {
                    return;
                }
            }
            let _ = write.shutdown().await;
        });
        Ok(())
    }

    async fn send_message(&self, msg: &proto::AgentSessionMessage) -> Result<(), HostError> {
        self.send_message_nowait(msg);
        Ok(())
    }

    fn send_message_nowait(&self, msg: &proto::AgentSessionMessage) {
        if let Some(message) = msg.message.clone() {
            self.send(message);
        }
    }

    /// End of stream also ends the microphone.
    async fn recv(&self) -> Result<Option<proto::AgentSessionMessage>, HostError> {
        let mut reader = self.reader.lock().await;
        let Some((read, decoder)) = reader.as_mut() else { return Ok(None) };
        let mut chunk = [0u8; 16 * 1024];
        loop {
            match decoder.next_frame() {
                Ok(Some(body)) => {
                    return proto::AgentSessionMessage::decode(body.as_slice())
                        .map(Some)
                        .map_err(|e| HostError::new(format!("console: bad message: {e}")));
                }
                Ok(None) => {}
                Err(e) => return Err(HostError::new(format!("console: {e}"))),
            }
            match read.read(&mut chunk).await {
                Ok(0) | Err(_) => {
                    self.mic.lock().unwrap().take();
                    return Ok(None);
                }
                Ok(n) => decoder.push(&chunk[..n]),
            }
        }
    }

    async fn close(&self) {
        self.outgoing.lock().unwrap().take();
        self.mic.lock().unwrap().take();
    }
}

impl ConsoleSink for ConsoleIo {
    fn push_input_frame(&self, f: &console_io::AudioFrame) {
        let frame = AudioFrame::new(
            f.data.clone(),
            f.sample_rate,
            f.num_channels as u16,
            f.samples_per_channel,
        );
        if let Some(mic) = self.mic.lock().unwrap().as_ref() {
            let _ = mic.send(frame);
        }
    }

    fn playout_finished(&self) {
        self.finish(false);
    }
}

#[async_trait]
impl AudioInput for ConsoleIo {
    fn label(&self) -> &str {
        "Console"
    }

    async fn next_frame(&self) -> lk_agents_abi::voice::io::Result<Option<AudioFrame>> {
        Ok(self.mic_rx.lock().await.recv().await)
    }
}

#[async_trait]
impl AudioOutput for ConsoleIo {
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
        Some(WIRE_SAMPLE_RATE)
    }

    async fn capture(&self, f: &AudioFrame) -> Result<(), IoError> {
        let now = lk_agents_abi::unix_now();
        let first = {
            let mut playback = self.playback.lock().unwrap();
            playback.pushed += Duration::from_secs_f64(f.duration());
            let first = playback.started.is_none();
            playback.started.get_or_insert(now);
            first
        };
        if first {
            self.tracker.report_started(now);
        }
        self.send(Message::AudioOutput(console_io::AudioFrame {
            data: f.data.to_vec(),
            sample_rate: f.sample_rate,
            num_channels: u32::from(f.num_channels),
            samples_per_channel: f.samples_per_channel,
        }));
        Ok(())
    }

    fn flush_segment(&self) {
        self.send(Message::AudioPlaybackFlush(console_io::AudioPlaybackFlush {}));
    }

    fn clear_buffer(&self) {
        self.send(Message::AudioPlaybackClear(console_io::AudioPlaybackClear {}));
        self.finish(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::voice::io::AudioOutputExt;
    use tokio::net::TcpListener;

    fn message(m: Message) -> Vec<u8> {
        frame(&proto::AgentSessionMessage { message: Some(m) }.encode_to_vec())
    }

    // Catches: the CLI's frames not reaching the session's microphone, the speaker's audio and
    // flush not reaching the CLI, a playout ack not finishing the segment, or the socket's end
    // not ending the microphone.
    #[tokio::test]
    async fn the_socket_carries_the_call_both_ways() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let io = Arc::new(ConsoleIo::new(addr, Arc::new(lk_agents_abi::TokioSpawner::current())));
        io.start().await.unwrap();
        let (mut cli, _) = listener.accept().await.unwrap();

        let mic = console_io::AudioFrame {
            data: vec![0; 960],
            sample_rate: 48_000,
            num_channels: 1,
            samples_per_channel: 480,
        };
        cli.write_all(&message(Message::AudioInput(mic))).await.unwrap();
        let Some(Message::AudioInput(f)) = io.recv().await.unwrap().unwrap().message else {
            panic!("not the microphone's frame")
        };
        io.push_input_frame(&f);
        assert_eq!(io.next_frame().await.unwrap().unwrap().samples_per_channel, 480);

        io.capture_frame(&AudioFrame::new(vec![0u8; 960], 48_000, 1, 480)).await.unwrap();
        AudioOutputExt::flush(&*io);
        let mut decoder = FrameDecoder::new();
        let mut kinds = Vec::new();
        let mut chunk = [0u8; 4096];
        while kinds.len() < 2 {
            let n = cli.read(&mut chunk).await.unwrap();
            decoder.push(&chunk[..n]);
            while let Some(body) = decoder.next_frame().unwrap() {
                kinds.push(proto::AgentSessionMessage::decode(body.as_slice()).unwrap().message);
            }
        }
        assert!(matches!(kinds[0], Some(Message::AudioOutput(_))));
        assert!(matches!(kinds[1], Some(Message::AudioPlaybackFlush(_))));

        io.playout_finished();
        let done = io.wait_for_playout().await;
        assert!(!done.interrupted);
        assert_eq!(done.playback_position, std::time::Duration::from_millis(10));

        drop(cli);
        assert!(io.recv().await.unwrap().is_none());
        assert!(io.next_frame().await.unwrap().is_none());
    }
}
