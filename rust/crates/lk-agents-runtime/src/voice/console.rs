//! `lk agent console`'s local call (`cli/tcp_console.py`): the CLI owns the microphone, the speakers
//! and the TUI, and reaches the agent over one TCP socket carrying length-prefixed
//! `AgentSessionMessage`s. [`ConsoleIo`] is that socket as the session sees it: the
//! [`SessionTransport`] and [`ConsoleSink`] the session's remote session host
//! ([`SessionServer`](super::remote_session::driver::SessionServer)) runs on, and the
//! session's audio input and output legs. What the legs account for is
//! [`ConsolePlayback`]'s; only the socket is here.

use lk_agents_net::task::TaskSet;
use std::sync::{Arc, Mutex};

use lk_agents_abi::errors::HostError;
use lk_agents_abi::voice::console::ConsolePort;
use lk_agents_abi::voice::io::{
    AudioInput, AudioOutput, AudioOutputCapabilities, IoError, PlaybackTracker,
};
use lk_agents_abi::{AudioFrame, Clock, Spawner};
use lk_agents_net::tcp::Connector;
use prost::Message as _;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::sync::{Mutex as AsyncMutex, mpsc};

use lk_agents_net::env::ProcessEnv;
use lk_agents_voice::remote_session::console::{
    CONSOLE_SAMPLE_RATE, ConsolePlayback, microphone_frame,
};
use lk_agents_voice::remote_session::{ConsoleSink, FrameDecoder, SessionTransport, frame, proto};
use proto::agent_session_message::{Message, console_io};

/// The CLI's socket: transport, console sink and both audio legs. Connected by `start()`; what is
/// sent before then is queued.
pub struct ConsoleIo {
    /// `host:port`, the CLI's `--connect-addr`.
    addr: String,
    connector: Arc<dyn Connector>,
    clock: Arc<dyn Clock>,
    outgoing: Mutex<Option<mpsc::UnboundedSender<Vec<u8>>>>,
    outgoing_rx: Mutex<Option<mpsc::UnboundedReceiver<Vec<u8>>>>,
    reader: AsyncMutex<Option<(Box<dyn AsyncRead + Unpin + Send>, FrameDecoder)>>,
    mic: Mutex<Option<mpsc::UnboundedSender<AudioFrame>>>,
    mic_rx: AsyncMutex<mpsc::UnboundedReceiver<AudioFrame>>,
    tracker: Arc<PlaybackTracker>,
    playback: Mutex<ConsolePlayback>,
}

/// The socket as the session's transport, writing on the session's spawner.
struct Transport {
    io: Arc<ConsoleIo>,
    spawner: Arc<dyn Spawner>,
    /// The socket writer, which ends once the outgoing queue closes.
    tasks: Arc<TaskSet>,
}

impl lk_agents_abi::voice::console::ConsoleIo for ConsoleIo {
    type Env = ProcessEnv;

    fn new(env: &ProcessEnv, addr: String) -> Self {
        Self::with_ports(
            addr,
            Arc::new(lk_agents_net::tcp::Tcp::new(env.runtime)),
            env.clock.clone(),
        )
    }
}

impl ConsoleIo {
    /// The console protocol over supplied byte IO and the session's clock.
    pub fn with_ports(addr: String, connector: Arc<dyn Connector>, clock: Arc<dyn Clock>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let (mic, mic_rx) = mpsc::unbounded_channel();
        ConsoleIo {
            addr,
            connector,
            clock,
            outgoing: Mutex::new(Some(tx)),
            outgoing_rx: Mutex::new(Some(rx)),
            reader: AsyncMutex::new(None),
            mic: Mutex::new(Some(mic)),
            mic_rx: AsyncMutex::new(mic_rx),
            tracker: Arc::new(PlaybackTracker::new()),
            playback: Mutex::new(ConsolePlayback::default()),
        }
    }
}

impl ConsolePort for ConsoleIo {
    fn as_transport(self: Arc<Self>, spawner: Arc<dyn Spawner>) -> Arc<dyn SessionTransport> {
        Arc::new(Transport { io: self, spawner, tasks: Arc::default() })
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

impl ConsoleIo {
    fn send(&self, message: Message) {
        let msg = proto::AgentSessionMessage { message: Some(message) };
        if let Some(tx) = self.outgoing.lock().unwrap().as_ref() {
            let _ = tx.send(frame(&msg.encode_to_vec()));
        }
    }

    fn finish(&self) {
        if let Some((segment, finished)) = self.playback.lock().unwrap().clear() {
            self.tracker.playback_finished(segment, finished);
        }
    }
}

impl SessionTransport for Transport {
    fn start(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            let Some(mut rx) = self.io.outgoing_rx.lock().unwrap().take() else {
                return Ok(());
            };
            let addr = &self.io.addr;
            let lk_agents_net::tcp::Connection { read, mut write } = self
                .io
                .connector
                .connect(addr)
                .await
                .map_err(|e| HostError::new(format!("console: connecting to {addr}: {e}")))?;
            *self.io.reader.lock().await = Some((read, FrameDecoder::new()));
            self.tasks.spawn(self.spawner.as_ref(), "console writer", async move {
                while let Some(bytes) = rx.recv().await {
                    if write.write_all(&bytes).await.is_err() {
                        return;
                    }
                }
                let _ = write.shutdown().await;
            });
            Ok(())
        })
    }

    fn send_message<'a>(
        &'a self,
        msg: &'a proto::AgentSessionMessage,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move {
            self.send_message_nowait(msg);
            Ok(())
        })
    }

    fn send_message_nowait(&self, msg: &proto::AgentSessionMessage) {
        if let Some(message) = msg.message.clone() {
            self.io.send(message);
        }
    }

    /// End of stream also ends the microphone.
    fn recv(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<proto::AgentSessionMessage>, HostError>> {
        Box::pin(async move {
            let mut reader = self.io.reader.lock().await;
            let Some((read, decoder)) = reader.as_mut() else {
                return Ok(None);
            };
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
                        self.io.mic.lock().unwrap().take();
                        return Ok(None);
                    }
                    Ok(n) => decoder.push(&chunk[..n]),
                }
            }
        })
    }

    fn close(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            self.io.outgoing.lock().unwrap().take();
            self.io.mic.lock().unwrap().take();
        })
    }
}

impl ConsoleSink for ConsoleIo {
    fn push_input_frame(&self, frame: &console_io::AudioFrame) {
        if let Some(mic) = self.mic.lock().unwrap().as_ref() {
            let _ = mic.send(microphone_frame(frame));
        }
    }

    fn playout_finished(&self) {
        log::error!(target: "livekit.agents", "console playback completion has no segment identity and was refused");
    }
}

impl AudioInput for ConsoleIo {
    fn label(&self) -> &str {
        "Console"
    }

    fn next_frame(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, lk_agents_abi::voice::io::Result<Option<AudioFrame>>> {
        Box::pin(async move { Ok(self.mic_rx.lock().await.recv().await) })
    }
}

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
        Some(CONSOLE_SAMPLE_RATE)
    }

    fn capture<'a>(
        &'a self,
        f: &'a AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), IoError>> {
        Box::pin(async move {
            let now = lk_agents_abi::unix_secs(self.clock.wall());
            let segment =
                self.tracker.current_segment().ok_or(IoError::PlaybackSegmentUnavailable)?;
            let (message, opened) = self.playback.lock().unwrap().capture(f, segment)?;
            if opened {
                self.tracker.report_started(segment, now);
            }
            self.send(message);
            Ok(())
        })
    }

    fn flush_segment(&self) {
        self.send(Message::AudioPlaybackFlush(console_io::AudioPlaybackFlush {}));
    }

    fn clear_buffer(&self) {
        self.send(Message::AudioPlaybackClear(console_io::AudioPlaybackClear {}));
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::voice::console::ConsoleIo as _;
    use lk_agents_abi::voice::io::AudioOutputExt;
    use std::time::Duration;
    use tokio::net::TcpListener;

    fn message(m: Message) -> Vec<u8> {
        frame(&proto::AgentSessionMessage { message: Some(m) }.encode_to_vec())
    }

    // Catches: the CLI's frames not reaching the session's microphone, the speaker's audio and
    // flush not reaching the CLI, an unkeyed ack finishing a segment, or the socket's end
    // not ending the microphone.
    #[tokio::test]
    async fn the_socket_carries_the_call_both_ways() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let io = Arc::new(ConsoleIo::new(&ProcessEnv::current_runtime(), addr));
        let transport =
            io.clone().as_transport(Arc::new(lk_agents_net::task::TokioSpawner::current()));
        transport.start().await.unwrap();
        let (mut cli, _) = listener.accept().await.unwrap();

        let mic = console_io::AudioFrame {
            data: vec![0; 960],
            sample_rate: 48_000,
            num_channels: 1,
            samples_per_channel: 480,
        };
        cli.write_all(&message(Message::AudioInput(mic))).await.unwrap();
        let Some(Message::AudioInput(f)) = transport.recv().await.unwrap().unwrap().message else {
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
        assert!(
            tokio::time::timeout(Duration::from_millis(10), io.wait_for_playout()).await.is_err(),
            "an unidentified acknowledgement must not settle the admitted segment"
        );
        io.clear_buffer();
        let done = io.wait_for_playout().await;
        assert!(done.interrupted);
        assert_eq!(
            done.playback_position,
            Duration::ZERO,
            "a local clear has no client playback receipt"
        );

        io.capture_frame(&AudioFrame::new(vec![0u8; 960], 48_000, 1, 480)).await.unwrap();
        AudioOutputExt::flush(&*io);
        io.playout_finished();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), io.wait_for_playout()).await.is_err(),
            "an old unidentified acknowledgement settled the replacement segment"
        );
        io.clear_buffer();
        assert!(io.wait_for_playout().await.interrupted);

        drop(cli);
        assert!(transport.recv().await.unwrap().is_none());
        assert!(io.next_frame().await.unwrap().is_none());
    }
}
