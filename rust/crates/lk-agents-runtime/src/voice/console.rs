//! `lk agent console`'s local call (`cli/tcp_console.py`): the CLI owns the microphone, the speakers
//! and the TUI, and reaches the agent over one TCP socket carrying length-prefixed
//! `AgentSessionMessage`s. [`ConsoleIo`] is that socket as the session sees it: the
//! [`SessionTransport`] and [`ConsoleSink`] the session's remote session host
//! ([`HostRuntime`](lk_agents_voice::remote_session::driver::HostRuntime)) runs on, and the
//! session's audio input and output legs. What the legs account for is
//! [`ConsolePlayback`]'s; only the socket is here.

use std::sync::{Arc, Mutex};

use lk_agents_abi::errors::HostError;
use lk_agents_abi::voice::console::ConsolePort;
use lk_agents_abi::voice::io::{
    AudioInput, AudioOutput, AudioOutputCapabilities, IoError, PlaybackTracker,
};
use lk_agents_abi::{AudioFrame, Spawner, SpawnerExt};
use prost::Message as _;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::net::tcp::OwnedReadHalf;
use tokio::sync::{Mutex as AsyncMutex, mpsc};

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
    outgoing: Mutex<Option<mpsc::UnboundedSender<Vec<u8>>>>,
    outgoing_rx: Mutex<Option<mpsc::UnboundedReceiver<Vec<u8>>>>,
    reader: AsyncMutex<Option<(OwnedReadHalf, FrameDecoder)>>,
    mic: Mutex<Option<mpsc::UnboundedSender<AudioFrame>>>,
    mic_rx: AsyncMutex<mpsc::UnboundedReceiver<AudioFrame>>,
    tracker: Arc<PlaybackTracker>,
    playback: Mutex<ConsolePlayback>,
}

/// The socket as the session's transport, writing on the session's spawner.
struct Transport {
    io: Arc<ConsoleIo>,
    spawner: Arc<dyn Spawner>,
}

impl lk_agents_abi::voice::console::ConsoleIo for ConsoleIo {
    fn new(addr: String) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let (mic, mic_rx) = mpsc::unbounded_channel();
        ConsoleIo {
            addr,
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
        Arc::new(Transport { io: self, spawner })
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

    fn finish(&self, interrupted: bool) {
        let finished = self.playback.lock().unwrap().finish(interrupted, lk_agents_abi::unix_now());
        if let Some(finished) = finished {
            self.tracker.playback_finished(finished);
        }
    }
}

impl SessionTransport for Transport {
    fn start(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            let Some(mut rx) = self.io.outgoing_rx.lock().unwrap().take() else { return Ok(()) };
            let addr = &self.io.addr;
            let stream = TcpStream::connect(addr)
                .await
                .map_err(|e| HostError::new(format!("console: connecting to {addr}: {e}")))?;
            let _ = stream.set_nodelay(true);
            let (read, mut write) = stream.into_split();
            *self.io.reader.lock().await = Some((read, FrameDecoder::new()));
            self.spawner.spawn_unboxed(async move {
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
        self.finish(false);
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
            let now = lk_agents_abi::unix_now();
            let (message, opened) = self.playback.lock().unwrap().capture(f, now);
            if opened {
                self.tracker.report_started(now);
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
        self.finish(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::voice::console::ConsoleIo as _;
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
        let io = Arc::new(ConsoleIo::new(addr));
        let transport = io.clone().as_transport(Arc::new(lk_agents_abi::TokioSpawner::current()));
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
        let done = io.wait_for_playout().await;
        assert!(!done.interrupted);
        assert_eq!(done.playback_position, std::time::Duration::from_millis(10));

        drop(cli);
        assert!(transport.recv().await.unwrap().is_none());
        assert!(io.next_frame().await.unwrap().is_none());
    }
}
