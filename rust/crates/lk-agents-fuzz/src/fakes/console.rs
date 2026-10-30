//! The CLI peer's bytes and paced speaker, driving the real console transport and audio legs.

use super::Registry;
use crate::record::Sink;
use lk_agents_abi::SpawnerExt as _;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::{BoxFuture, Clock, Spawner, Ts};
use lk_agents_runtime::net::tcp::{Connection, Connector};
use lk_agents_runtime::voice::console::ConsoleIo;
use lk_agents_voice::remote_session::{self, FrameDecoder, frame, proto};
use prost::Message as _;
use proto::agent_session_message::{Message, console_io};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A single simulated TCP connection. The CLI holds the far end.
struct Socket(Mutex<Option<Connection>>);

impl Connector for Socket {
    fn connect<'a>(&'a self, _: &'a str) -> BoxFuture<'a, std::io::Result<Connection>> {
        Box::pin(async move {
            self.0
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| std::io::Error::other("console socket already opened"))
        })
    }
}

/// The CLI's microphone, requests and playout acknowledgements, in socket order. As the Go CLI's
/// speaker, it acknowledges a flush once everything queued has played, and only the last flush:
/// a newer flush or a clear drops the acknowledgement still owed.
#[derive(Clone)]
pub struct FakeCli {
    to_agent: tokio::sync::mpsc::UnboundedSender<proto::AgentSessionMessage>,
    playing_until: Arc<Mutex<Option<Ts>>>,
    clock: Arc<dyn Clock>,
}

impl FakeCli {
    /// `receipts` records each acknowledgement the CLI sends.
    pub fn connect(
        world: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
        receipts: Option<Arc<Registry>>,
    ) -> (Arc<ConsoleIo>, FakeCli) {
        let (agent, cli) = tokio::io::duplex(64 * 1024);
        let (read, write) = tokio::io::split(agent);
        let socket =
            Socket(Mutex::new(Some(Connection { read: Box::new(read), write: Box::new(write) })));
        let io = Arc::new(ConsoleIo::with_ports("console".into(), Arc::new(socket), clock.clone()));
        let (mut reader, mut writer) = tokio::io::split(cli);
        let (to_agent, mut sent) =
            tokio::sync::mpsc::unbounded_channel::<proto::AgentSessionMessage>();
        // The world's own work: the run's executor ends it with the run.
        world.spawn_detached(async move {
            while let Some(message) = sent.recv().await {
                if writer.write_all(&frame(&message.encode_to_vec())).await.is_err() {
                    return;
                }
            }
            let _ = writer.shutdown().await;
        });
        let cli = FakeCli { to_agent, playing_until: Arc::default(), clock };
        let speaker = cli.clone();
        let world_inner = world.clone();
        // The world's own work: the run's executor ends it with the run.
        world.spawn_detached(async move {
            let (mut pending, mut segment) = (Duration::ZERO, 0u64);
            let played = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let mut decoder = FrameDecoder::new();
            let mut chunk = [0u8; 16 * 1024];
            loop {
                let n = match reader.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                decoder.push(&chunk[..n]);
                while let Some(bytes) = decoder.next_frame().expect("agent console frame") {
                    let message = proto::AgentSessionMessage::decode(bytes.as_slice())
                        .expect("agent console message");
                    match message.message {
                        Some(Message::AudioOutput(f)) => {
                            pending += Duration::from_secs_f64(
                                f64::from(f.samples_per_channel) / f64::from(f.sample_rate.max(1)),
                            );
                        }
                        Some(Message::AudioPlaybackFlush(_)) => {
                            segment += 1;
                            played.store(segment, std::sync::atomic::Ordering::Relaxed);
                            let (speaker, played, this, receipts) =
                                (speaker.clone(), played.clone(), segment, receipts.clone());
                            let audio = std::mem::take(&mut pending);
                            let until = speaker.clock.now().saturating_add(audio);
                            let mut playing = speaker.playing_until.lock().unwrap();
                            *playing = Some(playing.map_or(until, |at| at.max(until)));
                            drop(playing);
                            // The world's own work: the run's executor ends it with the run.
                            world_inner.spawn_detached(async move {
                                speaker.clock.wait_until(until).await;
                                if played.load(std::sync::atomic::Ordering::Relaxed) == this {
                                    speaker.send(Message::AudioPlaybackFinished(
                                        console_io::AudioPlaybackFinished {},
                                    ));
                                    if let Some(receipts) = &receipts {
                                        receipts.record_sink(Sink::ConsoleReceipt);
                                    }
                                }
                            });
                        }
                        Some(Message::AudioPlaybackClear(_)) => {
                            // The speaker's ring is emptied: nothing queued plays on.
                            pending = Duration::ZERO;
                            *speaker.playing_until.lock().unwrap() = None;
                            segment += 1;
                            played.store(segment, std::sync::atomic::Ordering::Relaxed);
                        }
                        _ => {}
                    }
                }
            }
        });
        (io, cli)
    }

    pub fn queued(&self) -> Duration {
        let until = *self.playing_until.lock().unwrap();
        until.map_or(Duration::ZERO, |at| at.saturating_duration_since(self.clock.now()))
    }

    pub fn send(&self, message: Message) -> bool {
        self.to_agent.send(proto::AgentSessionMessage { message: Some(message) }).is_ok()
    }
}

/// What the CLI asks of the session (`ConsoleCall`'s `_core_*` port), answered as the host would
/// from its own state. Toggling the legs stays with the case's own inputs, so `update_io` is
/// acknowledged and changes nothing.
pub struct FakeConsolePort;

impl remote_session::SessionPort for FakeConsolePort {
    fn chat_history(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Vec<lk_agents_abi::ChatItem>, HostError>> {
        Box::pin(async move { Ok(Vec::new()) })
    }
    fn agent_info(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<remote_session::AgentInfo, HostError>> {
        Box::pin(async move {
            Ok(("agent".into(), Some("You are a test agent.".into()), Vec::new(), Vec::new()))
        })
    }
    fn run_input<'a>(
        &'a self,
        _text: &'a str,
    ) -> lk_agents_abi::BoxFuture<'a, Result<remote_session::RunOutcome, HostError>> {
        Box::pin(async move { Ok((Some(Vec::new()), None)) })
    }
    fn session_state(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<remote_session::SessionState, HostError>> {
        Box::pin(async move {
            let options = lk_agents_abi::worker::remote_session::STATE_OPTION_FIELDS
                .iter()
                .map(|key| ((*key).to_owned(), "None".to_owned()))
                .collect();
            Ok(("listening".into(), "listening".into(), "agent".into(), options, Some(0.0)))
        })
    }
    fn rtc_stats(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<remote_session::RtcStats>, HostError>> {
        Box::pin(async move { Ok(None) })
    }
    fn session_usage(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<lk_agents_abi::usage::Usage, HostError>> {
        Box::pin(async move { Ok(Default::default()) })
    }
    fn framework_info(&self) -> (String, String) {
        ("rust".into(), "fuzz".into())
    }
    fn update_io(
        &self,
        _update: remote_session::IoUpdate,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move { Ok(()) })
    }
    fn finalize_simulation(
        &self,
        _provisional: remote_session::SimulationVerdict,
    ) -> lk_agents_abi::BoxFuture<'_, Result<remote_session::VerdictOutcome, HostError>> {
        Box::pin(async move { Ok((None, None)) })
    }
}
