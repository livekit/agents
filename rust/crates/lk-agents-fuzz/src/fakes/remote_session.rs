//! The console's remote host over fake transport and session ports.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use lk_agents_abi::voice::console::ConsolePort;
use lk_agents_abi::voice::events::{HostWire, ToolsExecutedEvent};
use lk_agents_abi::worker::remote_session::{ErrorWire, SessionHosting, SessionPort, WireEvent};
use lk_agents_abi::{BoxFuture, Spawner, TaskSet, errors::HostError};
use lk_agents_voice::agent_session::Timebase;
use lk_agents_voice::agent_session::api::RemoteSessionConfig;
use lk_agents_voice::remote_session::{
    SessionReply, encode_session_reply, encode_wire, internal_error_response, io_update, proto,
    session_event, stamp_event, tools_executed,
};

struct Hosting {
    console: Arc<dyn ConsolePort>,
    port: Arc<dyn SessionPort>,
    time: Timebase,
    spawner: Arc<dyn Spawner>,
    reader: Arc<TaskSet>,
    handlers: Arc<TaskSet>,
    started: AtomicBool,
    closing: AtomicBool,
    closed: tokio::sync::watch::Sender<bool>,
}

pub fn hosting(
    config: RemoteSessionConfig,
    time: Timebase,
    spawner: Arc<dyn Spawner>,
) -> Arc<dyn SessionHosting> {
    Arc::new(Hosting {
        console: config.console.expect("the simulation hosts remote requests on a console"),
        port: config.port,
        time,
        spawner,
        reader: Arc::default(),
        handlers: Arc::default(),
        started: AtomicBool::new(false),
        closing: AtomicBool::new(false),
        closed: tokio::sync::watch::channel(false).0,
    })
}

impl Hosting {
    fn send(&self, event: WireEvent, at: Option<f64>) {
        let Some(event) = encode_wire(event).expect("valid fake session event") else { return };
        let event = proto::AgentSessionEvent { created_at: None, event: Some(event) };
        let message = stamp_event(event, at.unwrap_or_else(|| self.time.wall_secs()))
            .expect("finite simulation time");
        self.console.clone().as_transport(self.spawner.clone()).send_message_nowait(&message);
    }
}

impl SessionHosting for Hosting {
    fn forward(&self, event: &HostWire) {
        if let Some((event, at)) = session_event(event) {
            self.send(event, at);
        }
    }

    fn forward_error(&self, message: String) {
        self.send(WireEvent::Error(ErrorWire { message }), None);
    }

    fn forward_tools_executed(&self, event: &ToolsExecutedEvent) {
        self.send(tools_executed(event), None);
    }

    fn start(&self) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            if self.closing.load(Ordering::Acquire) || self.started.swap(true, Ordering::AcqRel) {
                return Ok(());
            }
            let transport = self.console.clone().as_transport(self.spawner.clone());
            transport.start().await?;
            if self.closing.load(Ordering::Acquire) {
                return Ok(());
            }
            let console = self.console.clone().as_console();
            let (port, time) = (self.port.clone(), self.time.clone());
            let (handlers, spawner) = (self.handlers.clone(), self.spawner.clone());
            self.reader.spawn(&*self.spawner, "console receive", async move {
                while let Ok(Some(message)) = transport.recv().await {
                    use proto::agent_session_message::Message;
                    match message.message {
                        Some(Message::Request(request)) => {
                            let (port, time, transport) =
                                (port.clone(), time.clone(), transport.clone());
                            handlers.spawn(&*spawner, "console request", async move {
                                let response = answer(&*port, &request, &time).await;
                                let message = match response {
                                    Ok(Some(message)) => message,
                                    Ok(None) => return,
                                    Err(_) => internal_error_response(&request.request_id),
                                };
                                let _ = transport.send_message(&message).await;
                            });
                        }
                        Some(Message::AudioInput(frame)) => console.push_input_frame(&frame),
                        Some(Message::AudioPlaybackFinished(_)) => console.playout_finished(),
                        _ => {}
                    }
                }
            });
            Ok(())
        })
    }

    fn aclose(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            if !self.closing.swap(true, Ordering::AcqRel) {
                let (reader, handlers) = (self.reader.clone(), self.handlers.clone());
                let (clock, transport) =
                    (self.time.clock(), self.console.clone().as_transport(self.spawner.clone()));
                let closed = self.closed.clone();
                self.spawner.spawn(Box::pin(async move {
                    reader.cancel_and_wait(&*clock).await;
                    let deadline = clock.now().saturating_add(Duration::from_secs(3));
                    handlers.drain(&*clock, deadline).await;
                    transport.close().await;
                    closed.send_replace(true);
                }));
            }
            let _ = self.closed.subscribe().wait_for(|closed| *closed).await;
        })
    }
}

async fn answer(
    port: &dyn SessionPort,
    request: &proto::SessionRequest,
    time: &Timebase,
) -> Result<Option<proto::AgentSessionMessage>, HostError> {
    use proto::session_request::Request;
    let Some(operation) = &request.request else { return Ok(None) };
    let reply = match operation {
        Request::Ping(_) => SessionReply::Pong,
        Request::GetChatHistory(_) => SessionReply::ChatHistory(port.chat_history().await?),
        Request::GetAgentInfo(_) => SessionReply::AgentInfo(port.agent_info().await?),
        Request::RunInput(input) if input.text.is_empty() => {
            SessionReply::RunInput((None, Some("empty run_input text".to_owned())))
        }
        Request::RunInput(input) => SessionReply::RunInput(port.run_input(&input.text).await?),
        Request::GetSessionState(_) => SessionReply::SessionState(port.session_state().await?),
        Request::GetRtcStats(_) => SessionReply::RtcStats(port.rtc_stats().await?),
        Request::GetSessionUsage(_) => SessionReply::SessionUsage(port.session_usage().await?),
        Request::GetFrameworkInfo(_) => {
            let (sdk, version) = port.framework_info();
            SessionReply::FrameworkInfo(sdk, version)
        }
        Request::UpdateIo(io) => {
            port.update_io(io_update(io)).await?;
            SessionReply::UpdateIo
        }
        Request::FinalizeSimulation(input) => SessionReply::FinalizeSimulation(
            port.finalize_simulation((input.provisional_success, input.provisional_reason.clone()))
                .await?,
        ),
    };
    Ok(Some(encode_session_reply(request.request_id.clone(), reply, time.wall_secs())))
}
