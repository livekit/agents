//! Gateway I/O executor for the synchronous synthesis protocol.

use super::machine::{TtsSynthesis, TtsSynthesisEvent};
use super::*;
use crate::effects::TtsSynthesisEffect;
use base64::Engine as _;
use futures_util::future::{Either, select};
use lk_agents_abi::Ts;
use lk_agents_abi::errors::TimeoutPhase;
use lk_agents_abi::metrics::base::ConnectionAcquired;
use lk_agents_net::transport::{TransportError, WsMessage, connect_error};
use std::pin::pin;

/// How a synthesis on a connection ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SynthesisEnd {
    /// The gateway finished it: the connection can serve the next synthesis.
    Completed,
    /// The caller stopped it or stopped listening.
    Stopped,
}

/// What carrying out one effect leads to.
enum Flow {
    More(Vec<TtsSynthesisEffect>),
    End(SynthesisEnd),
}

/// Whether the caller can still send text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextInput {
    Open,
    Ended,
}

/// What goes to the gateway as input.
enum Outbound {
    Transcript(String),
    FinalFlush,
}

impl Pump {
    pub(super) async fn run(mut self) {
        let done = self.done.take().expect("pump completion owned once");
        let outcome = self.synthesize().await;
        if let Err(error) = &outcome {
            self.publish_failure(error);
        }
        let _ = done.send(outcome);
    }

    /// One synthesis: a connection reused or opened, driven to its end, then given back or
    /// closed. A stop or a closed output while opening ends it quietly.
    async fn synthesize(&mut self) -> Result<(), ApiError> {
        let request_id =
            create_inference_request_id(self.session_id.as_deref(), RequestKind::Tts, "");
        let mut core = TtsSynthesis::new(
            request_id,
            self.sample_rate,
            self.opts.conn_options.read_timeout.map(Duration::from_secs_f64),
            self.opts
                .conn_options
                .total_timeout
                .map(|timeout| self.accepted_at.saturating_add(Duration::from_secs_f64(timeout))),
        );
        let configuration = connection_configuration(&self.create_frame, &self.headers);
        let reused = self.reuse_connection(&configuration).await;
        let total_deadline = core.total_deadline();
        let opening = async {
            match core.handle(TtsSynthesisEvent::Start, self.clock.now()).into_iter().next() {
                Some(TtsSynthesisEffect::TtsSynthesisStart) => {}
                Some(TtsSynthesisEffect::Finished(Err(error))) => return Err(error),
                _ => return Err(ApiError::new("TTS synthesis was not authorized")),
            }
            let Some(reused) = reused else {
                let acquiring = self.clock.now();
                let (sender, receiver) = open_ws(
                    self.ws.as_ref(),
                    &self.url,
                    self.headers.clone(),
                    self.clock.as_ref(),
                    &self.opts.conn_options,
                    WHAT,
                    &self.create_frame,
                )
                .await?;
                let now = self.clock.now();
                let id = self.connections.lock().opened(now.as_secs_f64());
                let session_id = self.session_id.clone();
                let connection = Connection { sender, receiver, session_id, configuration };
                let acquire_time = now.saturating_duration_since(acquiring).as_secs_f64();
                return Ok(((id, connection), ConnectionAcquired { acquire_time, reused: false }));
            };
            Ok((reused, ConnectionAcquired { acquire_time: 0.0, reused: true }))
        };
        let total_timer = async {
            match total_deadline {
                Some(at) => self.clock.wait_until(at).await,
                None => std::future::pending::<()>().await,
            }
        };
        let opened = match select(
            pin!(select(pin!(self.stopped.wait_for(|stop| *stop)), pin!(self.audio_tx.closed()))),
            pin!(select(pin!(total_timer), pin!(opening))),
        )
        .await
        {
            Either::Left(_) => return Ok(()),
            Either::Right((Either::Left(_), _)) => {
                return Err(ApiError::timed_out(TimeoutPhase::Total));
            }
            Either::Right((Either::Right((opened, _)), _)) => opened,
        };
        let ((id, mut connection), acquired) = opened?;
        core.connected(acquired);
        let outcome =
            self.drive(&mut core, connection.receiver.as_mut(), connection.sender.as_ref()).await;
        if let Err(error) = &outcome {
            self.publish_failure(error);
        }
        self.release_connection(id, connection, &outcome).await;
        outcome.map(drop)
    }

    /// Takes a pooled connection opened for this configuration; the expired ones, and one opened
    /// for another configuration, are closed.
    async fn reuse_connection(&self, configuration: &str) -> Option<(u64, Connection)> {
        let (mut reused, mut expired) =
            self.connections.lock().take(self.clock.now().as_secs_f64());
        if let Some((id, connection)) = reused.take_if(|(_, connection)| {
            connection.configuration != configuration || connection.session_id != self.session_id
        }) {
            self.connections.lock().forget(id);
            expired.push(connection);
        }
        for connection in expired {
            connection.sender.close().await;
        }
        reused
    }

    /// A completed synthesis gives the connection back while the stream is live; otherwise it
    /// is closed.
    async fn release_connection(
        &self,
        id: u64,
        connection: Connection,
        outcome: &Result<SynthesisEnd, ApiError>,
    ) {
        if matches!(outcome, Ok(SynthesisEnd::Completed))
            && !*self.stopped.borrow()
            && !self.audio_tx.is_closed()
        {
            self.connections.lock().give_back(id, connection);
            return;
        }
        self.connections.lock().forget(id);
        connection.sender.close().await;
    }

    fn publish_failure(&self, error: &ApiError) {
        self.failure.send_if_modified(|failure| {
            if failure.is_some() {
                return false;
            }
            *failure = Some(error.clone());
            true
        });
    }

    async fn drive(
        &mut self,
        core: &mut TtsSynthesis,
        receiver: &mut dyn WsReceiver,
        sender: &dyn WsSender,
    ) -> Result<SynthesisEnd, ApiError> {
        let mut effects = std::collections::VecDeque::new();
        let mut input = TextInput::Open;
        loop {
            while let Some(effect) = effects.pop_front() {
                match self.apply_effect(core, sender, effect).await? {
                    Flow::More(more) => effects.extend(more),
                    Flow::End(end) => return Ok(end),
                }
            }
            let event = self.next_event(core, receiver, input).await;
            if matches!(event, TtsSynthesisEvent::InputEnded) {
                input = TextInput::Ended;
            }
            effects.extend(core.handle(event, self.clock.now()));
        }
    }

    async fn apply_effect(
        &mut self,
        core: &mut TtsSynthesis,
        sender: &dyn WsSender,
        effect: TtsSynthesisEffect,
    ) -> Result<Flow, ApiError> {
        let more = match effect {
            TtsSynthesisEffect::TtsSynthesisStart => Vec::new(),
            TtsSynthesisEffect::TtsTextSend(text) => {
                self.send_input(core, sender, Outbound::Transcript(text)).await
            }
            TtsSynthesisEffect::TtsFinalFlush => {
                self.send_input(core, sender, Outbound::FinalFlush).await
            }
            TtsSynthesisEffect::TtsAlignment { segment_id, tokens } => {
                self.transcript.push_alignment(segment_id, tokens);
                Vec::new()
            }
            TtsSynthesisEffect::TtsAudio(audio) => return self.deliver_audio(core, audio).await,
            TtsSynthesisEffect::Finished(Err(error)) => return Err(error),
            TtsSynthesisEffect::Finished(Ok(())) => return Ok(Flow::End(SynthesisEnd::Completed)),
            TtsSynthesisEffect::TtsCancel => return Ok(Flow::End(SynthesisEnd::Stopped)),
        };
        Ok(Flow::More(more))
    }

    /// Sends text or the final flush; the machine hears when it went out or why it failed.
    async fn send_input(
        &mut self,
        core: &mut TtsSynthesis,
        sender: &dyn WsSender,
        input: Outbound,
    ) -> Vec<TtsSynthesisEffect> {
        let (text, final_flush) = match input {
            Outbound::Transcript(text) => {
                (wire::input_transcript(&text, &self.opts).to_string(), false)
            }
            Outbound::FinalFlush => (wire::session_flush().to_string(), true),
        };
        let clock = self.clock.as_ref();
        let total_deadline = core.total_deadline();
        let result =
            Self::send(sender, text, &self.opts, clock, total_deadline, &mut self.stopped).await;
        let event = match result {
            Ok(()) => TtsSynthesisEvent::InputSent {
                final_flush,
                started_time: lk_agents_abi::unix_secs(clock.wall()),
            },
            Err(error) => TtsSynthesisEvent::Failed(error),
        };
        core.handle(event, clock.now())
    }

    /// Hands audio to the caller, waiting while the caller is behind, up to the total deadline.
    async fn deliver_audio(
        &mut self,
        core: &mut TtsSynthesis,
        mut audio: SynthesizedAudio,
    ) -> Result<Flow, ApiError> {
        audio.timed_transcript =
            self.transcript.take_audio_transcript(&audio.segment_id, audio.is_final);
        let clock = self.clock.as_ref();
        let blocked_at = clock.now();
        let total_timer = async {
            match core.total_deadline() {
                Some(at) => clock.wait_until(at).await,
                None => std::future::pending::<()>().await,
            }
        };
        let sent = match select(
            pin!(self.stopped.wait_for(|stop| *stop)),
            pin!(select(pin!(total_timer), pin!(self.audio_tx.send(Ok(audio))))),
        )
        .await
        {
            Either::Left(_) => return Ok(Flow::End(SynthesisEnd::Stopped)),
            Either::Right((Either::Left(_), _)) => {
                return Err(ApiError::timed_out(TimeoutPhase::Total));
            }
            Either::Right((Either::Right((sent, _)), _)) => sent,
        };
        if sent.is_err() {
            return Ok(Flow::End(SynthesisEnd::Stopped));
        }
        core.delivery_resumed(blocked_at, clock.now());
        Ok(Flow::More(Vec::new()))
    }

    /// The next event for the machine: a stop or a closed output, a deadline, then the caller's
    /// text or the gateway's message.
    async fn next_event(
        &mut self,
        core: &TtsSynthesis,
        receiver: &mut dyn WsReceiver,
        input: TextInput,
    ) -> TtsSynthesisEvent {
        let clock = self.clock.as_ref();
        let deadline = core.deadline();
        let timer = async {
            match deadline {
                Some(at) => clock.wait_until(at).await,
                None => std::future::pending::<()>().await,
            }
        };
        let text_rx = &mut self.text_rx;
        let command = async {
            if input == TextInput::Ended {
                return std::future::pending::<TtsSynthesisEvent>().await;
            }
            command_event(text_rx.recv().await)
        };
        match select(
            pin!(select(
                pin!(select(
                    pin!(self.stopped.wait_for(|stop| *stop)),
                    pin!(self.audio_tx.closed())
                )),
                pin!(timer),
            )),
            pin!(select(pin!(command), pin!(receiver.recv()))),
        )
        .await
        {
            Either::Left((Either::Left(_), _)) => TtsSynthesisEvent::Cancel,
            Either::Left((Either::Right(_), _)) => TtsSynthesisEvent::Wake,
            Either::Right((Either::Left((command, _)), _)) => command,
            Either::Right((Either::Right((received, _)), _)) => {
                received_event(received, &self.provider)
            }
        }
    }

    async fn send(
        sender: &dyn WsSender,
        text: String,
        opts: &TtsOptions,
        clock: &dyn Clock,
        deadline: Option<Ts>,
        stopped: &mut watch::Receiver<bool>,
    ) -> Result<(), ApiError> {
        let write_deadline = opts
            .conn_options
            .write_timeout
            .map(|seconds| clock.now().saturating_add(Duration::from_secs_f64(seconds)));
        let total_deadline = deadline;
        let deadline = deadline.into_iter().chain(write_deadline).min();
        let timer = async {
            match deadline {
                Some(at) => clock.wait_until(at).await,
                None => std::future::pending::<()>().await,
            }
        };
        match select(
            pin!(stopped.wait_for(|stop| *stop)),
            pin!(select(pin!(timer), pin!(sender.send_text(text)))),
        )
        .await
        {
            Either::Left(_) => Err(ApiError::new("TTS synthesis cancelled")),
            Either::Right((Either::Left(_), _)) => {
                Err(ApiError::timed_out(if total_deadline.is_some_and(|at| clock.now() >= at) {
                    TimeoutPhase::Total
                } else {
                    TimeoutPhase::Write
                }))
            }
            Either::Right((Either::Right((result, _)), _)) => {
                result.map_err(|error| connect_error(WHAT, error))
            }
        }
    }
}

/// The caller's text command as a machine event.
fn command_event(command: Option<TextCmd>) -> TtsSynthesisEvent {
    match command {
        Some(TextCmd::Push(text)) => TtsSynthesisEvent::Text(text),
        Some(TextCmd::Failed) => TtsSynthesisEvent::Failed(
            ApiError::new("TTS input queue overflowed at EOF").with_retryable(false),
        ),
        Some(TextCmd::End) | None => TtsSynthesisEvent::InputEnded,
    }
}

/// A gateway message as a machine event, with its audio decoded.
fn received_event(
    received: Result<WsMessage, TransportError>,
    provider: &str,
) -> TtsSynthesisEvent {
    let raw = match received {
        Err(error) => return TtsSynthesisEvent::Failed(connect_error(WHAT, error)),
        Ok(WsMessage::Closed) => return TtsSynthesisEvent::ServerEof,
        Ok(WsMessage::Binary(_) | WsMessage::Other) => return TtsSynthesisEvent::Wake,
        Ok(WsMessage::Text(raw)) => raw,
    };
    match wire::decode(&raw, provider) {
        Some(wire::Incoming::OutputAudio { session_id, audio }) => {
            match base64::engine::general_purpose::STANDARD.decode(audio) {
                Ok(data) => TtsSynthesisEvent::Audio { session_id, data },
                Err(_) => {
                    TtsSynthesisEvent::Failed(ApiError::new("TTS returned malformed base64 audio"))
                }
            }
        }
        Some(incoming) => TtsSynthesisEvent::Received(incoming),
        None => TtsSynthesisEvent::Failed(ApiError::new("TTS returned malformed protocol data")),
    }
}
