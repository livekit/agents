//! Gateway I/O executor for the synchronous synthesis protocol.

use super::machine::{TtsSynthesis, TtsSynthesisEvent};
use super::*;
use crate::effects::TtsSynthesisEffect;
use base64::Engine as _;
use futures_util::future::{Either, select};
use lk_agents_abi::Ts;
use lk_agents_abi::errors::TimeoutPhase;
use lk_agents_net::transport::{WsMessage, connect_error};
use std::pin::pin;

impl Pump {
    pub(super) async fn run(mut self) {
        let done = self.done.take().expect("pump completion owned once");
        let outcome = async {
            let request_id =
                create_inference_request_id(self.session_id.as_deref(), RequestKind::Tts, "");
            let mut core = TtsSynthesis::new(
                request_id,
                self.sample_rate,
                self.opts.conn_options.read_timeout.map(Duration::from_secs_f64),
                self.opts.conn_options.total_timeout.map(|timeout| {
                    self.accepted_at.saturating_add(Duration::from_secs_f64(timeout))
                }),
            );
            let configuration = connection_configuration(&self.create_frame, &self.headers);
            let (mut reused, mut expired) =
                self.connections.lock().take(self.clock.now().as_secs_f64());
            if reused.as_ref().is_some_and(|(_, connection)| {
                connection.configuration != configuration
                    || connection.session_id != self.session_id
            }) {
                let (id, connection) = reused.take().unwrap();
                self.connections.lock().forget(id);
                expired.push(connection);
            }
            for connection in expired {
                connection.sender.close().await;
            }
            let total_deadline = core.total_deadline();
            let opening = async {
                match core.handle(TtsSynthesisEvent::Start, self.clock.now()).into_iter().next() {
                    Some(TtsSynthesisEffect::TtsSynthesisStart) => {}
                    Some(TtsSynthesisEffect::Finished(Err(error))) => return Err(error),
                    _ => return Err(ApiError::new("TTS synthesis was not authorized")),
                }
                match reused {
                    Some(reused) => Ok(reused),
                    None => {
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
                        let id = self.connections.lock().opened(self.clock.now().as_secs_f64());
                        Ok((
                            id,
                            Connection {
                                sender,
                                receiver,
                                session_id: self.session_id.clone(),
                                configuration,
                            },
                        ))
                    }
                }
            };
            let total_timer = async {
                match total_deadline {
                    Some(at) => self.clock.wait_until(at).await,
                    None => std::future::pending::<()>().await,
                }
            };
            let opened = match select(
                pin!(select(
                    pin!(self.stopped.wait_for(|stop| *stop)),
                    pin!(self.audio_tx.closed())
                )),
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
            let (id, mut connection) = match opened {
                Ok(opened) => opened,
                Err(error) => return Err(error),
            };
            let outcome = self
                .drive(&mut core, connection.receiver.as_mut(), connection.sender.as_ref())
                .await;
            if let Err(error) = &outcome {
                self.publish_failure(error);
            }
            if matches!(outcome, Ok(true)) && !*self.stopped.borrow() && !self.audio_tx.is_closed()
            {
                self.connections.lock().give_back(id, connection);
            } else {
                self.connections.lock().forget(id);
                connection.sender.close().await;
            }
            outcome.map(|_| ())
        }
        .await;
        if let Err(error) = &outcome {
            self.publish_failure(error);
        }
        let _ = done.send(outcome);
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
    ) -> Result<bool, ApiError> {
        let opts = &self.opts;
        let provider = &self.provider;
        let clock = self.clock.as_ref();
        let text_rx = &mut self.text_rx;
        let audio_tx = &self.audio_tx;
        let stopped = &mut self.stopped;
        let transcript = &mut self.transcript;
        let mut effects = std::collections::VecDeque::new();
        let mut input_closed = false;
        loop {
            while let Some(effect) = effects.pop_front() {
                match effect {
                    TtsSynthesisEffect::TtsSynthesisStart => {}
                    TtsSynthesisEffect::TtsTextSend(text) => {
                        let result = Self::send(
                            sender,
                            wire::input_transcript(&text, opts).to_string(),
                            opts,
                            clock,
                            core.total_deadline(),
                            stopped,
                        )
                        .await;
                        effects.extend(match result {
                            Ok(()) => core.handle(
                                TtsSynthesisEvent::InputSent {
                                    final_flush: false,
                                    started_time: lk_agents_abi::unix_secs(clock.wall()),
                                },
                                clock.now(),
                            ),
                            Err(error) => {
                                core.handle(TtsSynthesisEvent::Failed(error), clock.now())
                            }
                        });
                    }
                    TtsSynthesisEffect::TtsFinalFlush => {
                        let result = Self::send(
                            sender,
                            wire::session_flush().to_string(),
                            opts,
                            clock,
                            core.total_deadline(),
                            stopped,
                        )
                        .await;
                        effects.extend(match result {
                            Ok(()) => core.handle(
                                TtsSynthesisEvent::InputSent {
                                    final_flush: true,
                                    started_time: lk_agents_abi::unix_secs(clock.wall()),
                                },
                                clock.now(),
                            ),
                            Err(error) => {
                                core.handle(TtsSynthesisEvent::Failed(error), clock.now())
                            }
                        });
                    }
                    TtsSynthesisEffect::TtsAlignment { segment_id, tokens } => {
                        transcript.push_alignment(segment_id, tokens);
                    }
                    TtsSynthesisEffect::TtsAudio(mut audio) => {
                        audio.timed_transcript =
                            transcript.take_audio_transcript(&audio.segment_id, audio.is_final);
                        let blocked_at = clock.now();
                        let total_timer = async {
                            match core.total_deadline() {
                                Some(at) => clock.wait_until(at).await,
                                None => std::future::pending::<()>().await,
                            }
                        };
                        let delivered = match select(
                            pin!(stopped.wait_for(|stop| *stop)),
                            pin!(select(pin!(total_timer), pin!(audio_tx.send(Ok(audio))))),
                        )
                        .await
                        {
                            Either::Left(_) => false,
                            Either::Right((Either::Left(_), _)) => {
                                return Err(ApiError::timed_out(TimeoutPhase::Total));
                            }
                            Either::Right((Either::Right((result, _)), _)) => result.is_ok(),
                        };
                        if !delivered {
                            return Ok(false);
                        }
                        core.delivery_resumed(blocked_at, clock.now());
                    }
                    TtsSynthesisEffect::Finished(Err(error)) => return Err(error),
                    TtsSynthesisEffect::Finished(Ok(())) => return Ok(true),
                    TtsSynthesisEffect::TtsCancel => return Ok(false),
                }
            }
            let deadline = core.deadline();
            let timer = async {
                match deadline {
                    Some(at) => clock.wait_until(at).await,
                    None => std::future::pending::<()>().await,
                }
            };
            let command = async {
                if input_closed {
                    return std::future::pending::<TtsSynthesisEvent>().await;
                }
                match text_rx.recv().await {
                    Some(TextCmd::Push(text)) => TtsSynthesisEvent::Text(text),
                    Some(TextCmd::Failed) => TtsSynthesisEvent::Failed(
                        ApiError::new("TTS input queue overflowed at EOF").with_retryable(false),
                    ),
                    Some(TextCmd::End) | None => TtsSynthesisEvent::InputEnded,
                }
            };
            let event = match select(
                pin!(select(
                    pin!(select(pin!(stopped.wait_for(|stop| *stop)), pin!(audio_tx.closed()))),
                    pin!(timer),
                )),
                pin!(select(pin!(command), pin!(receiver.recv()))),
            )
            .await
            {
                Either::Left((Either::Left(_), _)) => TtsSynthesisEvent::Cancel,
                Either::Left((Either::Right(_), _)) => TtsSynthesisEvent::Wake,
                Either::Right((Either::Left((command, _)), _)) => command,
                Either::Right((Either::Right((received, _)), _)) => match received {
                    Err(error) => TtsSynthesisEvent::Failed(connect_error(WHAT, error)),
                    Ok(WsMessage::Closed) => TtsSynthesisEvent::ServerEof,
                    Ok(WsMessage::Binary(_) | WsMessage::Other) => TtsSynthesisEvent::Wake,
                    Ok(WsMessage::Text(raw)) => match wire::decode(&raw, provider) {
                        Some(wire::Incoming::OutputAudio { session_id, audio }) => {
                            match base64::engine::general_purpose::STANDARD.decode(audio) {
                                Ok(data) => TtsSynthesisEvent::Audio { session_id, data },
                                Err(_) => TtsSynthesisEvent::Failed(ApiError::new(
                                    "TTS returned malformed base64 audio",
                                )),
                            }
                        }
                        Some(incoming) => TtsSynthesisEvent::Received(incoming),
                        None => TtsSynthesisEvent::Failed(ApiError::new(
                            "TTS returned malformed protocol data",
                        )),
                    },
                },
            };
            if matches!(event, TtsSynthesisEvent::InputEnded) {
                input_closed = true;
            }
            effects.extend(core.handle(event, clock.now()));
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
