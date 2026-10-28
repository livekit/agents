//! Gateway I/O executor for the synchronous synthesis protocol.

use super::machine::{Effect, Event, Synthesis};
use super::*;
use crate::attempts::{AttemptStep, Attempts, RequestPolicy};
use crate::transport::{WsMessage, connect_error};
use base64::Engine as _;
use futures_util::future::{Either, select};
use lk_agents_abi::Ts;
use lk_agents_abi::errors::TimeoutPhase;
use std::pin::pin;

impl Pump {
    pub(super) async fn run(mut self) {
        let done = self.done.take().expect("pump completion owned once");
        let outcome = self.run_attempts().await;
        if let Err(error) = &outcome {
            self.publish_failure(error);
        }
        let _ = done.send(outcome);
    }

    async fn run_attempts(&mut self) -> Result<(), ApiError> {
        let mut policy = RequestPolicy::from_options(&self.opts.conn_options);
        // The protocol and socket expose phase progress inside an attempt. This machine alone
        // chooses reopen/replay and keeps the original acceptance deadline across its waits.
        policy.connect_timeout = None;
        policy.write_timeout = None;
        policy.read_timeout = None;
        let mut steps = Vec::new();
        let mut attempts = Attempts::start(policy, self.accepted_at, &mut steps);
        let request_id =
            create_inference_request_id(self.session_id.as_deref(), RequestKind::Tts, "");
        let mut replay = Replay::default();
        loop {
            if *self.stopped.borrow() || self.audio_tx.is_closed() {
                attempts.cancel(self.clock.now(), &mut Vec::new());
                return Ok(());
            }
            attempts.wake(self.clock.now(), None, &mut steps);
            if let Some(error) = steps.iter().find_map(|step| match step {
                AttemptStep::GiveUp(error) => Some(error.clone()),
                _ => None,
            }) {
                return Err(error);
            }
            steps.clear();
            let Some(id) = attempts.current() else {
                let at = attempts.wake_at(None).expect("an undecided retry has a wake");
                if matches!(
                    select(
                        pin!(select(
                            pin!(self.stopped.wait_for(|stop| *stop)),
                            pin!(self.audio_tx.closed())
                        )),
                        pin!(self.clock.wait_until(at)),
                    )
                    .await,
                    Either::Left(_)
                ) {
                    attempts.cancel(self.clock.now(), &mut Vec::new());
                    return Ok(());
                }
                continue;
            };
            match self.attempt(&request_id, &mut replay, &mut attempts).await {
                Ok(completed) => {
                    if completed {
                        attempts.succeeded(id, self.clock.now());
                    } else {
                        attempts.cancel(self.clock.now(), &mut Vec::new());
                    }
                    attempts.closed(id, self.clock.now(), &mut Vec::new());
                    return Ok(());
                }
                Err(error) => {
                    steps.clear();
                    attempts.failed_error(id, error, self.clock.now(), &mut steps);
                    attempts.closed(id, self.clock.now(), &mut steps);
                }
            }
        }
    }

    async fn attempt(
        &mut self,
        request_id: &str,
        replay: &mut Replay,
        attempts: &mut Attempts,
    ) -> Result<bool, ApiError> {
        let mut core = Synthesis::new(
            request_id.to_owned(),
            self.sample_rate,
            self.expressive,
            self.opts.conn_options.read_timeout.map(Duration::from_secs_f64),
            self.opts
                .conn_options
                .total_timeout
                .map(|timeout| self.accepted_at.saturating_add(Duration::from_secs_f64(timeout))),
        );
        let configuration = connection_configuration(&self.create_frame, &self.headers);
        let (mut reused, mut expired) =
            self.connections.lock().unwrap().take(pool_secs(&*self.clock));
        if reused.as_ref().is_some_and(|(_, connection)| {
            connection.configuration != configuration || connection.session_id != self.session_id
        }) {
            let (id, connection) = reused.take().unwrap();
            self.connections.lock().unwrap().forget(id);
            expired.push(connection);
        }
        for connection in expired {
            connection.sender.close().await;
        }
        let total_deadline = core.total_deadline();
        let opening = async {
            match core.handle(Event::Start, self.clock.now()).into_iter().next() {
                Some(Effect::TtsSynthesisStart) => {}
                Some(Effect::TtsFailed(error)) => return Err(error),
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
                    let id = self.connections.lock().unwrap().opened(pool_secs(&*self.clock));
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
            pin!(select(pin!(self.stopped.wait_for(|stop| *stop)), pin!(self.audio_tx.closed()))),
            pin!(select(pin!(total_timer), pin!(opening))),
        )
        .await
        {
            Either::Left(_) => return Ok(false),
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
            .drive(
                &mut core,
                connection.receiver.as_mut(),
                connection.sender.as_ref(),
                replay,
                attempts,
            )
            .await;
        if matches!(outcome, Ok(true)) && !*self.stopped.borrow() && !self.audio_tx.is_closed() {
            self.connections.lock().unwrap().give_back(id, connection);
        } else {
            self.connections.lock().unwrap().forget(id);
            connection.sender.close().await;
        }
        outcome
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
        core: &mut Synthesis,
        receiver: &mut dyn WsReceiver,
        sender: &dyn WsSender,
        replay: &mut Replay,
        attempts: &mut Attempts,
    ) -> Result<bool, ApiError> {
        let opts = &self.opts;
        let provider = &self.provider;
        let clock = self.clock.as_ref();
        let text_rx = &mut self.text_rx;
        let audio_tx = &self.audio_tx;
        let stopped = &mut self.stopped;
        let mut effects = std::collections::VecDeque::new();
        let mut input_closed = false;
        let replay_count = replay.commands.len();
        let mut replayed = 0;
        loop {
            while let Some(effect) = effects.pop_front() {
                match effect {
                    Effect::TtsSynthesisStart => {}
                    Effect::TtsTextSend(text) => {
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
                                Event::InputSent {
                                    final_flush: false,
                                    started_time: lk_agents_abi::unix_secs(clock.wall()),
                                },
                                clock.now(),
                            ),
                            Err(error) => core.handle(Event::Failed(error), clock.now()),
                        });
                    }
                    Effect::TtsFinalFlush => {
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
                                Event::InputSent {
                                    final_flush: true,
                                    started_time: lk_agents_abi::unix_secs(clock.wall()),
                                },
                                clock.now(),
                            ),
                            Err(error) => core.handle(Event::Failed(error), clock.now()),
                        });
                    }
                    Effect::TtsAudio(audio) => {
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
                        attempts.delivered();
                        replay.delivered = true;
                        replay.commands.clear();
                        replay.bytes = 0;
                        core.delivery_resumed(blocked_at, clock.now());
                    }
                    Effect::TtsFailed(error) => return Err(error),
                    Effect::TtsCompleted => return Ok(true),
                    Effect::TtsCancel => return Ok(false),
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
                    return std::future::pending::<Event>().await;
                }
                let command = if replayed < replay_count {
                    let command = replay.commands[replayed].clone();
                    replayed += 1;
                    command
                } else {
                    let command = text_rx.recv().await.unwrap_or(TextCmd::End);
                    if let Err(error) = replay.accepted(&command) {
                        return Event::Failed(error);
                    }
                    command
                };
                match command {
                    TextCmd::Push(text) => Event::Text(text),
                    TextCmd::Failed => Event::Failed(
                        ApiError::new("TTS input queue overflowed at EOF").with_retryable(false),
                    ),
                    TextCmd::End => Event::EndInput,
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
                Either::Left((Either::Left(_), _)) => Event::Cancel,
                Either::Left((Either::Right(_), _)) => Event::Tick,
                Either::Right((Either::Left((command, _)), _)) => command,
                Either::Right((Either::Right((received, _)), _)) => match received {
                    Err(error) => Event::Failed(connect_error(WHAT, error)),
                    Ok(WsMessage::Closed) => Event::ServerEof,
                    Ok(WsMessage::Other) => Event::Tick,
                    Ok(WsMessage::Text(raw)) => match wire::decode(&raw, provider) {
                        Some(wire::Incoming::OutputAudio { session_id, audio }) => {
                            match base64::engine::general_purpose::STANDARD.decode(audio) {
                                Ok(data) => Event::Audio { session_id, data },
                                Err(_) => Event::Failed(ApiError::new(
                                    "TTS returned malformed base64 audio",
                                )),
                            }
                        }
                        Some(incoming) => Event::Received(incoming),
                        None => {
                            Event::Failed(ApiError::new("TTS returned malformed protocol data"))
                        }
                    },
                },
            };
            if matches!(event, Event::EndInput) {
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

#[derive(Default)]
struct Replay {
    commands: Vec<TextCmd>,
    bytes: usize,
    delivered: bool,
}

impl Replay {
    fn accepted(&mut self, command: &TextCmd) -> Result<(), ApiError> {
        if self.delivered {
            return Ok(());
        }
        let bytes = match command {
            TextCmd::Push(text) => text.len(),
            _ => 0,
        };
        // Empty pushes carry no synthesis input; retaining them would evade the byte bound.
        if matches!(command, TextCmd::Push(text) if text.is_empty()) {
            return Ok(());
        }
        if self.bytes.saturating_add(bytes) > MAX_PENDING_TEXT_BYTES {
            return Err(ApiError::new("TTS replay input capacity exceeded").with_retryable(false));
        }
        self.bytes += bytes;
        self.commands.push(command.clone());
        Ok(())
    }
}
