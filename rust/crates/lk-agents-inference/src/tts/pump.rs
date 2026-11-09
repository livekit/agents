use super::wire::{ErrorCode, Incoming};
use super::*;
use base64::Engine as _;
use futures_util::future::{Either, select};
use lk_agents_abi::errors::Timeout;
use lk_agents_abi::metrics::base::ConnectionAcquired;
use lk_agents_abi::{Closed, Ts};
use lk_agents_net::owned::Owned;
use lk_agents_net::transport::{TransportError, WsMessage, connect_error};
use std::collections::BTreeSet;
use std::pin::pin;

/// How a synthesis on a connection ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SynthesisEnd {
    /// The gateway finished it: the connection can serve the next synthesis.
    Completed,
    /// The caller stopped it or stopped listening.
    Stopped,
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
    /// The session flush that ends the input: the provider may complete after it.
    FinalFlush,
}

/// What the synthesis waits on next.
enum Event {
    Text(String),
    InputEnded,
    Received(Incoming),
    Audio {
        session_id: Option<String>,
        data: Vec<u8>,
    },
    Failed(ApiError),
    ServerEof,
    /// The caller stopped the synthesis or stopped listening.
    Cancel,
    /// A deadline is due, or a frame that does not concern the synthesis arrived.
    Wake,
}

/// A pooled connection checked out of the pool: dropped unreleased (a stop while opening, a
/// cancelled pump), it is forgotten by the pool and closed.
type Checkout = Owned<(u64, Connection)>;

/// One synthesis's progress on its connection: the segments it delivered and its deadlines.
struct Synthesis {
    request_id: String,
    segment_id: String,
    sample_rate: u32,
    read_timeout: Option<Duration>,
    response_deadline: Option<Ts>,
    total_deadline: Option<Ts>,
    started_time: Option<f64>,
    input_ended: bool,
    final_flush_sent: bool,
    segments: BTreeSet<String>,
    /// The synthesis's connection, until its first frame carries it.
    connection: Option<ConnectionAcquired>,
}

impl Synthesis {
    fn new(
        request_id: String,
        sample_rate: u32,
        read_timeout: Option<Duration>,
        total_deadline: Option<Ts>,
    ) -> Self {
        Self {
            segment_id: request_id.clone(),
            request_id,
            sample_rate,
            read_timeout,
            response_deadline: None,
            total_deadline,
            started_time: None,
            input_ended: false,
            final_flush_sent: false,
            segments: Default::default(),
            connection: None,
        }
    }

    fn deadline(&self) -> Option<Ts> {
        self.total_deadline.into_iter().chain(self.response_deadline).min()
    }

    /// Fails the synthesis once a deadline passed: the total one, else the response's.
    fn check(&self, now: Ts) -> Result<(), ApiError> {
        if !self.deadline().is_some_and(|deadline| now >= deadline) {
            return Ok(());
        }
        let phase = if self.total_deadline.is_some_and(|deadline| now >= deadline) {
            Timeout::Total
        } else {
            Timeout::Read
        };
        Err(self.fail(ApiError::timed_out(phase)))
    }

    /// Input went out: the response is expected within the read limit, restarted by the flush.
    fn input_sent(&mut self, final_flush: bool, started_time: f64, now: Ts) {
        self.started_time.get_or_insert(started_time);
        self.final_flush_sent |= final_flush;
        if final_flush || self.response_deadline.is_none() {
            self.response_deadline = self.read_timeout.map(|timeout| now.saturating_add(timeout));
        }
    }

    /// Time spent waiting on a slow caller does not count against the response.
    fn delivery_resumed(&mut self, blocked_at: Ts, now: Ts) {
        if let Some(deadline) = &mut self.response_deadline {
            *deadline = deadline.saturating_add(now.saturating_duration_since(blocked_at));
        }
    }

    /// Audio of the current segment; after the final flush each frame restarts the read limit.
    fn audio_received(
        &mut self,
        session_id: Option<String>,
        data: Vec<u8>,
        now: Ts,
    ) -> Result<SynthesizedAudio, ApiError> {
        self.segment(session_id)?;
        if !data.len().is_multiple_of(2) {
            return Err(self.fail(ApiError::new("TTS returned truncated PCM audio")));
        }
        self.response_deadline = if self.final_flush_sent {
            self.read_timeout.map(|timeout| now.saturating_add(timeout))
        } else {
            None
        };
        Ok(self.audio(data, false))
    }

    /// The segment an alignment belongs to; it counts as progress.
    fn aligned(&mut self, session_id: Option<String>, now: Ts) -> Result<String, ApiError> {
        self.segment(session_id)?;
        if self.response_deadline.is_some() {
            self.response_deadline = self.read_timeout.map(|timeout| now.saturating_add(timeout));
        }
        self.segments.insert(self.segment_id.clone());
        Ok(self.segment_id.clone())
    }

    /// The provider completed: a final frame for every segment, the current one last.
    fn done(&mut self, session_id: Option<String>) -> Result<Vec<SynthesizedAudio>, ApiError> {
        if !self.final_flush_sent {
            return Err(self.fail(ApiError::new("TTS completed before final flush")));
        }
        self.segment(session_id)?;
        let segment_id = self.segment_id.clone();
        let pending: Vec<_> =
            self.segments.iter().filter(|id| **id != segment_id).cloned().collect();
        let mut finals = Vec::new();
        for id in pending {
            self.segment_id = id;
            finals.push(self.audio(Vec::new(), true));
        }
        self.segment_id = segment_id;
        finals.push(self.audio(Vec::new(), true));
        Ok(finals)
    }

    fn segment(&mut self, id: Option<String>) -> Result<(), ApiError> {
        if let Some(id) = id {
            if id.is_empty() {
                return Err(self.fail(ApiError::new("TTS returned an empty segment identity")));
            }
            self.segment_id = id;
        }
        Ok(())
    }

    fn audio(&mut self, data: Vec<u8>, is_final: bool) -> SynthesizedAudio {
        self.segments.insert(self.segment_id.clone());
        SynthesizedAudio {
            is_final,
            segment_id: self.segment_id.clone(),
            started_time: self.started_time,
            connection: self.connection.take(),
            ..SynthesizedAudio::new(
                crate::vad::machine::mono(data, self.sample_rate),
                self.request_id.clone(),
            )
        }
    }

    /// `error`, naming this synthesis's request.
    fn fail(&self, mut error: ApiError) -> ApiError {
        error.request_id.get_or_insert_with(|| self.request_id.clone());
        error
    }
}

/// A gateway error frame as the error it reports.
fn gateway_error(code: ErrorCode, retryable: Option<bool>) -> ApiError {
    let message = format!("LiveKit Inference TTS returned {}", code.as_str());
    let status = match code {
        ErrorCode::AuthenticationError => Some(401),
        ErrorCode::RegionRestricted => Some(403),
        ErrorCode::NotFound => Some(404),
        ErrorCode::RateLimited => Some(429),
        ErrorCode::ServerError => Some(500),
        ErrorCode::UnsupportedMessageType
        | ErrorCode::MissingMessageType
        | ErrorCode::SessionNotCreated
        | ErrorCode::SessionAlreadyCreated
        | ErrorCode::InvalidData
        | ErrorCode::BadRequest => Some(400),
        _ => None,
    };
    match status {
        Some(status) => ApiError::status(message, status, None, retryable),
        None => {
            let mut error = ApiError::new(message);
            error.retryable = retryable.unwrap_or(matches!(code, ErrorCode::InactivityTimeout));
            error
        }
    }
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
        let mut synthesis = Synthesis::new(
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
        let total_deadline = synthesis.total_deadline;
        let opening = async {
            synthesis.check(self.clock.now())?;
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
                let checkout = self.checkout(id, Connection { sender, receiver, configuration });
                let acquire_time = now.saturating_duration_since(acquiring).as_secs_f64();
                return Ok((checkout, ConnectionAcquired { acquire_time, reused: false }));
            };
            Ok::<_, ApiError>((reused, ConnectionAcquired { acquire_time: 0.0, reused: true }))
        };
        let total_timer = async {
            match total_deadline {
                Some(at) => self.clock.wait_until(at).await,
                None => std::future::pending::<()>().await,
            }
        };
        let opened = match select(
            pin!(select(pin!(self.stopped.cancelled()), pin!(self.audio_tx.closed()))),
            pin!(select(pin!(total_timer), pin!(opening))),
        )
        .await
        {
            Either::Left(_) => return Ok(()),
            Either::Right((Either::Left(_), _)) => {
                return Err(ApiError::timed_out(Timeout::Total));
            }
            Either::Right((Either::Right((opened, _)), _)) => opened,
        };
        let (mut checkout, acquired) = opened?;
        synthesis.connection = Some(acquired);
        let (_, connection) = &mut *checkout;
        let outcome = self
            .drive(&mut synthesis, connection.receiver.as_mut(), connection.sender.as_ref())
            .await;
        if let Err(error) = &outcome {
            self.publish_failure(error);
        }
        self.release_connection(checkout, &outcome).await;
        outcome.map(drop)
    }

    /// Holds a connection checked out of the pool until it is given back or closed.
    fn checkout(&self, id: u64, connection: Connection) -> Checkout {
        let (connections, clock) = (self.connections.clone(), self.clock.clone());
        Owned::new(
            &self.background,
            (id, connection),
            Box::new(move |(id, connection): (u64, Connection), deadline| {
                Box::pin(async move {
                    connections.lock().forget(id);
                    let close = connection.sender.close();
                    match lk_agents_abi::before(clock.as_ref(), Some(deadline), close).await {
                        Ok(()) => Closed::Clean,
                        Err(_) => Closed::TimedOut,
                    }
                })
            }),
        )
    }

    /// Takes a pooled connection opened for this configuration; the expired ones, and one opened
    /// for another configuration, are closed.
    async fn reuse_connection(&self, configuration: &str) -> Option<Checkout> {
        let (mut reused, mut expired) =
            self.connections.lock().take(self.clock.now().as_secs_f64());
        if let Some((id, connection)) =
            reused.take_if(|(_, connection)| connection.configuration != configuration)
        {
            self.connections.lock().forget(id);
            expired.push(connection);
        }
        let reused = reused.map(|(id, connection)| self.checkout(id, connection));
        for connection in expired {
            connection.sender.close().await;
        }
        reused
    }

    /// A completed synthesis gives the connection back while the stream is live; otherwise it
    /// is closed.
    async fn release_connection(
        &self,
        checkout: Checkout,
        outcome: &Result<SynthesisEnd, ApiError>,
    ) {
        if matches!(outcome, Ok(SynthesisEnd::Completed))
            && !self.stopped.is_cancelled()
            && !self.audio_tx.is_closed()
        {
            let (id, connection) = checkout.into_inner();
            self.connections.lock().give_back(id, connection);
            return;
        }
        let _ = checkout.close().await;
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
        synthesis: &mut Synthesis,
        receiver: &mut dyn WsReceiver,
        sender: &dyn WsSender,
    ) -> Result<SynthesisEnd, ApiError> {
        let mut input = TextInput::Open;
        loop {
            let event = self.next_event(synthesis, receiver, input).await;
            if matches!(event, Event::InputEnded) {
                input = TextInput::Ended;
            }
            if let Some(end) = self.step(synthesis, sender, event).await? {
                return Ok(end);
            }
        }
    }

    /// Carries out one event; `Some` once the synthesis ended.
    async fn step(
        &mut self,
        synthesis: &mut Synthesis,
        sender: &dyn WsSender,
        event: Event,
    ) -> Result<Option<SynthesisEnd>, ApiError> {
        let now = self.clock.now();
        if !matches!(event, Event::Cancel) {
            synthesis.check(now)?;
        }
        match event {
            Event::Text(_) if synthesis.input_ended => {
                Err(synthesis.fail(ApiError::new("TTS text received after input EOF")))
            }
            Event::Text(text) => {
                self.send_input(synthesis, sender, Outbound::Transcript(text)).await?;
                Ok(None)
            }
            Event::InputEnded if synthesis.input_ended => Ok(None),
            Event::InputEnded => {
                synthesis.input_ended = true;
                self.send_input(synthesis, sender, Outbound::FinalFlush).await?;
                Ok(None)
            }
            Event::Audio { session_id, data } => {
                let audio = synthesis.audio_received(session_id, data, now)?;
                self.deliver_audio(synthesis, audio).await
            }
            Event::Received(Incoming::OutputAlignment { session_id, tokens }) => {
                let segment_id = synthesis.aligned(session_id, now)?;
                self.transcript.push_alignment(segment_id, tokens);
                Ok(None)
            }
            Event::Received(Incoming::Done { session_id }) => {
                for audio in synthesis.done(session_id)? {
                    if let Some(end) = self.deliver_audio(synthesis, audio).await? {
                        return Ok(Some(end));
                    }
                }
                Ok(Some(SynthesisEnd::Completed))
            }
            Event::Received(Incoming::SessionCreated { session_id }) => {
                synthesis.segment(session_id)?;
                Ok(None)
            }
            Event::Received(Incoming::Error { code, retryable }) => {
                Err(synthesis.fail(gateway_error(code, retryable)))
            }
            Event::Received(Incoming::FallbackActivated {
                session_id,
                fallback_type,
                provider,
                model,
                voice: _,
                cause,
            }) => {
                synthesis.segment(session_id)?;
                log::info!(target: lk_agents_abi::LOG_TARGET,
                    fallback_type = fallback_type.as_str(), provider = provider.as_str(),
                    model = model.as_str(), cause = cause.as_str();
                    "TTS gateway activated fallback");
                Ok(None)
            }
            Event::Received(Incoming::Malformed { detail }) => {
                Err(synthesis.fail(ApiError::new(detail)))
            }
            Event::Failed(error) => Err(synthesis.fail(error)),
            Event::ServerEof => {
                Err(synthesis.fail(ApiError::new("TTS gateway closed before synthesis completion")))
            }
            Event::Cancel => Ok(Some(SynthesisEnd::Stopped)),
            // Session-created and heartbeat frames do not satisfy outstanding synthesis work.
            Event::Received(_) | Event::Wake => Ok(None),
        }
    }

    /// Sends text or the final flush; a failure, or a deadline passed meanwhile, fails the
    /// synthesis.
    async fn send_input(
        &mut self,
        synthesis: &mut Synthesis,
        sender: &dyn WsSender,
        input: Outbound,
    ) -> Result<(), ApiError> {
        let (text, final_flush) = match input {
            Outbound::Transcript(text) => match wire::input_transcript(&text, &self.opts) {
                Ok(frame) => (frame.to_string(), false),
                Err(error) => {
                    synthesis.check(self.clock.now())?;
                    return Err(synthesis.fail(error));
                }
            },
            Outbound::FinalFlush => (wire::session_flush().to_string(), true),
        };
        let clock = self.clock.as_ref();
        let total_deadline = synthesis.total_deadline;
        let result =
            Self::send(sender, text, &self.opts, clock, total_deadline, &self.stopped).await;
        synthesis.check(clock.now())?;
        match result {
            Ok(()) => {
                let started_time = lk_agents_abi::unix_secs(clock.wall());
                synthesis.input_sent(final_flush, started_time, clock.now());
                Ok(())
            }
            Err(error) => Err(synthesis.fail(error)),
        }
    }

    /// Hands audio to the caller, waiting while the caller is behind, up to the total deadline.
    async fn deliver_audio(
        &mut self,
        synthesis: &mut Synthesis,
        mut audio: SynthesizedAudio,
    ) -> Result<Option<SynthesisEnd>, ApiError> {
        audio.timed_transcript =
            self.transcript.take_audio_transcript(&audio.segment_id, audio.is_final);
        let clock = self.clock.as_ref();
        let blocked_at = clock.now();
        let total_timer = async {
            match synthesis.total_deadline {
                Some(at) => clock.wait_until(at).await,
                None => std::future::pending::<()>().await,
            }
        };
        let sent = match select(
            pin!(self.stopped.cancelled()),
            pin!(select(pin!(total_timer), pin!(self.audio_tx.send(Ok(audio))))),
        )
        .await
        {
            Either::Left(_) => return Ok(Some(SynthesisEnd::Stopped)),
            Either::Right((Either::Left(_), _)) => {
                return Err(ApiError::timed_out(Timeout::Total));
            }
            Either::Right((Either::Right((sent, _)), _)) => sent,
        };
        if sent.is_err() {
            return Ok(Some(SynthesisEnd::Stopped));
        }
        synthesis.delivery_resumed(blocked_at, clock.now());
        Ok(None)
    }

    /// The next event: a stop or a closed output, a deadline, then the caller's
    /// text or the gateway's message.
    async fn next_event(
        &mut self,
        synthesis: &Synthesis,
        receiver: &mut dyn WsReceiver,
        input: TextInput,
    ) -> Event {
        let clock = self.clock.as_ref();
        let deadline = synthesis.deadline();
        let timer = async {
            match deadline {
                Some(at) => clock.wait_until(at).await,
                None => std::future::pending::<()>().await,
            }
        };
        let text_rx = &mut self.text_rx;
        let command = async {
            if input == TextInput::Ended {
                return std::future::pending::<Event>().await;
            }
            command_event(text_rx.recv().await)
        };
        match select(
            pin!(select(
                pin!(select(pin!(self.stopped.cancelled()), pin!(self.audio_tx.closed()))),
                pin!(timer),
            )),
            pin!(select(pin!(command), pin!(receiver.recv()))),
        )
        .await
        {
            Either::Left((Either::Left(_), _)) => Event::Cancel,
            Either::Left((Either::Right(_), _)) => Event::Wake,
            Either::Right((Either::Left((command, _)), _)) => command,
            Either::Right((Either::Right((received, _)), _)) => {
                received_event(received, self.provider)
            }
        }
    }

    async fn send(
        sender: &dyn WsSender,
        text: String,
        opts: &TtsOptions,
        clock: &dyn Clock,
        deadline: Option<Ts>,
        stopped: &CancellationToken,
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
            pin!(stopped.cancelled()),
            pin!(select(pin!(timer), pin!(sender.send_text(text)))),
        )
        .await
        {
            Either::Left(_) => Err(ApiError::new("TTS synthesis cancelled")),
            Either::Right((Either::Left(_), _)) => {
                Err(ApiError::timed_out(if total_deadline.is_some_and(|at| clock.now() >= at) {
                    Timeout::Total
                } else {
                    Timeout::Write
                }))
            }
            Either::Right((Either::Right((result, _)), _)) => {
                result.map_err(|error| connect_error(WHAT, error))
            }
        }
    }
}

/// The caller's text command as an event.
fn command_event(command: Option<TextCmd>) -> Event {
    match command {
        Some(TextCmd::Push(text)) => Event::Text(text),
        Some(TextCmd::Failed) => {
            Event::Failed(ApiError::new("TTS input queue overflowed at EOF").with_retryable(false))
        }
        Some(TextCmd::End) | None => Event::InputEnded,
    }
}

/// A gateway message as an event, with its audio decoded.
fn received_event(
    received: Result<WsMessage, TransportError>,
    provider: Option<InferenceProvider>,
) -> Event {
    let raw = match received {
        Err(error) => return Event::Failed(connect_error(WHAT, error)),
        Ok(WsMessage::Closed) => return Event::ServerEof,
        Ok(WsMessage::Binary(_) | WsMessage::Other) => return Event::Wake,
        Ok(WsMessage::Text(raw)) => raw,
    };
    match wire::decode(&raw, provider) {
        Some(wire::Incoming::OutputAudio { session_id, audio }) => {
            match base64::engine::general_purpose::STANDARD.decode(audio) {
                Ok(data) => Event::Audio { session_id, data },
                Err(_) => Event::Failed(ApiError::new("TTS returned malformed base64 audio")),
            }
        }
        Some(incoming) => Event::Received(incoming),
        None => Event::Failed(ApiError::new("TTS returned malformed protocol data")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(ms: u64) -> Ts {
        Ts::from_nanos(ms * 1_000_000)
    }

    fn synthesis() -> Synthesis {
        Synthesis::new("request".into(), 24_000, Some(Duration::from_secs(1)), None)
    }

    // Catches: an empty synthesis completing without the final flush, or without one final frame.
    #[test]
    fn empty_input_requires_the_final_flush_and_the_provider_completion() {
        let mut s = synthesis();
        assert!(s.done(None).is_err(), "no completion before the final flush");
        s.input_sent(true, 0.0, Ts::ZERO);
        let finals = s.done(None).unwrap();
        assert!(matches!(finals.as_slice(), [audio] if audio.is_final));
    }

    // Catches: a heartbeat treated as the response, or a later segment's input not re-arming the
    // read limit once audio cleared it.
    #[test]
    fn heartbeats_cannot_settle_response_and_later_segments_rearm_the_deadline() {
        let mut s = synthesis();
        s.input_sent(false, 0.0, Ts::ZERO);
        assert!(s.check(at(900)).is_ok());
        assert_eq!(s.check(at(1_000)).unwrap_err().timeout, Some(Timeout::Read));

        let mut s = synthesis();
        s.input_sent(false, 0.0, Ts::ZERO);
        s.audio_received(Some("first".into()), vec![0; 2], at(500)).unwrap();
        s.input_sent(false, 0.0, at(1_000));
        assert!(s.check(at(1_500)).is_ok());
        assert!(s.check(at(2_000)).is_err());
    }

    // Catches: delivered audio losing its segment or request identity, or a gateway error frame
    // read as retryable when it says otherwise.
    #[test]
    fn delivered_audio_keeps_its_identity_and_provider_failure_is_terminal() {
        let mut s = synthesis();
        for segment in ["first", "second"] {
            let audio = s.audio_received(Some(segment.into()), vec![0; 2], Ts::ZERO).unwrap();
            assert_eq!(
                (audio.segment_id.as_str(), audio.request_id.as_str()),
                (segment, "request")
            );
        }
        assert!(s.audio_received(None, vec![0; 3], Ts::ZERO).is_err(), "truncated PCM");
        let error = gateway_error(ErrorCode::ServerError, Some(true));
        assert_eq!((error.status_code, error.retryable), (Some(500), true));
    }

    // Catches: segment progress pushing the total deadline out.
    #[test]
    fn segment_progress_cannot_extend_the_total_deadline() {
        let mut s =
            Synthesis::new("request".into(), 24_000, Some(Duration::from_secs(1)), Some(at(1_500)));
        s.input_sent(false, 0.0, Ts::ZERO);
        s.audio_received(None, vec![0; 2], at(500)).unwrap();
        s.input_sent(false, 1.0, at(1_000));
        let error = s.check(at(1_500)).unwrap_err();
        assert_eq!(error.timeout, Some(Timeout::Total));
        assert_eq!(error.request_id.as_deref(), Some("request"));
    }
}
