//! RTC execution for the synchronous room state.

use super::input::{ParticipantAudioInput, ParticipantVideoInput};
use super::output::{
    ParticipantAudioOutput,
    transcription::{ParticipantTranscriptionOutput, SegmentIds},
};
use super::pre_connect_audio::PreConnectAudioHandler;
use crate::monotone::Monotone;
use crate::telemetry;
use futures_util::FutureExt as _;
use lk_agents_abi::voice::io::IoError;
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, Result, TextOutput};
use lk_agents_abi::voice::room_io::{
    ParticipantInfo, ParticipantLinked, ParticipantUnlinked, RoomIoConfig, TextInput,
};
use lk_agents_abi::voice::room_owner::{self, RoomIo as _, RoomIoEvent};
use lk_agents_abi::voice::rtc::capabilities::RtcRoom;
use lk_agents_abi::voice::rtc::events::RoomEvent;
use lk_agents_abi::voice::rtc::{
    ConnectionState, DisconnectReason, ParticipantIdentity, ParticipantKind, ProcessorFailure,
    RemoteParticipant,
};
use lk_agents_abi::{
    ATTRIBUTE_AGENT_STATE, BoxFuture, Clock, Closed, HOST_TEARDOWN_TIMEOUT, Spawner, SpawnerExt,
    TOPIC_CHAT, Ts,
};
use lk_agents_net::task::TaskSet;
use lk_agents_voice::room_io::{RoomIo as RoomIoState, RoomView};
use lk_agents_voice::room_io::{accepted_kinds, outbound_sip, publishes_on_behalf_of};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use tokio::sync::{Notify, mpsc, watch};

const ROOM_EVENT_CAPACITY: usize = 64;

pub struct RoomIo {
    room: Arc<dyn RtcRoom>,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    config: RoomIoConfig,

    audio_input: Option<Arc<ParticipantAudioInput>>,
    /// Here so it shares this object's room events and participant link with the audio leg.
    video_input: Option<Arc<ParticipantVideoInput>>,
    audio_output: Option<Arc<ParticipantAudioOutput>>,
    /// Non-delta: an STT sends successive full replacements.
    user_tr: Option<Arc<ParticipantTranscriptionOutput>>,
    /// Delta: an LLM sends tokens to accumulate.
    agent_tr: Option<Arc<ParticipantTranscriptionOutput>>,
    pre_connect: Option<Arc<PreConnectAudioHandler>>,

    /// One state owns decisions and wakes readers after each transition.
    state: Mutex<RoomIoState>,
    view: watch::Sender<RoomView>,
    connected: Monotone<bool>,

    events_tx: mpsc::Sender<RoomIoEvent>,
    /// The session's room events, and apart from them the input processors' failing spans: a
    /// failing span waits for the session rather than drop, or take the slots other events need.
    events_rx: Mutex<Option<(mpsc::Receiver<RoomIoEvent>, mpsc::Receiver<ProcessorFailure>)>>,

    tasks: Arc<TaskSet>,
    /// The close, which waits `tasks` out; it holds this object until it ends.
    closing: std::sync::OnceLock<lk_agents_abi::OwnedTask>,
    execution: Mutex<Execution>,
    stopped: Notify,
    started: Notify,
    agent_state: watch::Sender<Option<String>>,
    chat_registered: Mutex<bool>,
    /// The session's startup spans, under its `session_start`; ended with the first link.
    startup: Mutex<Option<Startup>>,
}

struct Startup {
    cx: telemetry::Cx,
    participant: Option<telemetry::ScopedSpan<telemetry::WaitForParticipant>>,
}

struct Execution {
    closing: bool,
    starting: usize,
    cleanup: Closed,
}

impl RoomIo {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        ids: Arc<SegmentIds>,
        config: RoomIoConfig,
        text_next_in_chain: Option<Arc<dyn TextOutput>>,
        pool: Option<lk_agents_media::pool::MediaPool>,
    ) -> Arc<Self> {
        let pre_connect = config.audio_input.as_ref().filter(|a| a.pre_connect_audio).map(|a| {
            PreConnectAudioHandler::new(
                room.clone(),
                spawner.clone(),
                clock.clone(),
                a.pre_connect_audio_timeout,
                (a.sample_rate, a.num_channels),
                pool.clone(),
            )
        });

        let (events_tx, events_rx) = mpsc::channel(ROOM_EVENT_CAPACITY);
        let (failures_tx, failures_rx) = mpsc::channel(ROOM_EVENT_CAPACITY);
        let audio_input = config.audio_input.as_ref().map(|a| {
            ParticipantAudioInput::new(
                room.clone(),
                spawner.clone(),
                clock.clone(),
                a.clone(),
                pre_connect.clone(),
                failures_tx.clone(),
            )
        });

        // Not `start()`ed: driven by `RoomIo::handle_event`.
        let video_input = config
            .video_input
            .then(|| ParticipantVideoInput::new(room.clone(), spawner.clone(), clock.clone()));

        let audio_output = config.audio_output.as_ref().map(|a| {
            ParticipantAudioOutput::new(room.clone(), clock.clone(), spawner.clone(), a.clone())
        });

        let (user_tr, agent_tr) = match &config.text_output {
            Some(t) => (
                Some(ParticipantTranscriptionOutput::new(
                    room.clone(),
                    spawner.clone(),
                    clock.clone(),
                    ids.clone(),
                    false,
                    false,
                    None,
                )),
                Some(ParticipantTranscriptionOutput::new(
                    room.clone(),
                    spawner.clone(),
                    clock.clone(),
                    ids,
                    true,
                    t.json_format,
                    text_next_in_chain,
                )),
            ),
            None => (None, None),
        };

        let state = RoomIoState::new(config.participant_identity.clone());
        Arc::new(RoomIo {
            view: watch::channel(state.state()).0,
            state: Mutex::new(state),
            room,
            clock,
            spawner,
            config,
            audio_input,
            video_input,
            audio_output,
            user_tr,
            agent_tr,
            pre_connect,
            connected: Monotone::new(false),
            events_tx,
            events_rx: Mutex::new(Some((events_rx, failures_rx))),
            tasks: Arc::default(),
            closing: std::sync::OnceLock::new(),
            execution: Mutex::new(Execution {
                closing: false,
                starting: 0,
                cleanup: Closed::Clean,
            }),
            stopped: Notify::new(),
            started: Notify::new(),
            agent_state: watch::channel(None).0,
            chat_registered: Mutex::new(false),
            startup: Mutex::new(None),
        })
    }

    /// The participant linked during startup: `wait_for_participant` ends with it, and the wait
    /// for its audio begins.
    fn linked_at_startup(&self, identity: &ParticipantIdentity, sid: &str, kind: ParticipantKind) {
        let (cx, span) = {
            let mut startup = self.startup.lock();
            let Some(Startup { cx, participant }) = startup.as_mut() else { return };
            let Some(span) = participant.take() else { return };
            (cx.clone(), span)
        };
        let info = ParticipantInfo { identity: identity.clone(), sid: sid.to_owned(), kind };
        telemetry::end(span, Some(telemetry::participant_attrs(&info)));
        if let Some(input) = &self.audio_input {
            let identity = telemetry::Content(identity.as_str().to_owned());
            input.wait_for_track(telemetry::start(
                &cx,
                telemetry::WaitForAudioTrack { participant_identity: identity },
            ));
        }
    }

    pub fn events(
        &self,
    ) -> Option<(mpsc::Receiver<RoomIoEvent>, mpsc::Receiver<ProcessorFailure>)> {
        self.events_rx.lock().take()
    }

    async fn read_events(
        self: Arc<Self>,
        mut events: lk_agents_abi::voice::rtc::Receiver<RoomEvent>,
    ) {
        let result = async {
            if self.room.connection_state()?.is_connected() {
                self.connected.set(true);
            }
            loop {
                let stopped = self.stopped.notified();
                tokio::pin!(stopped);
                stopped.as_mut().enable();
                if self.is_closing() {
                    return Ok(());
                }
                let event = tokio::select! {
                    biased;
                    () = &mut stopped => return Ok(()),
                    event = events.next() => event.map_err(IoError::from)?,
                };
                let Some(event) = event else {
                    return Ok(());
                };
                tokio::select! {
                    biased;
                    () = &mut stopped => return Ok(()),
                    result = self.handle_event(&event) => result?,
                }
                if matches!(
                    event,
                    RoomEvent::ConnectionStateChanged { state: ConnectionState::Connected }
                ) {
                    let last = self.agent_state.borrow().clone();
                    if let Some(state) = last {
                        self.clone().on_agent_state_changed(&state);
                    }
                }
            }
        }
        .await;
        let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        self.record_cleanup(events.close(deadline).await);
        if let Err(error) = result {
            self.fail(error);
        }
    }

    fn fail(self: &Arc<Self>, error: IoError) {
        self.transition(|state| state.finish_start(Err(error)));
        self.record_cleanup(Closed::Failed);
        self.request_close();
    }

    /// `_init_task`.
    async fn init(self: Arc<Self>) -> Result<()> {
        self.connected.wait_for(|set| *set).await;

        // Replay participants who joined before the handler was registered.
        for p in self.room.remote_participants()? {
            self.consider(p.identity(), p.sid(), p.kind(), &p.attributes()?).await?;
        }

        {
            let mut state = self.view.subscribe();
            let state =
                state.wait_for(|state| state.linked.is_some() || state.ready.is_some()).await;
            match state {
                Ok(state) if state.linked.is_some() => {}
                Ok(_) | Err(_) => return Ok(()),
            }
        }

        if let Some(agent_tr) = &self.agent_tr {
            // The agent transcribes *itself*, so its transcript hangs off its own track.
            agent_tr.set_participant(Some(self.room.local().identity()?)).await?;
        }
        let startup = self.startup.lock().take().map(|startup| startup.cx);
        let outcome = match &self.audio_output {
            Some(out) => {
                let span = startup.map(|cx| telemetry::start(&cx, telemetry::PublishAudioOutput));
                let outcome = out.start().await;
                match (span, &outcome) {
                    (Some(span), Ok(())) => {
                        let track_id = out.published().map(|sid| sid.as_str().to_owned());
                        telemetry::end(
                            span,
                            telemetry::PublishAudioOutputEnd {
                                track_id: track_id.unwrap_or_default(),
                            },
                        );
                    }
                    (Some(span), Err(error)) => {
                        telemetry::end(span, error);
                    }
                    (None, _) => {}
                }
                outcome
            }
            None => Ok(()),
        };
        self.transition(|state| state.finish_start(outcome));
        Ok(())
    }

    async fn attach_participant(&self, identity: Option<ParticipantIdentity>) -> Result<()> {
        if let Some(input) = &self.audio_input {
            input.set_participant(identity.clone())?;
        }
        if let Some(video) = &self.video_input {
            video.set_participant(identity.clone())?;
        }
        if let Some(user_tr) = &self.user_tr {
            user_tr.set_participant(identity).await?;
        }
        Ok(())
    }

    /// `_on_participant_connected`'s filter.
    async fn consider(
        &self,
        candidate: &ParticipantIdentity,
        sid: &str,
        kind: ParticipantKind,
        attrs: &BTreeMap<String, String>,
    ) -> Result<()> {
        let agent_identity = self.room.local().identity()?;
        if self.transition(|state| {
            state.consider(
                candidate,
                sid,
                kind,
                publishes_on_behalf_of(attrs),
                &agent_identity,
                accepted_kinds(&self.config),
            )
        }) {
            self.emit(RoomIoEvent::ParticipantInfo {
                participant: ParticipantInfo {
                    identity: candidate.clone(),
                    sid: sid.to_owned(),
                    kind,
                },
                local: false,
            })?;
            self.emit(RoomIoEvent::ParticipantLinked {
                participant: ParticipantLinked { identity: candidate.clone() },
                outbound_sip: outbound_sip(kind, attrs),
            })?;
            self.linked_at_startup(candidate, sid, kind);
            self.attach_participant(None).await?;
            self.attach_participant(Some(candidate.clone())).await?;
        }
        Ok(())
    }

    async fn handle_event(self: &Arc<Self>, event: &RoomEvent) -> Result<()> {
        if let Some(participant) = event_participant(event) {
            let listed = self.room.participant(participant.identity())?;
            let departure = matches!(event, RoomEvent::ParticipantDisconnected { .. });
            if !self.state.lock().current_event(
                participant.identity(),
                participant.sid(),
                listed.as_ref().map(|listed| listed.sid()),
                departure,
            ) {
                return Ok(());
            }
        }
        if let Some(input) = &self.audio_input {
            input.handle_event(event)?;
        }
        if let Some(video) = &self.video_input {
            video.handle_event(event)?;
        }
        if let Some(user_tr) = &self.user_tr {
            user_tr.handle_event(event).await?;
        }
        if let Some(agent_tr) = &self.agent_tr {
            agent_tr.handle_event(event).await?;
        }

        match event {
            RoomEvent::ConnectionStateChanged { state } => {
                self.emit(RoomIoEvent::ConnectionStateChanged(*state))?;
                if *state == ConnectionState::Connected {
                    self.connected.set(true);
                    // Connected or reconnected: the agent's own identity for the session's spans.
                    let local = self.room.local();
                    self.emit(RoomIoEvent::ParticipantInfo {
                        participant: ParticipantInfo {
                            identity: local.identity()?,
                            sid: local.sid()?,
                            kind: local.kind()?,
                        },
                        local: true,
                    })?;
                }
            }
            RoomEvent::ParticipantConnected { participant } => {
                let attrs = participant.attributes()?;
                self.consider(
                    participant.identity(),
                    participant.sid(),
                    participant.kind(),
                    &attrs,
                )
                .await?;
            }
            RoomEvent::ParticipantDisconnected { participant, reason } => {
                self.on_participant_disconnected(
                    participant.identity(),
                    participant.sid(),
                    *reason,
                )
                .await?;
            }
            RoomEvent::Close { reason } => {
                if let Some(identity) = self.linked_participant() {
                    self.unset_participant().await?;
                    self.emit(RoomIoEvent::ParticipantUnlinked(ParticipantUnlinked {
                        identity,
                        reason: *reason,
                    }))?;
                }
                self.request_close();
            }
            RoomEvent::TrackSubscribed { .. }
            | RoomEvent::TrackUnsubscribed { .. }
            | RoomEvent::TrackPublished { .. }
            | RoomEvent::TrackUnpublished { .. }
            | RoomEvent::LocalTrackPublished { .. }
            | RoomEvent::ParticipantActive { .. }
            | RoomEvent::ParticipantAttributesChanged { .. }
            | RoomEvent::SipDtmf { participant: None, .. } => {}
            RoomEvent::SipDtmf { participant: Some(who), digit } => {
                if self.state.lock().accepts_key_press(who) {
                    self.emit(RoomIoEvent::KeyPressed(digit.clone()))?;
                }
            }
        }
        Ok(())
    }

    async fn on_participant_disconnected(
        &self,
        identity: &ParticipantIdentity,
        sid: &str,
        reason: DisconnectReason,
    ) -> Result<()> {
        if !self.transition(|state| state.disconnect(identity, sid)) {
            return Ok(());
        }
        self.attach_participant(None).await?;
        // Whether the departure closes the session is `RoomLink`'s.
        self.emit(RoomIoEvent::ParticipantUnlinked(ParticipantUnlinked {
            identity: identity.clone(),
            reason,
        }))?;
        Ok(())
    }

    async fn write_agent_state(self: Arc<Self>) {
        let mut updates = self.agent_state.subscribe();
        loop {
            let stopped = self.stopped.notified();
            tokio::pin!(stopped);
            stopped.as_mut().enable();
            if self.is_closing() {
                return;
            }
            let pending = updates.borrow_and_update().clone();
            let result = async {
                if let Some(state) = pending
                    && self.room.connection_state()?.is_connected()
                {
                    self.room
                        .local()
                        .set_attributes(
                            &telemetry::Cx::NONE,
                            BTreeMap::from([(ATTRIBUTE_AGENT_STATE.to_owned(), state)]),
                        )
                        .await?;
                }
                Result::Ok(())
            };
            tokio::select! {
                biased;
                () = &mut stopped => return,
                changed = updates.changed() => { if changed.is_err() { return; } continue; },
                result = result => { if let Err(error) = result { self.fail(error); return; } },
            }
            tokio::select! {
                biased;
                () = &mut stopped => return,
                changed = updates.changed() => { if changed.is_err() { return; } },
            }
        }
    }

    fn register_text_input(self: &Arc<Self>) {
        if !self.begin_work() {
            return;
        }
        let streams = crate::rtc::events::text_streams(self.room.as_ref(), TOPIC_CHAT);
        let Ok(streams) = streams else {
            self.finish_starting();
            return;
        };
        *self.chat_registered.lock() = true;
        let owner = self.clone();
        self.spawn_reserved("room chat listener", async move {
            owner.read_chat(streams).await;
        });
    }

    async fn read_chat(
        self: Arc<Self>,
        mut streams: lk_agents_abi::voice::rtc::Receiver<lk_agents_abi::voice::rtc::TextStream>,
    ) {
        let result = async {
            loop {
                let stopped = self.stopped.notified();
                tokio::pin!(stopped);
                stopped.as_mut().enable();
                if self.is_closing() {
                    return Ok(());
                }
                let stream = tokio::select! {
                    biased;
                    () = &mut stopped => return Ok(()),
                    stream = streams.next() => stream.map_err(IoError::from)?,
                };
                let Some(stream) = stream else {
                    return Ok(());
                };
                // Drop the state read guard before entering the SDK: host callbacks may reenter.
                let attachment = self.state.lock().attachment(&stream.sender);
                let attachment = match attachment {
                    Some(id) if self.room.participant(&stream.sender)?.is_some() => Some(id),
                    _ => None,
                };
                let Some(attachment) = attachment.filter(|_| self.begin_work()) else {
                    let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
                    self.record_cleanup(stream.close(deadline).await);
                    continue;
                };
                let owner = self.clone();
                self.spawn_reserved("room chat reader", async move {
                    owner.read_chat_message(attachment, stream).await;
                });
            }
        }
        .await;
        let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        self.record_cleanup(streams.close(deadline).await);
        let registered = std::mem::take(&mut *self.chat_registered.lock());
        if registered && self.room.unregister_text_stream_handler(TOPIC_CHAT).is_err() {
            self.record_cleanup(Closed::Failed);
        }
        if let Err(error) = result {
            self.fail(error);
        }
    }

    async fn read_chat_message(
        self: Arc<Self>,
        attachment: u64,
        mut stream: lk_agents_abi::voice::rtc::TextStream,
    ) {
        let stopped = self.stopped.notified();
        tokio::pin!(stopped);
        stopped.as_mut().enable();
        let result = if self.is_closing() {
            None
        } else {
            tokio::select! {
                biased;
                () = &mut stopped => None,
                text = read_text(&mut stream.chunks) => Some(text),
            }
        };
        let sender = stream.sender.clone();
        let info = stream.info.clone();
        let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
        let cleanup = stream.close(deadline).await;
        self.record_cleanup(cleanup);
        match result {
            Some(Ok(text)) if cleanup == Closed::Clean => {
                let accepts = self.state.lock().accepts(attachment, &sender);
                if accepts
                    && let Err(error) = self.emit(RoomIoEvent::TextInput(TextInput {
                        text,
                        participant: sender,
                        info,
                    }))
                {
                    self.fail(error);
                }
            }
            Some(Err(error)) => self.fail(error.into()),
            _ => {}
        }
    }

    fn reserve_event(&self) -> Result<mpsc::Permit<'_, RoomIoEvent>> {
        self.events_tx.try_reserve().map_err(|error| {
            IoError::Rtc(match error {
                mpsc::error::TrySendError::Full(_) => "room event capacity exceeded".into(),
                mpsc::error::TrySendError::Closed(_) => "room event consumer is closed".into(),
            })
        })
    }

    fn emit(&self, event: RoomIoEvent) -> Result<()> {
        self.reserve_event()?.send(event);
        Ok(())
    }

    fn begin_work(&self) -> bool {
        let mut state = self.execution.lock();
        if state.closing {
            return false;
        }
        state.starting += 1;
        true
    }

    fn finish_starting(&self) {
        self.execution.lock().starting -= 1;
        self.started.notify_waiters();
    }

    fn spawn_reserved(&self, name: &str, work: impl Future<Output = ()> + Send + 'static) {
        self.tasks.spawn(self.spawner.as_ref(), name, work);
        self.finish_starting();
    }

    fn is_closing(&self) -> bool {
        self.execution.lock().closing
    }

    fn record_cleanup(&self, outcome: Closed) {
        let mut state = self.execution.lock();
        state.cleanup = state.cleanup.and(outcome);
    }

    fn transition<T>(&self, transition: impl FnOnce(&mut RoomIoState) -> T) -> T {
        let mut state = self.state.lock();
        let outcome = transition(&mut state);
        self.view.send_replace(state.state());
        outcome
    }

    fn request_close(self: &Arc<Self>) {
        let closing = {
            let mut execution = self.execution.lock();
            if execution.closing {
                return;
            }
            execution.closing = true;
            self.transition(RoomIoState::begin_close)
        };
        self.stopped.notify_waiters();
        if closing {
            let owner = self.clone();
            let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
            let close = self.spawner.spawn(async move {
                let outcome = owner.close(deadline).await;
                owner.transition(|state| state.finish_close(outcome));
            });
            let _ = self.closing.set(close);
        }
    }

    /// `closing`, given what is left of `deadline`.
    async fn within(&self, deadline: Ts, closing: impl Future<Output = Closed>) -> Closed {
        let remaining = deadline.saturating_duration_since(self.clock.now());
        lk_agents_abi::timeout(self.clock.as_ref(), remaining, closing)
            .await
            .unwrap_or(Closed::TimedOut)
    }

    async fn close(&self, deadline: Ts) -> Closed {
        loop {
            let started = self.started.notified();
            tokio::pin!(started);
            started.as_mut().enable();
            if self.execution.lock().starting == 0 {
                break;
            }
            if lk_agents_abi::timeout(
                self.clock.as_ref(),
                deadline.saturating_duration_since(self.clock.now()),
                &mut started,
            )
            .await
            .is_err()
            {
                break;
            }
        }
        let joined = self.tasks.wait_empty(self.clock.as_ref(), Some(deadline)).await;
        let mut outcome = self.execution.lock().cleanup;
        if !joined || self.execution.lock().starting != 0 {
            outcome = outcome.and(Closed::TimedOut);
        }
        if let Some(pc) = &self.pre_connect {
            outcome = outcome.and(self.within(deadline, pc.aclose()).await);
        }
        if let Some(input) = &self.audio_input {
            outcome = outcome.and(self.within(deadline, input.aclose()).await);
        }
        if let Some(video) = &self.video_input {
            // `aclose` makes a waiting `next_frame` return `None`.
            outcome = outcome.and(self.within(deadline, video.aclose()).await);
        }
        // Each owned performer keeps retiring after its waiter reaches the common deadline.
        for tr in [&self.user_tr, &self.agent_tr].into_iter().flatten() {
            let closed =
                async { if tr.aclose().await.is_ok() { Closed::Clean } else { Closed::Failed } };
            outcome = outcome.and(self.within(deadline, closed).await);
        }
        if let Some(out) = &self.audio_output {
            outcome = outcome.and(self.within(deadline, out.aclose()).await);
        }
        outcome
    }
}

impl room_owner::RoomIo for RoomIo {
    fn forward_events(
        self: Arc<Self>,
        listener: Arc<dyn Fn(RoomIoEvent) -> bool + Send + Sync>,
        rtc_listener: Arc<dyn Fn(RoomEvent) -> bool + Send + Sync>,
    ) {
        if !self.begin_work() {
            return;
        }
        let Some((mut events, mut failures)) = self.events() else {
            self.finish_starting();
            return;
        };
        let mut rtc_events = self.room.events();
        let owner = self.clone();
        self.spawn_reserved("room session events", async move {
            loop {
                let stopped = owner.stopped.notified();
                tokio::pin!(stopped);
                stopped.as_mut().enable();
                if owner.is_closing() {
                    // What the room said before the stop still reaches the session: the room's
                    // `Close` stops this loop at the instant it is queued here (seed 19655).
                    while let Ok(event) = events.try_recv() && listener(event) {}
                    while let Ok(failure) = failures.try_recv()
                        && listener(RoomIoEvent::InputProcessorFailed(failure))
                    {}
                    while let Some(Ok(Some(event))) = rtc_events.next().now_or_never()
                        && rtc_listener(event)
                    {}
                    break;
                }
                tokio::select! {
                    biased;
                    () = &mut stopped => continue,
                    event = events.recv() => { if event.is_none_or(|event| !listener(event)) { break; } },
                    // Without an audio input nothing sends, and the arm is off.
                    Some(failure) = failures.recv() => {
                        if !listener(RoomIoEvent::InputProcessorFailed(failure)) { break; }
                    },
                    event = rtc_events.next() => match event {
                        Ok(Some(event)) => { if !rtc_listener(event) { break; } },
                        Ok(None) => break,
                        Err(error) => { owner.fail(IoError::from(error)); break; },
                    },
                }
            }
            let deadline = owner.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
            owner.record_cleanup(rtc_events.close(deadline).await);
        });
    }

    fn audio_input(&self) -> Option<Arc<dyn AudioInput>> {
        self.audio_input.clone().map(|a| a as Arc<dyn AudioInput>)
    }

    /// The microphone leg itself, as the session hands it to the host.
    fn audio_input_leg(&self) -> Option<Arc<dyn room_owner::RoomAudioInputPort>> {
        self.audio_input.clone().map(|input| input as Arc<dyn room_owner::RoomAudioInputPort>)
    }

    fn video_input(&self) -> Option<Arc<dyn room_owner::RoomVideoInputPort>> {
        self.video_input.clone().map(|input| input as Arc<dyn room_owner::RoomVideoInputPort>)
    }

    /// The caller wraps it with a synchroniser if configured.
    fn audio_output(&self) -> Option<Arc<dyn AudioOutput>> {
        self.audio_output.clone().map(|a| a as Arc<dyn AudioOutput>)
    }

    fn transcription_output(&self) -> Option<Arc<dyn TextOutput>> {
        self.agent_tr.clone().map(|a| a as Arc<dyn TextOutput>)
    }

    fn wants_transcript_sync(&self, external_audio: bool) -> bool {
        lk_agents_voice::room_io::wants_transcript_sync(
            &self.config,
            self.audio_output.is_some() || external_audio,
        )
    }

    fn linked_participant(&self) -> Option<ParticipantIdentity> {
        self.state.lock().state().linked
    }

    fn start(self: Arc<Self>, startup: telemetry::Cx) {
        if !self.transition(RoomIoState::start) {
            return;
        }
        if let Some(input) = &self.audio_input {
            input.observe(startup.clone().continuing(None));
        }
        if let Some(span) = startup.span() {
            let cx = startup.under::<telemetry::SessionStart>(span);
            let filter = self.config.participant_identity.is_some();
            let participant =
                telemetry::start(&cx, telemetry::WaitForParticipant { participant_filter: filter });
            *self.startup.lock() = Some(Startup { cx, participant: Some(participant) });
        }
        if let Some(pc) = &self.pre_connect {
            pc.register();
        }
        if self.config.text_input {
            self.register_text_input();
        }
        if self.begin_work() {
            let this = self.clone();
            let events = self.room.events();
            self.spawn_reserved("room events", async move {
                this.read_events(events).await;
            });
        }
        if self.begin_work() {
            let this = self.clone();
            self.spawn_reserved("room startup", async move {
                let stopped = this.stopped.notified();
                tokio::pin!(stopped);
                stopped.as_mut().enable();
                if this.is_closing() { return; }
                tokio::select! {
                    biased;
                    () = &mut stopped => {},
                    result = this.clone().init() => { if let Err(error) = result { this.fail(error); } },
                }
            });
        }
        if self.begin_work() {
            let this = self.clone();
            self.spawn_reserved("room participant attributes", async move {
                this.write_agent_state().await;
            });
        }
    }

    /// Keep one ordered writer; a newer state cancels the pending older write.
    fn on_agent_state_changed(self: Arc<Self>, state: &str) {
        if !self.is_closing() {
            self.agent_state.send_replace(Some(state.to_owned()));
        }
    }

    /// `wait_for_ready`.
    fn wait_for_ready<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let mut view = self.view.subscribe();
            loop {
                if let Some(outcome) = view.borrow_and_update().ready.clone() {
                    return outcome;
                }
                if view.changed().await.is_err() {
                    return Err(IoError::Rtc("room readiness publisher closed".into()));
                }
            }
        })
    }

    /// `set_participant`.
    fn set_participant<'a>(
        &'a self,
        identity: Option<ParticipantIdentity>,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let participant = match &identity {
                Some(identity) => self.room.participant(identity)?,
                None => None,
            };
            let listed = participant.as_ref().map(|p| p.sid().to_owned());
            let Some(relink) =
                self.transition(|state| state.set_participant(identity.clone(), listed.as_deref()))
            else {
                return Ok(());
            };
            if relink.announce
                && let Some(identity) = &identity
            {
                let outbound_sip = match &participant {
                    Some(p) => outbound_sip(p.kind(), &p.attributes()?),
                    None => false,
                };
                if let Some(p) = &participant {
                    self.emit(RoomIoEvent::ParticipantInfo {
                        participant: ParticipantInfo {
                            identity: identity.clone(),
                            sid: p.sid().to_owned(),
                            kind: p.kind(),
                        },
                        local: false,
                    })?;
                    self.linked_at_startup(identity, p.sid(), p.kind());
                }
                self.emit(RoomIoEvent::ParticipantLinked {
                    participant: ParticipantLinked { identity: identity.clone() },
                    outbound_sip,
                })?;
            }
            if relink.detach {
                self.attach_participant(None).await?;
            }
            match identity {
                Some(identity) => self.attach_participant(Some(identity)).await,
                None => Ok(()),
            }
        })
    }

    fn unset_participant<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        self.set_participant(None)
    }

    /// `_on_user_input_transcribed`, then `_forward_user_transcript`, as one await.
    fn on_user_input_transcribed<'a>(
        &'a self,
        transcript: &'a str,
        is_final: bool,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let Some(user_tr) = &self.user_tr else { return Ok(()) };
            user_tr.capture_text(&telemetry::Cx::NONE, transcript).await?;
            if is_final {
                user_tr.flush();
            }
            Ok(())
        })
    }

    fn aclose(self: Arc<Self>) -> BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.request_close();
            let mut view = self.view.subscribe();
            loop {
                if let Some(outcome) = view.borrow_and_update().closed {
                    return outcome;
                }
                if view.changed().await.is_err() {
                    return Closed::Failed;
                }
            }
        })
    }
}

/// A chat message's whole text; ingress admission already bounds its bytes.
async fn read_text(
    chunks: &mut lk_agents_abi::voice::rtc::Receiver<String>,
) -> std::result::Result<String, lk_agents_abi::errors::ProviderError> {
    let mut text = String::new();
    while let Some(chunk) = chunks.next().await? {
        text.push_str(&chunk);
    }
    Ok(text)
}

/// The remote participant a room event is about, whose lifetime
/// the room state checks.
pub fn event_participant(event: &RoomEvent) -> Option<&Arc<dyn RemoteParticipant>> {
    match event {
        RoomEvent::TrackSubscribed { participant, .. }
        | RoomEvent::TrackUnsubscribed { participant, .. }
        | RoomEvent::TrackPublished { participant, .. }
        | RoomEvent::TrackUnpublished { participant, .. }
        | RoomEvent::ParticipantConnected { participant }
        | RoomEvent::ParticipantActive { participant }
        | RoomEvent::ParticipantAttributesChanged { participant }
        | RoomEvent::ParticipantDisconnected { participant, .. } => Some(participant),
        _ => None,
    }
}
