//! RTC execution for the synchronous room coordinator.

use super::input::{ParticipantAudioInput, ParticipantVideoInput};
use super::output::{
    ParticipantAudioOutput,
    transcription::{ParticipantTranscriptionOutput, SegmentIds},
};
use super::pre_connect_audio::PreConnectAudioHandler;
use lk_agents_abi::voice::io::IoError;
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, Result, TextOutput};
use lk_agents_abi::voice::room_io::{
    ParticipantLinked, ParticipantUnlinked, RoomIoConfig, TextInput,
};
use lk_agents_abi::voice::room_owner::RoomIoEvent;
use lk_agents_abi::voice::rtc::capabilities::RtcRoom;
use lk_agents_abi::voice::rtc::events::RoomEvent;
use lk_agents_abi::voice::rtc::{
    ConnectionState, DisconnectReason, ParticipantIdentity, ParticipantKind,
};
use lk_agents_abi::{
    ATTRIBUTE_AGENT_STATE, Clock, Closed, HOST_TEARDOWN_TIMEOUT, Monotone, Spawner, SpawnerExt,
    TOPIC_CHAT, TaskSet, Ts,
};
use lk_agents_voice::room_io::coordinator::{RoomCoordinator, event_participant};
use lk_agents_voice::room_io::{accepted_kinds, publishes_on_behalf_of};
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
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

    /// One coordinator owns decisions and wakes readers after each transition.
    coordinator: watch::Sender<RoomCoordinator>,
    connected: Monotone<bool>,

    events_tx: mpsc::Sender<RoomIoEvent>,
    events_rx: Mutex<Option<mpsc::Receiver<RoomIoEvent>>>,

    tasks: Arc<TaskSet>,
    execution: Mutex<Execution>,
    stopped: Notify,
    started: Notify,
    agent_state: watch::Sender<Option<String>>,
    chat_registered: Mutex<bool>,
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
        pool: Option<lk_agents_media::workers::MediaWorkers>,
    ) -> Arc<Self> {
        let room = crate::rtc::room::Room::wrap(room);
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

        let audio_input = config.audio_input.as_ref().map(|a| {
            ParticipantAudioInput::new(
                room.clone(),
                spawner.clone(),
                clock.clone(),
                a.clone(),
                pre_connect.clone(),
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

        let (events_tx, events_rx) = mpsc::channel(ROOM_EVENT_CAPACITY);
        let coordinator = RoomCoordinator::new(config.participant_identity.clone());
        Arc::new(RoomIo {
            coordinator: watch::channel(coordinator).0,
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
            events_rx: Mutex::new(Some(events_rx)),
            tasks: Arc::default(),
            execution: Mutex::new(Execution {
                closing: false,
                starting: 0,
                cleanup: Closed::Clean,
            }),
            stopped: Notify::new(),
            started: Notify::new(),
            agent_state: watch::channel(None).0,
            chat_registered: Mutex::new(false),
        })
    }

    pub fn events(&self) -> Option<mpsc::Receiver<RoomIoEvent>> {
        self.events_rx.lock().unwrap().take()
    }

    pub fn audio_input(&self) -> Option<Arc<dyn AudioInput>> {
        self.audio_input.clone().map(|a| a as Arc<dyn AudioInput>)
    }

    /// The microphone leg itself, as the session hands it to the host.
    pub fn audio_input_leg(&self) -> Option<Arc<ParticipantAudioInput>> {
        self.audio_input.clone()
    }

    pub fn video_input(&self) -> Option<Arc<ParticipantVideoInput>> {
        self.video_input.clone()
    }

    /// The caller wraps it with a synchroniser if configured.
    pub fn audio_output(&self) -> Option<Arc<dyn AudioOutput>> {
        self.audio_output.clone().map(|a| a as Arc<dyn AudioOutput>)
    }

    pub fn transcription_output(&self) -> Option<Arc<dyn TextOutput>> {
        self.agent_tr.clone().map(|a| a as Arc<dyn TextOutput>)
    }

    pub fn wants_transcript_sync(&self, has_external_audio_output: bool) -> bool {
        lk_agents_voice::room_io::wants_transcript_sync(
            &self.config,
            self.audio_output.is_some() || has_external_audio_output,
        )
    }

    pub fn linked_participant(&self) -> Option<ParticipantIdentity> {
        self.coordinator.borrow().state().linked
    }

    /// `wait_for_ready`.
    pub async fn wait_for_ready(&self) -> Result<()> {
        let mut state = self.coordinator.subscribe();
        let state = state
            .wait_for(|state| state.state().ready.is_some())
            .await
            .expect("RoomIo retains the state sender");
        state.state().ready.expect("readiness matched a terminal outcome")
    }

    pub fn start(self: &Arc<Self>) {
        if !self.coordinate(RoomCoordinator::start) {
            return;
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
                        self.on_agent_state_changed(&state);
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
        self.coordinate(|coordinator| coordinator.finish_start(Err(error)));
        self.record_cleanup(Closed::Failed);
        self.request_close();
    }

    /// `_init_task`.
    async fn init(self: Arc<Self>) -> Result<()> {
        self.connected.wait_for(|set| *set).await;

        // Replay participants who joined before the handler was registered.
        for p in self.room.remote_participants()? {
            let attrs = p.attributes()?;
            self.consider(p.identity(), p.sid(), p.kind(), publishes_on_behalf_of(&attrs)).await?;
        }

        {
            let mut state = self.coordinator.subscribe();
            let state = state
                .wait_for(|coordinator| {
                    let state = coordinator.state();
                    state.linked.is_some() || state.ready.is_some()
                })
                .await;
            match state {
                Ok(state) if state.state().linked.is_some() => {}
                Ok(_) | Err(_) => return Ok(()),
            }
        }

        if let Some(agent_tr) = &self.agent_tr {
            // The agent transcribes *itself*, so its transcript hangs off its own track.
            agent_tr.set_participant(Some(self.room.local().identity()?)).await?;
        }
        let outcome = match &self.audio_output {
            Some(out) => out.start().await,
            None => Ok(()),
        };
        self.coordinate(|coordinator| coordinator.finish_start(outcome));
        Ok(())
    }

    /// `set_participant`.
    pub async fn set_participant(&self, identity: Option<ParticipantIdentity>) -> Result<()> {
        let participant = match &identity {
            Some(identity) => self.room.participant(identity)?,
            None => None,
        };
        let listed = participant.as_ref().map(|p| p.sid().to_owned());
        let Some(relink) = self.coordinate(|coordinator| {
            coordinator.set_participant(identity.clone(), listed.as_deref())
        }) else {
            return Ok(());
        };
        if relink.announce
            && let Some(identity) = &identity
        {
            self.emit(RoomIoEvent::ParticipantLinked(ParticipantLinked {
                identity: identity.clone(),
            }))?;
        }
        if relink.detach {
            self.attach_participant(None).await?;
        }
        match identity {
            Some(identity) => self.attach_participant(Some(identity)).await,
            None => Ok(()),
        }
    }

    pub async fn unset_participant(&self) -> Result<()> {
        self.set_participant(None).await
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
        on_behalf_of: Option<&str>,
    ) -> Result<()> {
        let agent_identity = self.room.local().identity()?;
        if self.coordinate(|coordinator| {
            coordinator.consider(
                candidate,
                sid,
                kind,
                on_behalf_of,
                &agent_identity,
                accepted_kinds(&self.config),
            )
        }) {
            self.emit(RoomIoEvent::ParticipantLinked(ParticipantLinked {
                identity: candidate.clone(),
            }))?;
            self.attach_participant(None).await?;
            self.attach_participant(Some(candidate.clone())).await?;
        }
        Ok(())
    }

    async fn handle_event(self: &Arc<Self>, event: &RoomEvent) -> Result<()> {
        if let Some(participant) = event_participant(event) {
            let listed = self.room.participant(participant.identity())?;
            let departure = matches!(event, RoomEvent::ParticipantDisconnected { .. });
            if !self.coordinator.borrow().current_event(
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
                if *state == ConnectionState::Connected {
                    self.connected.set(true);
                }
            }
            RoomEvent::ParticipantConnected { participant } => {
                let attrs = participant.attributes()?;
                self.consider(
                    participant.identity(),
                    participant.sid(),
                    participant.kind(),
                    publishes_on_behalf_of(&attrs),
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
            RoomEvent::SipDtmf { participant: Some(who), .. } => {
                if self.coordinator.borrow().accepts_key_press(who) {
                    self.emit(RoomIoEvent::KeyPressed)?;
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
        if !self.coordinate(|coordinator| coordinator.disconnect(identity, sid)) {
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

    /// Keep one ordered writer; a newer state cancels the pending older write.
    pub fn on_agent_state_changed(self: &Arc<Self>, state: &str) {
        if !self.is_closing() {
            self.agent_state.send_replace(Some(state.to_owned()));
        }
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
                        .set_attributes(BTreeMap::from([(ATTRIBUTE_AGENT_STATE.to_owned(), state)]))
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

    /// `_on_user_input_transcribed`, then `_forward_user_transcript`, as one await.
    pub async fn on_user_input_transcribed(&self, transcript: &str, is_final: bool) -> Result<()> {
        let Some(user_tr) = &self.user_tr else { return Ok(()) };
        user_tr.capture_text(transcript).await?;
        if is_final {
            user_tr.flush();
        }
        Ok(())
    }

    fn register_text_input(self: &Arc<Self>) {
        if !self.begin_work() {
            return;
        }
        let streams = self.room.text_streams(TOPIC_CHAT);
        let Ok(streams) = streams else {
            self.finish_starting();
            return;
        };
        *self.chat_registered.lock().unwrap() = true;
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
                let attachment =
                    self.coordinator.borrow().chat_attachment(&stream.sender, || {
                        self.room.participant(&stream.sender).map(|listed| listed.is_some())
                    })?;
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
        let registered = std::mem::take(&mut *self.chat_registered.lock().unwrap());
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
                let accepts = self.coordinator.borrow().accepts(attachment, &sender);
                if accepts
                    && let Err(error) = self.emit(RoomIoEvent::TextInput(TextInput {
                        text,
                        participant: sender,
                        info: Box::new(info),
                    }))
                {
                    self.fail(error);
                }
            }
            Some(Err(error)) => self.fail(error.into()),
            _ => {}
        }
    }

    pub(super) fn forward_events(
        self: &Arc<Self>,
        listener: Arc<dyn Fn(RoomIoEvent) -> bool + Send + Sync>,
        rtc_listener: Arc<dyn Fn(RoomEvent) -> bool + Send + Sync>,
    ) {
        if !self.begin_work() {
            return;
        }
        let Some(mut events) = self.events() else {
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
                if owner.is_closing() { break; }
                tokio::select! {
                    biased;
                    () = &mut stopped => break,
                    event = events.recv() => { if event.is_none_or(|event| !listener(event)) { break; } },
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
        let mut state = self.execution.lock().unwrap();
        if state.closing {
            return false;
        }
        state.starting += 1;
        true
    }

    fn finish_starting(&self) {
        self.execution.lock().unwrap().starting -= 1;
        self.started.notify_waiters();
    }

    fn spawn_reserved(&self, name: &str, work: impl Future<Output = ()> + Send + 'static) {
        self.tasks.spawn(self.spawner.as_ref(), name, work);
        self.finish_starting();
    }

    fn is_closing(&self) -> bool {
        self.execution.lock().unwrap().closing
    }

    fn record_cleanup(&self, outcome: Closed) {
        let mut state = self.execution.lock().unwrap();
        state.cleanup = state.cleanup.and(outcome);
    }

    fn coordinate<T>(&self, transition: impl FnOnce(&mut RoomCoordinator) -> T) -> T {
        let mut outcome = None;
        self.coordinator.send_modify(|coordinator| outcome = Some(transition(coordinator)));
        outcome.expect("watch applies each transition synchronously")
    }

    fn request_close(self: &Arc<Self>) {
        let closing = {
            let mut execution = self.execution.lock().unwrap();
            if execution.closing {
                return;
            }
            execution.closing = true;
            self.coordinate(RoomCoordinator::begin_close)
        };
        self.stopped.notify_waiters();
        if closing {
            let owner = self.clone();
            let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
            self.spawner.spawn_unboxed(async move {
                let outcome = owner.close(deadline).await;
                owner.coordinate(|coordinator| coordinator.finish_close(outcome));
            });
        }
    }

    pub async fn aclose(self: &Arc<Self>) -> Closed {
        self.request_close();
        let mut state = self.coordinator.subscribe();
        let state = state
            .wait_for(|state| state.state().closed.is_some())
            .await
            .expect("RoomIo retains the cleanup result sender");
        state.state().closed.expect("cleanup matched a terminal outcome")
    }

    async fn close(&self, deadline: Ts) -> Closed {
        loop {
            let started = self.started.notified();
            tokio::pin!(started);
            started.as_mut().enable();
            if self.execution.lock().unwrap().starting == 0 {
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
        let mut outcome = self.execution.lock().unwrap().cleanup;
        if !joined || self.execution.lock().unwrap().starting != 0 {
            outcome = outcome.and(Closed::TimedOut);
        }
        if let Some(pc) = &self.pre_connect {
            outcome = outcome.and(
                lk_agents_abi::timeout(
                    self.clock.as_ref(),
                    deadline.saturating_duration_since(self.clock.now()),
                    pc.aclose(),
                )
                .await
                .unwrap_or(Closed::TimedOut),
            );
        }
        if let Some(input) = &self.audio_input {
            outcome = outcome.and(
                lk_agents_abi::timeout(
                    self.clock.as_ref(),
                    deadline.saturating_duration_since(self.clock.now()),
                    input.aclose(),
                )
                .await
                .unwrap_or(Closed::TimedOut),
            );
        }
        if let Some(video) = &self.video_input {
            // `aclose` makes a waiting `next_frame` return `None`.
            outcome = outcome.and(
                lk_agents_abi::timeout(
                    self.clock.as_ref(),
                    deadline.saturating_duration_since(self.clock.now()),
                    video.aclose(),
                )
                .await
                .unwrap_or(Closed::TimedOut),
            );
        }
        // Each owned performer keeps retiring after its waiter reaches the common deadline.
        if let Some(user_tr) = &self.user_tr {
            outcome = outcome.and(
                match lk_agents_abi::timeout(
                    self.clock.as_ref(),
                    deadline.saturating_duration_since(self.clock.now()),
                    user_tr.aclose(),
                )
                .await
                {
                    Ok(Ok(())) => Closed::Clean,
                    Ok(Err(_)) => Closed::Failed,
                    Err(_) => Closed::TimedOut,
                },
            );
        }
        if let Some(agent_tr) = &self.agent_tr {
            outcome = outcome.and(
                match lk_agents_abi::timeout(
                    self.clock.as_ref(),
                    deadline.saturating_duration_since(self.clock.now()),
                    agent_tr.aclose(),
                )
                .await
                {
                    Ok(Ok(())) => Closed::Clean,
                    Ok(Err(_)) => Closed::Failed,
                    Err(_) => Closed::TimedOut,
                },
            );
        }
        if let Some(out) = &self.audio_output {
            let remaining = deadline.saturating_duration_since(self.clock.now());
            let closed = lk_agents_abi::timeout(self.clock.as_ref(), remaining, out.aclose())
                .await
                .unwrap_or(Closed::TimedOut);
            outcome = outcome.and(closed);
        }
        outcome
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

#[cfg(test)]
#[path = "owner_tests.rs"]
pub(super) mod tests;
