pub mod _input;
pub mod _output;
pub mod _pre_connect_audio;
mod coordinator;
mod publication;
pub mod rooms;
#[cfg(test)]
mod tests;

use self::coordinator::RoomCoordinator;
use crate::agent_session::effects::{Out, StreamKind};
use crate::agent_session::{Close, Driver, HostEvent};
use crate::room_io::_input::{ParticipantAudioInput, ParticipantVideoInput};
use crate::room_io::_output::ParticipantAudioOutput;
use crate::room_io::_output::transcription::{ParticipantTranscriptionOutput, SegmentIds};
use crate::room_io::_pre_connect_audio::PreConnectAudioHandler;
use crate::trace::{RoomLinkStep, RoomStep, Step};
use lk_agents_abi::AbortHandle;
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::voice::io::Result;
use lk_agents_abi::voice::io::{
    AudioInput as IoRoom_IoAudioInput, AudioOutput as IoRoom_IoAudioOutput, TextOutput,
};
pub use lk_agents_abi::voice::room_io::{
    AudioInputConfig, AudioOutputConfig, DeleteRoomRequested, ParticipantLinked,
    ParticipantUnlinked, RoomIoConfig, TextInput, TextOutputConfig,
};
use lk_agents_abi::{
    ATTRIBUTE_AGENT_STATE, ATTRIBUTE_PUBLISH_ON_BEHALF, Clock, Closed, HOST_TEARDOWN_TIMEOUT,
    Monotone, Spawner, SpawnerExt, TOPIC_CHAT, Ts,
};
use lk_agents_rtc::{
    ConnectionState, DisconnectReason, LocalParticipant, ParticipantIdentity, ParticipantKind,
    RoomEvent, RtcRoom, TrackSid, TrackSource,
};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::HashMap;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, watch};

/// `job.DEFAULT_PARTICIPANT_KINDS`: no `Agent` or `Egress`, so agents never auto-link to each other.
pub const DEFAULT_PARTICIPANT_KINDS: &[ParticipantKind] =
    &[ParticipantKind::Connector, ParticipantKind::Sip, ParticipantKind::Standard];

/// The track sources the audio input leg will read. Microphone only, as in the reference.
pub const AUDIO_INPUT_SOURCES: &[TrackSource] = &[TrackSource::Microphone];

/// `_pre_connect_audio.PRE_CONNECT_AUDIO_BUFFER_STREAM`.
pub const PRE_CONNECT_AUDIO_BUFFER_STREAM: &str = "lk.agent.pre-connect-audio-buffer";

/// `PreConnectAudioHandler._max_delta_s`: a buffer older than this is stale and dropped.
pub const PRE_CONNECT_MAX_DELTA: Duration = Duration::from_secs(1);

/// The kinds a [`RoomIoConfig`] auto-links: `participant_kinds`, or the defaults when empty.
pub fn accepted_kinds(config: &RoomIoConfig) -> &[ParticipantKind] {
    if config.participant_kinds.is_empty() {
        DEFAULT_PARTICIPANT_KINDS
    } else {
        &config.participant_kinds
    }
}

/// `RecorderIO._write_task`'s `WRITE_INTERVAL`, on the session clock.
pub const RECORDER_FLUSH: Duration = Duration::from_millis(2500);

pub struct StreamOpened {
    pub open: u64,
    pub leg: StreamKind,
    pub outcome: std::result::Result<Opened, lk_agents_abi::errors::ProviderError>,
}

pub enum Opened {
    /// Runtime owns both sides until the core accepts or discards the prepared resource.
    Stt(lk_agents_abi::SttSink, lk_agents_abi::SttSource),
    Vad(lk_agents_abi::VadSink, lk_agents_abi::VadSource),
}

impl Opened {
    /// A stream nobody takes still reaches the provider's `close`, or a plugin leaks its
    /// connection.
    pub async fn close(self, deadline: lk_agents_abi::Ts) {
        let _ = match self {
            Opened::Stt(_, source) => source.close(deadline).await,
            Opened::Vad(_, source) => source.close(deadline).await,
        };
    }
}

/// The legs an agent has after `reopen_legs`: given, or the table's row where none was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InputProviders {
    pub(crate) stt: bool,
    pub(crate) vad: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoomIoEvent {
    /// `AgentSession._on_room_io_participant_linked`.
    ParticipantLinked(ParticipantLinked),
    /// The linked participant left; whether that closes the session is `RoomLink`'s call.
    ParticipantUnlinked(ParticipantUnlinked),
    /// A `lk.chat` message. `text_input_cb(session, TextInputEvent(...))`.
    TextInput(TextInput),
    /// `job_ctx.delete_room(room_name)`, a request since deletion must outlive the session.
    DeleteRoomRequested(DeleteRoomRequested),
    /// The linked participant pressed a keypad key: `reset_away_timer`. Not on the host stream.
    KeyPressed,
}

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

    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    events_tx: mpsc::UnboundedSender<RoomIoEvent>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    events_rx: Mutex<Option<mpsc::UnboundedReceiver<RoomIoEvent>>>,

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    tasks: Mutex<Vec<AbortHandle>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// The in-flight `set_attributes`, cancelled before the next as in the reference.
    state_task: Mutex<Option<AbortHandle>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// The agent's last state, written again once the room connects.
    agent_state: Mutex<Option<String>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    chat_registered: Mutex<bool>,
}

impl RoomIo {
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub fn new(
        room: Arc<dyn RtcRoom>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        ids: Arc<SegmentIds>,
        config: RoomIoConfig,
        text_next_in_chain: Option<Arc<dyn TextOutput>>,
    ) -> Arc<Self> {
        let pre_connect = config.audio_input.as_ref().filter(|a| a.pre_connect_audio).map(|a| {
            PreConnectAudioHandler::new(
                room.clone(),
                spawner.clone(),
                clock.clone(),
                a.pre_connect_audio_timeout,
                (a.sample_rate, a.num_channels),
            )
        });

        let audio_input = config.audio_input.as_ref().map(|a| {
            ParticipantAudioInput::new(
                room.clone(),
                spawner.clone(),
                a.clone(),
                pre_connect.clone(),
            )
        });

        // Not `start()`ed: driven by `RoomIo::handle_event`.
        let video_input =
            config.video_input.then(|| ParticipantVideoInput::new(room.clone(), spawner.clone()));

        let audio_output = config.audio_output.as_ref().map(|a| {
            ParticipantAudioOutput::new(room.clone(), clock.clone(), spawner.clone(), a.clone())
        });

        let (user_tr, agent_tr) = match &config.text_output {
            Some(t) => (
                Some(ParticipantTranscriptionOutput::new(
                    room.clone(),
                    spawner.clone(),
                    ids.clone(),
                    false,
                    false,
                    None,
                )),
                Some(ParticipantTranscriptionOutput::new(
                    room.clone(),
                    spawner.clone(),
                    ids,
                    true,
                    t.json_format,
                    text_next_in_chain,
                )),
            ),
            None => (None, None),
        };

        let (events_tx, events_rx) = mpsc::unbounded_channel();
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
            tasks: Mutex::new(Vec::new()),
            state_task: Mutex::new(None),
            agent_state: Mutex::new(None),
            chat_registered: Mutex::new(false),
        })
    }

    #[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
    pub fn events(&self) -> Option<mpsc::UnboundedReceiver<RoomIoEvent>> {
        self.events_rx.lock().unwrap().take()
    }

    pub fn audio_input(&self) -> Option<Arc<dyn IoRoom_IoAudioInput>> {
        self.audio_input.clone().map(|a| a as Arc<dyn IoRoom_IoAudioInput>)
    }

    /// The microphone leg itself, as the session hands it to the host.
    pub fn audio_input_leg(&self) -> Option<Arc<ParticipantAudioInput>> {
        self.audio_input.clone()
    }

    pub fn video_input(&self) -> Option<Arc<ParticipantVideoInput>> {
        self.video_input.clone()
    }

    /// The caller wraps it with a synchroniser if configured.
    pub fn audio_output(&self) -> Option<Arc<dyn IoRoom_IoAudioOutput>> {
        self.audio_output.clone().map(|a| a as Arc<dyn IoRoom_IoAudioOutput>)
    }

    pub fn transcription_output(&self) -> Option<Arc<dyn TextOutput>> {
        self.agent_tr.clone().map(|a| a as Arc<dyn TextOutput>)
    }

    /// `NOT_GIVEN` means yes if there is an audio output (`has_external_audio_output`).
    pub fn wants_transcript_sync(&self, has_external_audio_output: bool) -> bool {
        let Some(t) = &self.config.text_output else { return false };
        if t.sync_transcription == Some(false) {
            return false;
        }
        self.audio_output.is_some() || has_external_audio_output
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

        let mut events = self.room.events();
        let this = self.clone();
        let handle = self.spawner.spawn_unboxed(async move {
            while let Some(ev) = events.recv().await {
                this.handle_event(&ev).await;
                // A state the agent took before the room connected was not written: write it now.
                if let RoomEvent::ConnectionStateChanged { state: ConnectionState::Connected } = ev
                {
                    let last = this.agent_state.lock().unwrap().clone();
                    if let Some(state) = last {
                        this.on_agent_state_changed(&state);
                    }
                }
            }
        });
        self.keep_task(handle);

        // The room may already be connected, so read the state rather than wait for the event.
        if self.room.connection_state().is_connected() {
            self.connected.set(true);
        }

        let this = self.clone();
        let handle = self.spawner.spawn_unboxed(async move { this.init().await });
        self.keep_task(handle);
    }

    /// `_init_task`.
    async fn init(self: Arc<Self>) {
        self.connected.wait_for(|set| *set).await;

        // Replay participants who joined before the handler was registered.
        for p in self.room.remote_participants() {
            let attrs = p.attributes();
            self.consider(p.identity(), p.kind(), publishes_on_behalf_of(&attrs)).await;
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
                Ok(_) | Err(_) => return,
            }
        }

        if let Some(agent_tr) = &self.agent_tr {
            // The agent transcribes *itself*, so its transcript hangs off its own track.
            agent_tr.set_participant(Some(self.room.local().identity())).await;
        }
        let outcome = match &self.audio_output {
            Some(out) => out.start().await,
            None => Ok(()),
        };
        self.coordinate(|coordinator| coordinator.finish_start(outcome));
    }

    /// `set_participant`.
    pub async fn set_participant(&self, identity: Option<ParticipantIdentity>) {
        let Some(identity) = identity else {
            self.unset_participant().await;
            return;
        };

        let present = self.room.participant(&identity).is_some();
        let Some(previous) = self.coordinate(|coordinator| {
            let previous = coordinator.state().linked;
            coordinator.select(Some(identity.clone()), present).then_some(previous)
        }) else {
            return;
        };
        if present && previous.as_ref() != Some(&identity) {
            let _ = self.events_tx.send(RoomIoEvent::ParticipantLinked(ParticipantLinked {
                identity: identity.clone(),
            }));
        }

        // A repeated selection retires the old attachment's buffered media too.
        if previous.as_ref() == Some(&identity) {
            self.attach_participant(None).await;
        }
        self.attach_participant(Some(identity)).await;
    }

    pub async fn unset_participant(&self) {
        if self.coordinate(|coordinator| coordinator.select(None, false)) {
            self.attach_participant(None).await;
        }
    }

    async fn attach_participant(&self, identity: Option<ParticipantIdentity>) {
        if let Some(input) = &self.audio_input {
            input.set_participant(identity.clone());
        }
        if let Some(video) = &self.video_input {
            video.set_participant(identity.clone());
        }
        if let Some(user_tr) = &self.user_tr {
            user_tr.set_participant(identity).await;
        }
    }

    /// `_on_participant_connected`'s filter.
    async fn consider(
        &self,
        candidate: &ParticipantIdentity,
        kind: lk_agents_rtc::ParticipantKind,
        on_behalf_of: Option<&str>,
    ) {
        let agent_identity = self.room.local().identity();
        if self.coordinate(|coordinator| {
            coordinator.consider(
                candidate,
                kind,
                on_behalf_of,
                &agent_identity,
                accepted_kinds(&self.config),
            )
        }) {
            let _ = self.events_tx.send(RoomIoEvent::ParticipantLinked(ParticipantLinked {
                identity: candidate.clone(),
            }));
            self.attach_participant(Some(candidate.clone())).await;
        }
    }

    async fn handle_event(self: &Arc<Self>, event: &RoomEvent) {
        if let Some(input) = &self.audio_input {
            input.handle_event(event);
        }
        if let Some(video) = &self.video_input {
            video.handle_event(event);
        }
        if let Some(user_tr) = &self.user_tr {
            user_tr.handle_event(event).await;
        }
        if let Some(agent_tr) = &self.agent_tr {
            agent_tr.handle_event(event).await;
        }

        match event {
            RoomEvent::ConnectionStateChanged { state } => {
                if *state == ConnectionState::Connected {
                    self.connected.set(true);
                }
            }
            RoomEvent::ParticipantConnected { participant } => {
                let attrs = participant.attributes();
                self.consider(
                    participant.identity(),
                    participant.kind(),
                    publishes_on_behalf_of(&attrs),
                )
                .await;
            }
            RoomEvent::ParticipantDisconnected { participant, reason } => {
                self.on_participant_disconnected(participant.identity(), *reason).await;
            }
            RoomEvent::Close { reason } => {
                if let Some(identity) = self.linked_participant() {
                    self.on_participant_disconnected(&identity, *reason).await;
                }
                self.aclose().await;
            }
            RoomEvent::TrackSubscribed { .. }
            | RoomEvent::TrackUnsubscribed { .. }
            | RoomEvent::TrackPublished { .. }
            | RoomEvent::TrackUnpublished { .. }
            | RoomEvent::LocalTrackPublished { .. }
            | RoomEvent::ParticipantActive { .. }
            | RoomEvent::ParticipantAttributesChanged { .. }
            | RoomEvent::SipDtmf { participant: None, .. } => {}
            // `_on_sip_dtmf_received`: a key the linked participant pressed is activity.
            RoomEvent::SipDtmf { participant: Some(who), .. } => {
                if self.coordinator.borrow().state().linked.as_ref() == Some(who) {
                    let _ = self.events_tx.send(RoomIoEvent::KeyPressed);
                }
            }
        }
    }

    async fn on_participant_disconnected(
        &self,
        identity: &ParticipantIdentity,
        reason: DisconnectReason,
    ) {
        if !self.coordinate(|coordinator| coordinator.disconnect(identity)) {
            return;
        }
        self.attach_participant(None).await;
        // Whether the departure closes the session is `RoomLink`'s.
        let _ = self.events_tx.send(RoomIoEvent::ParticipantUnlinked(ParticipantUnlinked {
            identity: identity.clone(),
            reason,
        }));
    }

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// `_on_agent_state_changed`: cancels the in-flight write so changes land in order.
    pub fn on_agent_state_changed(self: &Arc<Self>, state: &str) {
        if !self.coordinator.borrow().is_open() {
            return;
        }
        let room = self.room.clone();
        let state = state.to_owned();
        *self.agent_state.lock().unwrap() = Some(state.clone());
        let handle = self.spawner.spawn_unboxed(async move {
            if !room.connection_state().is_connected() {
                return;
            }
            let _ = room
                .local()
                .set_attributes(HashMap::from([(ATTRIBUTE_AGENT_STATE.to_owned(), state)]))
                .await;
        });
        let coordinator = self.coordinator.borrow();
        if coordinator.is_open() {
            if let Some(old) = self.state_task.lock().unwrap().replace(handle) {
                old.abort();
            }
        } else {
            handle.abort();
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

    /// `register_text_input`: the `lk.chat` topic.
    fn register_text_input(self: &Arc<Self>) {
        let Ok(mut streams) = self.room.text_streams(TOPIC_CHAT) else {
            // Already registered by another `RoomIO`. Warn-and-continue, as the reference does.
            return;
        };
        let registered = {
            let coordinator = self.coordinator.borrow();
            if coordinator.is_open() {
                *self.chat_registered.lock().unwrap() = true;
                true
            } else {
                false
            }
        };
        if !registered {
            self.room.unregister_text_stream_handler(TOPIC_CHAT);
            return;
        }

        let this = self.clone();
        let handle = self.spawner.spawn_unboxed(async move {
            while let Some(mut stream) = streams.recv().await {
                // The stream belongs to the attachment that accepted its header.
                let Some(attachment) = this.coordinator.borrow().attachment(&stream.sender) else {
                    continue;
                };
                if this.room.participant(&stream.sender).is_none() {
                    continue;
                }
                // One task per stream, so a stalled sender holds back no other message.
                let owner = this.clone();
                let reader = this.spawner.spawn_unboxed(async move {
                    let text = stream.read_all().await;
                    let coordinator = owner.coordinator.borrow();
                    if !coordinator.accepts(attachment, &stream.sender) {
                        return;
                    }
                    let _ = owner.events_tx.send(RoomIoEvent::TextInput(TextInput {
                        text,
                        participant: stream.sender,
                        info: Box::new(stream.info),
                    }));
                });
                this.keep_task(reader);
            }
        });
        self.keep_task(handle);
    }

    fn keep_task(&self, task: AbortHandle) {
        let coordinator = self.coordinator.borrow();
        if !coordinator.is_open() {
            task.abort();
            return;
        }
        let mut tasks = self.tasks.lock().unwrap();
        tasks.retain(|task| !task.is_finished());
        tasks.push(task);
    }

    fn coordinate<T>(&self, transition: impl FnOnce(&mut RoomCoordinator) -> T) -> T {
        let mut outcome = None;
        self.coordinator.send_modify(|coordinator| outcome = Some(transition(coordinator)));
        outcome.expect("watch applies each transition synchronously")
    }

    pub async fn aclose(self: &Arc<Self>) -> Closed {
        let closing = self.coordinate(RoomCoordinator::begin_close);
        if closing {
            // The owner survives cancellation of any caller waiting for cleanup.
            let this = self.clone();
            let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
            self.spawner.spawn_unboxed(async move {
                let outcome = this.close(deadline).await;
                this.coordinate(|coordinator| coordinator.finish_close(outcome));
            });
        }
        let mut state = self.coordinator.subscribe();
        let state = state
            .wait_for(|state| state.state().closed.is_some())
            .await
            .expect("RoomIo retains the cleanup result sender");
        state.state().closed.expect("cleanup matched a terminal outcome")
    }

    async fn close(&self, deadline: Ts) -> Closed {
        for h in self.tasks.lock().unwrap().drain(..) {
            h.abort();
        }
        if let Some(h) = self.state_task.lock().unwrap().take() {
            h.abort();
        }
        if *self.chat_registered.lock().unwrap() {
            *self.chat_registered.lock().unwrap() = false;
            self.room.unregister_text_stream_handler(TOPIC_CHAT);
        }
        let mut outcome = Closed::Clean;
        if let Some(pc) = &self.pre_connect {
            outcome =
                outcome.and(Closed::bounded(self.clock.as_ref(), deadline, pc.aclose()).await);
        }
        if let Some(input) = &self.audio_input {
            outcome =
                outcome.and(Closed::bounded(self.clock.as_ref(), deadline, input.aclose()).await);
        }
        if let Some(video) = &self.video_input {
            // `aclose` makes a waiting `next_frame` return `None`.
            outcome =
                outcome.and(Closed::bounded(self.clock.as_ref(), deadline, video.aclose()).await);
        }
        // Each owned performer keeps retiring after its waiter reaches the common deadline.
        if let Some(user_tr) = &self.user_tr {
            outcome =
                outcome.and(Closed::bounded(self.clock.as_ref(), deadline, user_tr.aclose()).await);
        }
        if let Some(agent_tr) = &self.agent_tr {
            outcome = outcome
                .and(Closed::bounded(self.clock.as_ref(), deadline, agent_tr.aclose()).await);
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

/// `RoomIO._on_participant_connected`. Not when someone is already linked, when an explicit target
/// is configured and this is not it, or when the candidate is an avatar publishing for the agent.
/// The publish-on-behalf filter applies only without an explicit target (the reference's `elif`).
pub fn decide_link(
    already_linked: bool,
    configured_target: Option<&ParticipantIdentity>,
    candidate: &ParticipantIdentity,
    candidate_kind: ParticipantKind,
    candidate_publishes_on_behalf_of: Option<&str>,
    agent_identity: &ParticipantIdentity,
    accepted_kinds: &[ParticipantKind],
) -> bool {
    let target_ok = match configured_target {
        Some(target) => target == candidate,
        None => candidate_publishes_on_behalf_of != Some(agent_identity.as_str()),
    };
    !already_linked && target_ok && accepted_kinds.contains(&candidate_kind)
}

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
pub fn publishes_on_behalf_of(
    attributes: &std::collections::HashMap<String, String>,
) -> Option<&str> {
    attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF).map(String::as_str)
}

/// `_ParticipantInputStream._on_track_available`: not the track already read (reopening would
/// restart the STT mid-utterance), nor a screenshare on the microphone leg.
pub fn decide_track(
    linked: Option<&ParticipantIdentity>,
    currently_reading: Option<&TrackSid>,
    owner: &ParticipantIdentity,
    source: TrackSource,
    sid: &TrackSid,
    accepted_sources: &[TrackSource],
) -> bool {
    linked == Some(owner) && accepted_sources.contains(&source) && currently_reading != Some(sid)
}

/// `_ParticipantLegacyTranscriptionOutput._is_local_proxy_participant`: publishing on behalf of the
/// agent is a reason to accept (the caption belongs on the avatar's track).
pub fn is_local_proxy_participant(
    subject: Option<&ParticipantIdentity>,
    candidate: &ParticipantIdentity,
    candidate_publishes_on_behalf_of: Option<&str>,
) -> bool {
    let Some(subject) = subject else { return false };
    subject == candidate || candidate_publishes_on_behalf_of == Some(subject.as_str())
}

/// `transcription/_utils.find_micro_track_id`; `None` instead of `ValueError`.
pub fn find_micro_track_id(room: &dyn RtcRoom, identity: &ParticipantIdentity) -> Option<TrackSid> {
    if *identity == room.local().identity() {
        return local_micro_track_id(room.local().as_ref());
    }
    room.participant(identity)?
        .track_publications()
        .into_iter()
        .find(|p| p.source() == TrackSource::Microphone)
        .map(|p| p.sid().clone())
}

/// The agent's own microphone publication, for `_on_local_track_published`.
pub fn local_micro_track_id(local: &dyn LocalParticipant) -> Option<TrackSid> {
    local
        .track_publications()
        .into_iter()
        .find(|p| p.source() == TrackSource::Microphone)
        .map(|p| p.sid().clone())
}

/// `RoomOptions`' two session-level switches and the room they apply to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomLinkConfig {
    pub close_on_disconnect: bool,
    pub delete_room_on_close: bool,
    pub room: String,
}

pub(crate) struct RoomLink {
    config: RoomLinkConfig,
    /// A later disconnect closes nothing, a second close deletes nothing.
    deletion_requested: bool,
}

impl RoomLink {
    pub(crate) fn new(config: Option<RoomLinkConfig>) -> Self {
        let config = config.unwrap_or(RoomLinkConfig {
            close_on_disconnect: true,
            delete_room_on_close: false,
            room: String::new(),
        });
        RoomLink { config, deletion_requested: false }
    }

    /// Does this disconnect end the session? A deletion in flight is the end of the call already
    /// (`not self._delete_room_task`).
    fn disconnect_closes(&self, reason: DisconnectReason) -> bool {
        !self.deletion_requested
            && self.config.close_on_disconnect
            && reason.closes_session_by_default()
    }

    /// `delete_room_on_close`, once: the room to delete.
    fn request_deletion(&mut self) -> Option<String> {
        if self.deletion_requested || !self.config.delete_room_on_close {
            return None;
        }
        self.deletion_requested = true;
        Some(self.config.room.clone())
    }
}

impl Driver {
    pub(crate) fn on_room_io(&mut self, now: Ts, ev: RoomIoEvent, out: &mut Out) {
        let closing = self.session.closing_past_edge();
        match &ev {
            RoomIoEvent::ParticipantUnlinked(unlinked) => {
                out.machine().append(self.session.states.on_link(now, false, closing));
                let reason = unlinked.reason;
                // A deletion in flight is the end of the call already (`not self._delete_room_task`).
                let closes = self.session.room.disconnect_closes(reason);
                out.trace(RoomStep::Link(RoomLinkStep::Unlinked { reason, closes }));
                if closes {
                    let reason = CloseReason::ParticipantDisconnected;
                    self.begin_closing(now, Close { reason, drain: false, error: None }, out);
                }
            }
            RoomIoEvent::ParticipantLinked(_) => {
                out.trace(RoomStep::Link(RoomLinkStep::Linked));
                out.machine().append(self.session.states.on_link(now, true, closing));
            }
            RoomIoEvent::KeyPressed => {
                out.machine().append(self.session.states.reset_away_timer(now, closing));
                return;
            }
            RoomIoEvent::TextInput(_) | RoomIoEvent::DeleteRoomRequested(_) => {}
        }
        out.machine().push(crate::agent_session::effects::Effect::Host(HostEvent::Room(ev)));
    }

    pub(crate) fn on_session_closed(&mut self) {
        if let Some(room) = self.session.room.request_deletion() {
            self.journal(Step::Room(RoomStep::Link(RoomLinkStep::DeleteRoomRequested)));
            self.outbox.push(crate::agent_session::effects::Effect::Host(HostEvent::Room(
                RoomIoEvent::DeleteRoomRequested(DeleteRoomRequested { room }),
            )));
        }
    }
}
