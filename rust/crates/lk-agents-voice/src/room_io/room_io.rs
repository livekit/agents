use crate::agent_session::effects::StreamKind;
use crate::io::{
    AudioInput as IoRoom_IoAudioInput, AudioOutput as IoRoom_IoAudioOutput, Result, TextOutput,
};
use crate::room_io::_input::{ParticipantAudioInput, ParticipantVideoInput};
use crate::room_io::_output::ParticipantAudioOutput;
use crate::room_io::_output::transcription::{CountingSegmentIds, ParticipantTranscriptionOutput};
use crate::room_io::_pre_connect_audio::PreConnectAudioHandler;
use crate::room_io::types::RoomIoConfig;
use lk_agents_abi::AbortHandle;
pub use lk_agents_abi::voice::room_io::{
    DeleteRoomRequested, ParticipantLinked, ParticipantUnlinked, TextInput,
};
use lk_agents_abi::{
    ATTRIBUTE_AGENT_STATE, ATTRIBUTE_PUBLISH_ON_BEHALF, Clock, Monotone, Spawner, SpawnerExt,
    TOPIC_CHAT,
};
use lk_agents_rtc::{
    ConnectionState, DisconnectReason, LocalParticipant, ParticipantIdentity, ParticipantKind,
    RoomEvent, RtcRoom, TrackSid, TrackSource,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, watch};

/// `RecorderIO._write_task`'s `WRITE_INTERVAL`, on the session clock.
pub const RECORDER_FLUSH: Duration = Duration::from_millis(2500);

/// A provider parked between `Effect::Provider(OpenLeg)` and the performer's spawn.
pub(crate) enum PendingOpen {
    Stt(Arc<dyn lk_agents_abi::Stt>),
    Vad(Arc<dyn lk_agents_abi::Vad>),
}

pub struct StreamOpened {
    pub open: u64,
    pub leg: StreamKind,
    pub outcome: std::result::Result<Opened, lk_agents_abi::errors::ProviderError>,
}

pub enum Opened {
    Stt(lk_agents_abi::SttSink, lk_agents_abi::SttSource),
    Vad(lk_agents_abi::VadSink, lk_agents_abi::VadSource),
}

/// The legs a frame has after `reopen_legs`: given, or the table's row where none was given.
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
}

pub struct RoomIo {
    room: Arc<dyn RtcRoom>,
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

    target: Mutex<Option<ParticipantIdentity>>,
    /// `_participant_available_fut`. Always `send_replace`: `send` fails with no receiver, which
    /// would drop a link made before `_init_task` subscribes.
    linked: watch::Sender<Option<ParticipantIdentity>>,
    connected: Monotone<bool>,
    ready: Monotone<bool>,

    events_tx: mpsc::UnboundedSender<RoomIoEvent>,
    events_rx: Mutex<Option<mpsc::UnboundedReceiver<RoomIoEvent>>>,

    tasks: Mutex<Vec<AbortHandle>>,
    /// The in-flight `set_attributes`, cancelled before the next as in the reference.
    state_task: Mutex<Option<AbortHandle>>,
    chat_registered: Mutex<bool>,
}

impl RoomIo {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        ids: Arc<CountingSegmentIds>,
        config: RoomIoConfig,
        text_next_in_chain: Option<Arc<dyn TextOutput>>,
    ) -> Arc<Self> {
        let pre_connect = config.audio_input.as_ref().filter(|a| a.pre_connect_audio).map(|a| {
            PreConnectAudioHandler::new(
                room.clone(),
                spawner.clone(),
                clock.clone(),
                a.pre_connect_audio_timeout,
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
        Arc::new(RoomIo {
            target: Mutex::new(config.participant_identity.clone()),
            room,
            spawner,
            config,
            audio_input,
            video_input,
            audio_output,
            user_tr,
            agent_tr,
            pre_connect,
            linked: watch::channel(None).0,
            connected: Monotone::new(false),
            ready: Monotone::new(false),
            events_tx,
            events_rx: Mutex::new(Some(events_rx)),
            tasks: Mutex::new(Vec::new()),
            state_task: Mutex::new(None),
            chat_registered: Mutex::new(false),
        })
    }

    pub fn events(&self) -> Option<mpsc::UnboundedReceiver<RoomIoEvent>> {
        self.events_rx.lock().unwrap().take()
    }

    pub fn audio_input(&self) -> Option<Arc<dyn IoRoom_IoAudioInput>> {
        self.audio_input.clone().map(|a| a as Arc<dyn IoRoom_IoAudioInput>)
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

    pub fn agent_transcription(&self) -> Option<Arc<ParticipantTranscriptionOutput>> {
        self.agent_tr.clone()
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
        self.linked.borrow().clone()
    }

    /// `wait_for_ready`.
    pub async fn wait_for_ready(&self) {
        self.ready.wait_for(|set| *set).await;
    }

    pub fn start(self: &Arc<Self>) {
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
            }
        });
        self.tasks.lock().unwrap().push(handle);

        // The room may already be connected, so read the state rather than wait for the event.
        if self.room.connection_state().is_connected() {
            self.connected.set(true);
        }

        let this = self.clone();
        let handle = self.spawner.spawn_unboxed(async move { this.init().await });
        self.tasks.lock().unwrap().push(handle);
    }

    /// `_init_task`.
    async fn init(self: Arc<Self>) {
        self.connected.wait_for(|set| *set).await;

        // Replay participants who joined before the handler was registered.
        for p in self.room.remote_participants() {
            let attrs = p.attributes();
            self.consider(p.identity(), p.kind(), publishes_on_behalf_of(&attrs));
        }

        let identity = {
            let mut rx = self.linked.subscribe();
            let v = rx.wait_for(|v| v.is_some()).await;
            match v {
                Ok(v) => v.clone().expect("wait_for matched Some"),
                Err(_) => return,
            }
        };
        self.set_participant(Some(identity)).await;

        if let Some(agent_tr) = &self.agent_tr {
            // The agent transcribes *itself*, so its transcript hangs off its own track.
            agent_tr.set_participant(Some(self.room.local().identity())).await;
        }
        if let Some(out) = &self.audio_output {
            let _ = out.start().await;
        }
        self.ready.set(true);
    }

    /// `set_participant`.
    pub async fn set_participant(&self, identity: Option<ParticipantIdentity>) {
        let Some(identity) = identity else {
            self.unset_participant().await;
            return;
        };

        let previous = self.target.lock().unwrap().clone();
        if previous.as_ref().is_some_and(|p| *p != identity) {
            // Switching: forget the old link, then re-check whether the new one is already here.
            self.linked.send_replace(None);
            if self.room.participant(&identity).is_some() {
                self.linked.send_replace(Some(identity.clone()));
            }
        }
        *self.target.lock().unwrap() = Some(identity.clone());

        if let Some(input) = &self.audio_input {
            input.set_participant(Some(identity.clone()));
        }
        if let Some(video) = &self.video_input {
            // Same participant as the audio leg, from the same call.
            video.set_participant(Some(identity.clone()));
        }
        if let Some(user_tr) = &self.user_tr {
            user_tr.set_participant(Some(identity)).await;
        }
    }

    pub async fn unset_participant(&self) {
        *self.target.lock().unwrap() = None;
        self.linked.send_replace(None);
        if let Some(input) = &self.audio_input {
            input.set_participant(None);
        }
        if let Some(video) = &self.video_input {
            // Stop reading the camera of someone who left.
            video.set_participant(None);
        }
        if let Some(user_tr) = &self.user_tr {
            user_tr.set_participant(None).await;
        }
    }

    /// `_on_participant_connected`'s filter.
    fn consider(
        &self,
        candidate: &ParticipantIdentity,
        kind: lk_agents_rtc::ParticipantKind,
        on_behalf_of: Option<&str>,
    ) {
        let target = self.target.lock().unwrap().clone();
        let link = decide_link(
            self.linked.borrow().is_some(),
            target.as_ref(),
            candidate,
            kind,
            on_behalf_of,
            &self.room.local().identity(),
            crate::room_io::types::accepted_kinds(&self.config),
        );
        if link {
            self.linked.send_replace(Some(candidate.clone()));
            let _ = self.events_tx.send(RoomIoEvent::ParticipantLinked(ParticipantLinked {
                identity: candidate.clone(),
            }));
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
                );
            }
            RoomEvent::ParticipantDisconnected { participant, reason } => {
                self.on_participant_disconnected(participant.identity(), *reason);
            }
            RoomEvent::TrackSubscribed { .. }
            | RoomEvent::TrackPublished { .. }
            | RoomEvent::TrackUnpublished { .. }
            | RoomEvent::LocalTrackPublished { .. }
            | RoomEvent::Close { .. } => {}
        }
    }

    fn on_participant_disconnected(
        &self,
        identity: &ParticipantIdentity,
        reason: DisconnectReason,
    ) {
        if self.linked.borrow().as_ref() != Some(identity) {
            return;
        }
        // Whether the departure closes the session is `RoomLink`'s.
        self.linked.send_replace(None);
        let _ = self.events_tx.send(RoomIoEvent::ParticipantUnlinked(ParticipantUnlinked {
            identity: identity.clone(),
            reason,
        }));
    }

    /// `_on_agent_state_changed`: cancels the in-flight write so changes land in order.
    pub fn on_agent_state_changed(self: &Arc<Self>, state: &str) {
        let room = self.room.clone();
        let state = state.to_owned();
        let handle = self.spawner.spawn_unboxed(async move {
            if !room.connection_state().is_connected() {
                return;
            }
            let _ = room
                .local()
                .set_attributes(HashMap::from([(ATTRIBUTE_AGENT_STATE.to_owned(), state)]))
                .await;
        });
        if let Some(old) = self.state_task.lock().unwrap().replace(handle) {
            old.abort();
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
        *self.chat_registered.lock().unwrap() = true;

        let this = self.clone();
        let handle = self.spawner.spawn_unboxed(async move {
            while let Some(mut stream) = streams.recv().await {
                // Only the linked participant may drive the agent.
                if let Some(linked) = this.linked.borrow().clone()
                    && stream.sender != linked
                {
                    continue;
                }
                if this.room.participant(&stream.sender).is_none() {
                    continue;
                }
                let sender = stream.sender.clone();
                let info = stream.info.clone();
                let text = stream.read_all().await;
                let _ = this.events_tx.send(RoomIoEvent::TextInput(TextInput {
                    text,
                    participant: sender,
                    info: Box::new(info),
                }));
            }
        });
        self.tasks.lock().unwrap().push(handle);
    }

    pub async fn aclose(&self) {
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
        if let Some(pc) = &self.pre_connect {
            pc.aclose().await;
        }
        if let Some(input) = &self.audio_input {
            input.aclose().await;
        }
        if let Some(video) = &self.video_input {
            // `aclose` makes a waiting `next_frame` return `None`.
            video.aclose().await;
        }
        // Transcripts first: the final header must go out while the room is usable.
        if let Some(user_tr) = &self.user_tr {
            user_tr.aclose().await;
        }
        if let Some(agent_tr) = &self.agent_tr {
            agent_tr.aclose().await;
        }
        if let Some(out) = &self.audio_output {
            out.aclose().await;
        }
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
