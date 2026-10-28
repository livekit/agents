//! A simulated room executor. Selection and readiness use the production room coordinator.

use super::{
    input::{ParticipantAudioInput, ParticipantVideoInput},
    output::RoomAudioOutput,
    pre_connect_audio::PreConnectAudioHandler,
    transcription::RoomTranscriptionOutput,
};
use lk_agents_abi::voice::io::{AudioInput, AudioOutput, IoError, Result, TextOutput};
use lk_agents_abi::voice::room_io::{
    ParticipantLinked, ParticipantUnlinked, RoomIoConfig, TextInput,
};
use lk_agents_abi::voice::room_owner::{self, RoomAudioInput, RoomIoEvent, RoomVideoInput};
use lk_agents_abi::voice::rtc::{ParticipantIdentity, RoomEvent, RtcRoom};
use lk_agents_abi::{AbortHandle, Clock, Closed, Monotone, Spawner, TaskSet, Ts};
use super::transcription::SegmentIds;
use lk_agents_voice::room_io::coordinator::RoomCoordinator;
use lk_agents_voice::room_io::{accepted_kinds, publishes_on_behalf_of};
use std::sync::{Arc, Mutex, Weak};
use tokio::sync::Notify;

type Listener = Arc<dyn Fn(RoomIoEvent) -> bool + Send + Sync>;
type RtcListener = Arc<dyn Fn(RoomEvent) -> bool + Send + Sync>;
pub struct RoomIo {
    weak: Weak<Self>,
    cleanup_outcome: Mutex<Closed>,
    chat_registered: Mutex<bool>,
    state_task: Mutex<Option<AbortHandle>>,
    room: Arc<dyn RtcRoom>,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    config: RoomIoConfig,
    coordinator: Mutex<RoomCoordinator>,
    changed: Notify,
    tasks: Arc<TaskSet>,
    closed: Monotone<bool>,
    listener: Mutex<Option<(Listener, RtcListener)>>,
    audio: Option<Arc<ParticipantAudioInput>>,
    video: Option<Arc<ParticipantVideoInput>>,
    output: Option<Arc<RoomAudioOutput>>,
    user_text: Option<Arc<RoomTranscriptionOutput>>,
    agent_text: Option<Arc<RoomTranscriptionOutput>>,
    pre_connect: Option<Arc<PreConnectAudioHandler>>,
}
impl RoomIo {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        config: RoomIoConfig,
    ) -> Arc<Self> {
        let ids = Arc::new(SegmentIds::default());
        let pre_connect = config
            .audio_input
            .as_ref()
            .filter(|a| a.pre_connect_audio)
            .map(|a| {
                PreConnectAudioHandler::new(
                    room.clone(),
                    spawner.clone(),
                    clock.clone(),
                    a.pre_connect_audio_timeout,
                    (a.sample_rate, a.num_channels),
                )
            })
            .transpose();
        let (pre_connect, startup_error) = match pre_connect {
            Ok(head) => (head, None),
            Err(error) => (None, Some(error)),
        };
        let audio = config.audio_input.as_ref().map(|a| {
            ParticipantAudioInput::new(
                room.clone(),
                clock.clone(),
                spawner.clone(),
                a.clone(),
                pre_connect.clone(),
            )
        });
        let video = config
            .video_input
            .then(|| ParticipantVideoInput::new(room.clone(), clock.clone(), spawner.clone()));
        let output = config
            .audio_output
            .as_ref()
            .map(|a| RoomAudioOutput::new(room.clone(), clock.clone(), spawner.clone(), a.clone()));
        let user_text = config.text_output.as_ref().map(|_| {
            RoomTranscriptionOutput::new(
                room.clone(),
                spawner.clone(),
                clock.clone(),
                ids.clone(),
                false,
                false,
                None,
            )
        });
        let agent_text = config.text_output.as_ref().map(|text| {
            RoomTranscriptionOutput::new(
                room.clone(),
                spawner.clone(),
                clock.clone(),
                ids,
                true,
                text.json_format,
                None,
            )
        });
        let mut coordinator = RoomCoordinator::new(config.participant_identity.clone());
        if let Some(error) = startup_error {
            coordinator.finish_start(Err(error));
        }
        Arc::new_cyclic(|weak| Self {
            weak: weak.clone(),
            cleanup_outcome: Mutex::new(Closed::Clean),
            chat_registered: Mutex::new(false),
            state_task: Mutex::new(None),
            coordinator: Mutex::new(coordinator),
            room,
            clock,
            spawner,
            config,
            changed: Notify::new(),
            tasks: Arc::default(),
            closed: Monotone::new(false),
            listener: Mutex::default(),
            audio,
            video,
            output,
            user_text,
            agent_text,
            pre_connect,
        })
    }
    fn emit(&self, event: RoomIoEvent) {
        let listener = self.listener.lock().unwrap().as_ref().map(|(l, _)| l.clone());
        if let Some(listener) = listener {
            listener(event);
        }
    }
    async fn attach(&self, who: Option<ParticipantIdentity>) -> Result<()> {
        if let Some(input) = &self.audio {
            input.set_participant(who.clone())?
        }
        if let Some(video) = &self.video {
            video.set_participant(who.clone())?
        }
        if let Some(text) = &self.user_text {
            text.set_participant(who).await?
        }
        Ok(())
    }
    async fn consider(
        &self,
        p: &Arc<dyn lk_agents_abi::voice::rtc::RemoteParticipant>,
    ) -> Result<()> {
        let attrs = p.attributes()?;
        let agent = self.room.local().identity()?;
        let selected = self.coordinator.lock().unwrap().consider(
            p.identity(),
            p.sid(),
            p.kind(),
            publishes_on_behalf_of(&attrs),
            &agent,
            accepted_kinds(&self.config),
        );
        if selected {
            self.emit(RoomIoEvent::ParticipantLinked(ParticipantLinked {
                identity: p.identity().clone(),
            }));
            self.attach(None).await?;
            self.attach(Some(p.identity().clone())).await?;
            self.changed.notify_waiters();
        }
        Ok(())
    }
    async fn event(&self, event: &RoomEvent) -> Result<()> {
        match event {
            RoomEvent::ParticipantConnected { participant }
            | RoomEvent::ParticipantAttributesChanged { participant } => {
                self.consider(participant).await?
            }
            RoomEvent::ParticipantDisconnected { participant, reason } => {
                if self
                    .coordinator
                    .lock()
                    .unwrap()
                    .disconnect(participant.identity(), participant.sid())
                {
                    self.attach(None).await?;
                    self.emit(RoomIoEvent::ParticipantUnlinked(ParticipantUnlinked {
                        identity: participant.identity().clone(),
                        reason: *reason,
                    }));
                }
            }
            _ => {}
        }
        let linked_event = match event {
            RoomEvent::TrackSubscribed { participant, .. }
            | RoomEvent::TrackUnsubscribed { participant, .. }
            | RoomEvent::TrackPublished { participant, .. }
            | RoomEvent::TrackUnpublished { participant, .. } => self
                .coordinator
                .lock()
                .unwrap()
                .accepts_participant(participant.identity(), participant.sid()),
            _ => true,
        };
        if linked_event {
            if let Some(input) = &self.audio {
                input.handle_event(event)?;
            }
            if let Some(video) = &self.video {
                video.handle_event(event)?;
            }
            if let Some(text) = &self.user_text {
                text.handle_event(event).await?;
            }
        }
        if let Some(text) = &self.agent_text {
            text.handle_event(event).await?
        }
        let listener = self.listener.lock().unwrap().as_ref().map(|(_, l)| l.clone());
        if let Some(listener) = listener {
            listener(event.clone());
        }
        Ok(())
    }
    fn fail(&self, error: IoError) {
        self.coordinator.lock().unwrap().finish_start(Err(error));
        self.record_cleanup(Closed::Failed);
        self.request_close();
    }
    async fn initialize(&self) -> Result<()> {
        if let Some(Err(error)) = self.coordinator.lock().unwrap().state().ready {
            return Err(error);
        }
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if !self.coordinator.lock().unwrap().is_open() {
                return Err(IoError::Rtc("room I/O closed during startup".into()));
            }
            if self.room.connection_state()?.is_connected() {
                break;
            }
            changed.await;
        }
        for p in self.room.remote_participants()? {
            self.consider(&p).await?
        }
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let state = self.coordinator.lock().unwrap().state();
            if state.linked.is_some() {
                break;
            }
            if let Some(result) = state.ready {
                return result;
            }
            changed.await;
        }
        if let Some(text) = &self.agent_text {
            text.set_participant(Some(self.room.local().identity()?)).await?
        }
        if let Some(output) = &self.output {
            output.start().await?
        }
        Ok(())
    }
    fn record_cleanup(&self, outcome: Closed) {
        let mut recorded = self.cleanup_outcome.lock().unwrap();
        *recorded = recorded.and(outcome);
    }

    fn request_close(&self) {
        if !self.coordinator.lock().unwrap().begin_close() {
            return;
        }
        self.changed.notify_waiters();
        let Some(owner) = self.weak.upgrade() else {
            return;
        };
        let deadline = self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        self.spawner.spawn(Box::pin(async move {
            let outcome = owner.cleanup(deadline).await;
            owner.coordinator.lock().unwrap().finish_close(outcome);
            owner.closed.set(true);
        }));
    }

    async fn cleanup(&self, deadline: Ts) -> Closed {
        let mut outcome = *self.cleanup_outcome.lock().unwrap();
        if let Some(input) = &self.audio {
            outcome = outcome.and(
                lk_agents_abi::timeout(
                    self.clock.as_ref(),
                    deadline.saturating_duration_since(self.clock.now()),
                    room_owner::RoomAudioInputPort::aclose(input.as_ref()),
                )
                .await
                .unwrap_or(Closed::TimedOut),
            );
        }
        if let Some(video) = &self.video {
            outcome = outcome.and(
                lk_agents_abi::timeout(
                    self.clock.as_ref(),
                    deadline.saturating_duration_since(self.clock.now()),
                    room_owner::RoomVideoInputPort::aclose(video.as_ref()),
                )
                .await
                .unwrap_or(Closed::TimedOut),
            );
        }
        if let Some(head) = &self.pre_connect {
            outcome = outcome.and(
                lk_agents_abi::timeout(
                    self.clock.as_ref(),
                    deadline.saturating_duration_since(self.clock.now()),
                    head.aclose(),
                )
                .await
                .unwrap_or(Closed::TimedOut),
            );
            if self
                .room
                .unregister_byte_stream_handler(
                    lk_agents_voice::room_io::PRE_CONNECT_AUDIO_BUFFER_STREAM,
                )
                .is_err()
            {
                outcome = outcome.and(Closed::Failed);
            }
        }
        for text in [&self.user_text, &self.agent_text].into_iter().flatten() {
            match lk_agents_abi::timeout(
                self.clock.as_ref(),
                deadline.saturating_duration_since(self.clock.now()),
                text.aclose(),
            )
            .await
            {
                Ok(Ok(())) => {}
                Ok(Err(_)) => outcome = outcome.and(Closed::Failed),
                Err(_) => outcome = outcome.and(Closed::TimedOut),
            }
        }
        if let Some(output) = &self.output {
            outcome = outcome.and(
                lk_agents_abi::timeout(
                    self.clock.as_ref(),
                    deadline.saturating_duration_since(self.clock.now()),
                    output.aclose(),
                )
                .await
                .unwrap_or(Closed::TimedOut),
            );
        }
        if !self.tasks.wait_empty(self.clock.as_ref(), Some(deadline)).await {
            outcome = outcome.and(Closed::TimedOut);
        }
        outcome.and(*self.cleanup_outcome.lock().unwrap())
    }

    async fn read_chat(&self, attachment: u64, mut stream: lk_agents_abi::voice::rtc::TextStream) {
        let sender = stream.sender.clone();
        let result = {
            let read = crate::rtc::read_text(&mut stream.chunks);
            tokio::pin!(read);
            loop {
                let changed = self.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if !self.coordinator.lock().unwrap().accepts(attachment, &sender) {
                    break None;
                }
                tokio::select! {
                    biased;
                    () = &mut changed => {},
                    text = &mut read => break Some(text),
                }
            }
        };
        let sender = stream.sender.clone();
        let info = stream.info.clone();
        let cleanup = stream
            .close(self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
            .await;
        self.record_cleanup(cleanup);
        match result {
            Some(Ok(text))
                if cleanup == Closed::Clean
                    && self.coordinator.lock().unwrap().accepts(attachment, &sender) =>
            {
                self.emit(RoomIoEvent::TextInput(TextInput {
                    text,
                    participant: sender,
                    info: Box::new(info),
                }))
            }
            Some(Err(error)) => self.fail(error.into()),
            _ => {}
        }
    }
}
impl room_owner::RoomIo for RoomIo {
    fn forward_events(self: Arc<Self>, listener: Listener, rtc_listener: RtcListener) {
        *self.listener.lock().unwrap() = Some((listener, rtc_listener));
    }
    fn audio_input(&self) -> Option<Arc<dyn AudioInput>> {
        self.audio.clone().map(|a| a as Arc<dyn AudioInput>)
    }
    fn audio_input_leg(&self) -> Option<Arc<RoomAudioInput>> {
        self.audio.clone().map(|a| Arc::new(RoomAudioInput(a)))
    }
    fn video_input(&self) -> Option<Arc<RoomVideoInput>> {
        self.video.clone().map(|v| Arc::new(RoomVideoInput(v)))
    }
    fn audio_output(&self) -> Option<Arc<dyn AudioOutput>> {
        self.output.clone().map(|a| a as Arc<dyn AudioOutput>)
    }
    fn transcription_output(&self) -> Option<Arc<dyn TextOutput>> {
        self.agent_text.clone().map(|t| t as Arc<dyn TextOutput>)
    }
    fn wants_transcript_sync(&self, external_audio: bool) -> bool {
        self.config.text_output.as_ref().is_some_and(|t| {
            t.sync_transcription.unwrap_or(self.output.is_some() || external_audio)
        })
    }
    fn linked_participant(&self) -> Option<ParticipantIdentity> {
        self.coordinator.lock().unwrap().state().linked
    }
    fn start(self: Arc<Self>) {
        if !self.coordinator.lock().unwrap().start() {
            return;
        }
        let mut events = self.room.events();
        let owner = self.clone();
        self.tasks.spawn(self.spawner.as_ref(), "simulated room events", async move {
            loop {
                let changed = owner.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if !owner.coordinator.lock().unwrap().is_open() {
                    break;
                }
                let event = tokio::select! {event=events.next()=>event,()=&mut changed=>continue};
                match event {
                    Ok(Some(event)) => {
                        if let Err(error) = owner.event(&event).await {
                            owner.fail(error);
                            break;
                        }
                        owner.changed.notify_waiters();
                    }
                    Ok(None) => {
                        owner.fail(IoError::Rtc("room event source ended".into()));
                        break;
                    }
                    Err(error) => {
                        owner.fail(error.into());
                        break;
                    }
                }
            }
            owner.record_cleanup(
                events
                    .close(owner.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
                    .await,
            );
        });
        let owner = self.clone();
        self.tasks.spawn(self.spawner.as_ref(), "simulated room startup", async move {
            match owner.initialize().await {
                Ok(()) => {
                    owner.coordinator.lock().unwrap().finish_start(Ok(()));
                    owner.changed.notify_waiters();
                }
                Err(error) => {
                    if owner.coordinator.lock().unwrap().is_open() {
                        owner.fail(error);
                    }
                }
            }
        });
        if self.config.text_input {
            match self.room.text_streams(lk_agents_abi::TOPIC_CHAT) {
                Ok(mut streams) => {
                    *self.chat_registered.lock().unwrap() = true;
                    let owner = self.clone();
                    self.tasks.spawn(self.spawner.as_ref(), "simulated room chat", async move {
                        loop {
                            let changed = owner.changed.notified();
                            tokio::pin!(changed);
                            changed.as_mut().enable();
                            if !owner.coordinator.lock().unwrap().is_open() {
                                break;
                            }
                            let stream = tokio::select! {
                                biased;
                                () = &mut changed => continue,
                                stream = streams.next() => stream,
                            };
                            match stream {
                                Ok(Some(stream)) => {
                                    let attachment = owner
                                        .coordinator
                                        .lock()
                                        .unwrap()
                                        .attachment(&stream.sender);
                                    if let Some(attachment) = attachment {
                                        let reader = owner.clone();
                                        owner.tasks.spawn(
                                            owner.spawner.as_ref(),
                                            "simulated room chat message",
                                            async move {
                                                reader.read_chat(attachment, stream).await;
                                            },
                                        );
                                    } else {
                                        owner.record_cleanup(
                                            stream
                                                .close(owner.clock.now().saturating_add(
                                                    lk_agents_abi::HOST_TEARDOWN_TIMEOUT,
                                                ))
                                                .await,
                                        );
                                    }
                                }
                                Ok(None) => break,
                                Err(error) => {
                                    owner.fail(error.into());
                                    break;
                                }
                            }
                        }
                        owner.record_cleanup(
                            streams
                                .close(
                                    owner
                                        .clock
                                        .now()
                                        .saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT),
                                )
                                .await,
                        );
                        let registered =
                            std::mem::take(&mut *owner.chat_registered.lock().unwrap());
                        if registered
                            && owner
                                .room
                                .unregister_text_stream_handler(lk_agents_abi::TOPIC_CHAT)
                                .is_err()
                        {
                            owner.record_cleanup(Closed::Failed);
                        }
                    });
                }
                Err(error) => self.fail(error.into()),
            }
        }
    }
    fn on_agent_state_changed(self: Arc<Self>, state: &str) {
        let local = self.room.local().clone();
        let attrs = std::collections::BTreeMap::from([(
            lk_agents_abi::ATTRIBUTE_AGENT_STATE.into(),
            state.into(),
        )]);
        if !self.coordinator.lock().unwrap().is_open() {
            return;
        }
        let mut pending = self.state_task.lock().unwrap();
        if let Some(old) = pending.take() {
            old.abort();
        }
        *pending = Some(self.tasks.spawn(
            self.spawner.as_ref(),
            "simulated agent attributes",
            async move {
                let _ = local.set_attributes(attrs).await;
            },
        ));
    }
    fn wait_for_ready(&self) -> lk_agents_abi::BoxFuture<'_, Result<()>> {
        Box::pin(async {
            loop {
                let changed = self.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if let Some(result) = self.coordinator.lock().unwrap().state().ready {
                    return result;
                }
                changed.await
            }
        })
    }
    fn set_participant(
        &self,
        who: Option<ParticipantIdentity>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let sid = who
                .as_ref()
                .map(|who| self.room.participant(who))
                .transpose()?
                .flatten()
                .map(|p| p.sid().to_owned());
            if self.coordinator.lock().unwrap().select(who.clone(), sid.as_deref()) {
                self.attach(None).await?;
                self.attach(who).await?;
                self.changed.notify_waiters();
            }
            Ok(())
        })
    }
    fn unset_participant(&self) -> lk_agents_abi::BoxFuture<'_, Result<()>> {
        self.set_participant(None)
    }
    fn on_user_input_transcribed<'a>(
        &'a self,
        text: &'a str,
        is_final: bool,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if let Some(output) = &self.user_text {
                output.capture_text(text).await?;
                if is_final {
                    output.flush();
                }
            }
            Ok(())
        })
    }
    fn aclose(self: Arc<Self>) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.request_close();
            self.closed.wait_for(|closed| *closed).await;
            self.coordinator.lock().unwrap().state().closed.expect("cleanup settled")
        })
    }
}
