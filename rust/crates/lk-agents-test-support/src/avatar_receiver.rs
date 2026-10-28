//! Independent supplied-clock avatar peer. It feeds the production receiver state, not the
//! runtime's stream tasks, and races one active stream directly against incoming headers.

use lk_agents_abi::voice::io::AvatarPlaybackSegment;
use lk_agents_abi::voice::rtc::{ParticipantIdentity, ParticipantKind, RpcHandler, RtcRoom};
use lk_agents_abi::{AbortHandle, Clock, Spawner, SpawnerExt};
use lk_agents_voice::avatar::receiver::{Receipt, ReceiverState};
use lk_agents_voice::avatar::{
    AUDIO_STREAM_TOPIC, RPC_CLEAR_BUFFER, RPC_PLAYBACK_FINISHED, RPC_PLAYBACK_STARTED, Received,
    ReceiverEnd, STREAM_IDLE_TIMEOUT, playback_method,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub struct DataStreamAudioReceiver {
    room: Arc<dyn RtcRoom>,
    sender: Option<String>,
    frame_ms: u32,
    retries: u32,
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    clear: Option<Arc<dyn Fn() + Send + Sync>>,
    state: Arc<Mutex<ReceiverState>>,
    wake: Arc<tokio::sync::Notify>,
    tasks: Mutex<Vec<AbortHandle>>,
    reports: tokio::sync::mpsc::UnboundedSender<(&'static str, AvatarPlaybackSegment, String)>,
    report_rx: Mutex<
        Option<tokio::sync::mpsc::UnboundedReceiver<(&'static str, AvatarPlaybackSegment, String)>>,
    >,
}

impl DataStreamAudioReceiver {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        sender: Option<String>,
        frame_ms: u32,
        retries: u32,
        clear: Option<Arc<dyn Fn() + Send + Sync>>,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        let (reports, report_rx) = tokio::sync::mpsc::unbounded_channel();
        Self {
            room,
            sender,
            frame_ms,
            retries,
            clock,
            spawner,
            clear,
            state: Arc::new(Mutex::new(ReceiverState::default())),
            wake: Arc::new(tokio::sync::Notify::new()),
            tasks: Mutex::new(Vec::new()),
            reports,
            report_rx: Mutex::new(Some(report_rx)),
        }
    }
    pub async fn start(&self) -> Result<(), ReceiverEnd> {
        let remote = self
            .sender
            .clone()
            .or_else(|| {
                self.room
                    .remote_participants()
                    .ok()?
                    .into_iter()
                    .find(|p| p.kind() == ParticipantKind::Agent)
                    .map(|p| p.identity().0.clone())
            })
            .ok_or_else(|| ReceiverEnd::Failed("fake avatar peer has no agent".into()))?;

        let (state, wake, clear, expected) =
            (self.state.clone(), self.wake.clone(), self.clear.clone(), remote.clone());
        self.room
            .local()
            .register_rpc_method(
                RPC_CLEAR_BUFFER.into(),
                RpcHandler::new(move |caller, _| {
                    if caller != expected {
                        return "reject".into();
                    }
                    state.lock().unwrap().clear();
                    wake.notify_waiters();
                    if let Some(clear) = &clear {
                        clear();
                    }
                    "ok".into()
                }),
            )
            .map_err(|e| ReceiverEnd::Failed(e.to_string()))?;
        let mut streams = self
            .room
            .byte_streams(AUDIO_STREAM_TOPIC)
            .map_err(|e| ReceiverEnd::Failed(e.to_string()))?;
        let mut report_rx =
            self.report_rx.lock().unwrap().take().expect("fake receiver starts once");
        let (room, report_state, report_wake, report_clock, retries, report_remote) = (
            self.room.clone(),
            self.state.clone(),
            self.wake.clone(),
            self.clock.clone(),
            self.retries,
            remote.clone(),
        );
        let reports = self.spawner.spawn_unboxed(async move {
            while let Some((base, segment, payload)) = report_rx.recv().await {
                let method = match playback_method(base, &segment) {
                    Ok(method) => method,
                    Err(error) => {
                        report_state.lock().unwrap().end(error);
                        report_wake.notify_waiters();
                        return;
                    }
                };
                let mut failure = None;
                for attempt in 0..retries {
                    match room
                        .local()
                        .perform_rpc(
                            ParticipantIdentity(report_remote.clone()),
                            method.clone(),
                            payload.clone(),
                        )
                        .await
                    {
                        Ok((0, reply)) if !reply.starts_with("reject:") => {
                            failure = None;
                            break;
                        }
                        Ok((1400, _)) if base == RPC_PLAYBACK_STARTED => {
                            failure = None;
                            break;
                        }
                        Ok((code, message)) => {
                            failure =
                                Some(ReceiverEnd::Failed(format!("RpcError {code}: {message}")))
                        }
                        Err(e) => {
                            failure = Some(ReceiverEnd::Failed(e.to_string()));
                            break;
                        }
                    }
                    if attempt + 1 < retries {
                        report_clock
                            .wait_until(
                                report_clock.now().saturating_add(Duration::from_millis(100)),
                            )
                            .await;
                    }
                }
                if let Some(error) = failure {
                    report_state.lock().unwrap().end(error);
                    report_wake.notify_waiters();
                    return;
                }
            }
        });
        self.tasks.lock().unwrap().push(reports);
        let (state, wake, clock, frame_ms) =
            (self.state.clone(), self.wake.clone(), self.clock.clone(), self.frame_ms);
        let task = self.spawner.spawn_unboxed(async move {
            let result: Result<(), ReceiverEnd> = async {
                let mut pending = VecDeque::new();
                loop {
                    let mut stream = match pending.pop_front() {
                        Some(stream) => stream,
                        None => match streams.next().await.map_err(|e| ReceiverEnd::Failed(e.to_string()))? {
                            Some(stream) => stream, None => return Ok(()),
                        },
                    };
                    if stream.sender.0 != remote {
                        let _ = stream.close(clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)).await;
                        continue;
                    }
                    let mut reader = state.lock().unwrap().begin(&stream.info, frame_ms)?;
                    loop {
                        let cleared = wake.notified();
                        tokio::pin!(cleared);
                        cleared.as_mut().enable();
                        if state.lock().unwrap().superseded() { break; }
                        let deadline = clock.now().saturating_add(STREAM_IDLE_TIMEOUT);
                        tokio::select! {
                            biased;
                            data = stream.chunks.next() => match data.map_err(|e| ReceiverEnd::BadStream(e.to_string()))? {
                                Some(data) => { let keep = state.lock().unwrap().chunk(&mut reader, &data); wake.notify_waiters(); if !keep { break; } }
                                None => break,
                            },
                            header = streams.next() => {
                                if let Some(header) = header.map_err(|e| ReceiverEnd::BadStream(e.to_string()))? {
                                    if header.sender.0 == remote { state.lock().unwrap().stream_admitted(); }
                                    pending.push_back(header);
                                } else { break; }
                            },
                            () = &mut cleared => { if state.lock().unwrap().superseded() { break; } },
                            () = clock.wait_until(deadline) => break,
                        }
                    }
                    state.lock().unwrap().flush(&mut reader);
                    let outcome = stream.close(clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)).await;
                    if outcome != lk_agents_abi::Closed::Clean { return Err(ReceiverEnd::Failed(format!("fake peer stream close: {outcome:?}"))); }
                    state.lock().unwrap().finish(Some(&mut reader));
                    wake.notify_waiters();
                }
            }.await;
            state.lock().unwrap().end(result.err().unwrap_or(ReceiverEnd::Closed));
            wake.notify_waiters();
        });
        self.tasks.lock().unwrap().push(task);
        Ok(())
    }
    pub async fn recv(&self) -> Result<Receipt, ReceiverEnd> {
        loop {
            let wake = self.wake.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            if let Some(item) = self.state.lock().unwrap().take()? {
                return Ok(item);
            }
            wake.await;
        }
    }
    fn report(
        &self,
        base: &'static str,
        segment: AvatarPlaybackSegment,
        payload: String,
    ) -> Result<(), ReceiverEnd> {
        if self.state.lock().unwrap().ended() {
            return Err(ReceiverEnd::Closed);
        }
        playback_method(base, &segment)?;
        self.reports.send((base, segment, payload)).map_err(|_| ReceiverEnd::Closed)?;
        Ok(())
    }
    pub fn notify_playback_started(
        &self,
        segment: AvatarPlaybackSegment,
    ) -> Result<(), ReceiverEnd> {
        self.report(RPC_PLAYBACK_STARTED, segment, String::new())
    }
    pub fn notify_playback_finished(
        &self,
        segment: AvatarPlaybackSegment,
        playback_position: f64,
        interrupted: bool,
    ) -> Result<(), ReceiverEnd> {
        self.report(RPC_PLAYBACK_FINISHED, segment, serde_json::json!({"playback_position": playback_position, "interrupted": interrupted, "synchronized_transcript": null}).to_string())
    }
    pub fn aclose(&self) {
        self.state.lock().unwrap().end(ReceiverEnd::Closed);
        self.wake.notify_waiters();
        for task in self.tasks.lock().unwrap().drain(..) {
            task.abort();
        }
    }
}
impl Drop for DataStreamAudioReceiver {
    fn drop(&mut self) {
        self.aclose();
    }
}
