//! Simulated buffered-head transport using production transfer admission and PCM normalization.

use lk_agents_abi::voice::io::{IoError, Result};
use lk_agents_abi::voice::rtc::{ByteStream, RtcRoom, TrackSid};
use lk_agents_abi::{AudioFrame, Clock, Closed, Spawner, TaskSet};
use lk_agents_voice::room_io::PRE_CONNECT_AUDIO_BUFFER_STREAM;
use lk_agents_voice::room_io::pre_connect_state::{
    HeadBytes, PreConnectBuffer, PreConnectState, replayed,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

pub struct PreConnectAudioHandler {
    state: Mutex<PreConnectState>,
    clock: Arc<dyn Clock>,
    format: (u32, u16),
    timeout: Duration,
    changed: Notify,
    tasks: Arc<TaskSet>,
    closed: Mutex<bool>,
    stopped: Notify,
    cleanup: Mutex<Option<Closed>>,
    outcome: Mutex<Closed>,
}
impl PreConnectAudioHandler {
    pub fn new(
        room: Arc<dyn RtcRoom>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
        timeout: Duration,
        format: (u32, u16),
    ) -> Result<Arc<Self>> {
        let mut streams = room.byte_streams(PRE_CONNECT_AUDIO_BUFFER_STREAM)?;
        let handler = Arc::new(Self {
            state: Mutex::default(),
            clock,
            format,
            timeout,
            changed: Notify::new(),
            tasks: Arc::default(),
            closed: Mutex::new(false),
            stopped: Notify::new(),
            cleanup: Mutex::new(None),
            outcome: Mutex::new(Closed::Clean),
        });
        let owner = handler.clone();
        let executor = spawner.clone();
        handler.tasks.spawn(spawner.as_ref(), "simulated buffered-head ingress", async move {
            loop {
                let stopped = owner.stopped.notified();
                tokio::pin!(stopped);
                stopped.as_mut().enable();
                if *owner.closed.lock().unwrap() {
                    break;
                }
                let stream = tokio::select! {stream=streams.next()=>stream,()=&mut stopped=>break};
                let Ok(Some(stream)) = stream else { break };
                let Some(track) =
                    stream.info.attributes.get("trackId").map(|id| TrackSid(id.clone()))
                else {
                    owner.record_cleanup(
                        stream
                            .close(
                                owner
                                    .clock
                                    .now()
                                    .saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT),
                            )
                            .await,
                    );
                    continue;
                };
                let slot =
                    owner.state.lock().unwrap().receive(track.clone(), stream.info.id.clone());
                if let Ok(Some(id)) = slot {
                    let task_owner = owner.clone();
                    owner.tasks.spawn(executor.as_ref(), "simulated buffered head", async move {
                        task_owner.read_stream(track, id, stream).await;
                    });
                } else {
                    owner.record_cleanup(
                        stream
                            .close(
                                owner
                                    .clock
                                    .now()
                                    .saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT),
                            )
                            .await,
                    );
                }
            }
            owner.record_cleanup(
                streams
                    .close(owner.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
                    .await,
            );
        });
        Ok(handler)
    }
    async fn read_stream(&self, track: TrackSid, id: u64, mut stream: ByteStream) {
        let stopped = self.stopped.notified();
        tokio::pin!(stopped);
        stopped.as_mut().enable();
        let outcome = if *self.closed.lock().unwrap() {
            Err("pre-connect input is closed".into())
        } else {
            tokio::select! {
                biased;
                () = &mut stopped => Err("pre-connect input is closed".into()),
                received = lk_agents_abi::timeout(self.clock.as_ref(), self.timeout, self.receive(&mut stream)) =>
                    received.unwrap_or_else(|_| Err("pre-connect receive deadline expired".into())),
            }
        };
        let cleanup = stream
            .close(self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
            .await;
        self.record_cleanup(cleanup);
        let outcome = if cleanup == Closed::Clean {
            outcome
        } else {
            Err("pre-connect source cleanup failed".into())
        };
        self.state.lock().unwrap().complete(
            &track,
            id,
            outcome.map(|frames| PreConnectBuffer { received_at: self.clock.now(), frames }),
        );
        self.changed.notify_waiters();
    }

    /// The simulation decodes the admitted head inline; production runs the same decode on its
    /// media pool.
    async fn receive(
        &self,
        stream: &mut ByteStream,
    ) -> std::result::Result<Vec<AudioFrame>, String> {
        let mut head = HeadBytes::new(&stream.info)?;
        while let Some(chunk) = stream.chunks.next().await.map_err(|e| e.to_string())? {
            head.push(&chunk)?;
        }
        head.decode(self.format)
    }
    pub async fn wait_for_data(&self, track: &TrackSid) -> Result<Vec<AudioFrame>> {
        let id = self.state.lock().unwrap().slot(track)?;
        let waiting = async {
            loop {
                let changed = self.changed.notified();
                tokio::pin!(changed);
                changed.as_mut().enable();
                if let Some(outcome) = self.state.lock().unwrap().take(id) {
                    return outcome;
                }
                changed.await;
            }
        };
        let outcome = lk_agents_abi::timeout(self.clock.as_ref(), self.timeout, waiting).await;
        match outcome {
            Ok(Ok(buffer)) => Ok(replayed(buffer, self.clock.now())),
            Ok(Err(error)) => Err(IoError::PreConnectMalformed(error)),
            Err(_) => {
                self.state.lock().unwrap().expire(id);
                Err(IoError::PreConnectTimeout(track.clone()))
            }
        }
    }
    fn record_cleanup(&self, outcome: Closed) {
        let mut recorded = self.outcome.lock().unwrap();
        *recorded = recorded.and(outcome);
    }

    pub async fn aclose(&self) -> Closed {
        if let Some(outcome) = *self.cleanup.lock().unwrap() {
            return outcome;
        }
        *self.closed.lock().unwrap() = true;
        self.state.lock().unwrap().close();
        self.changed.notify_waiters();
        self.stopped.notify_waiters();
        let outcome = if self
            .tasks
            .wait_empty(
                self.clock.as_ref(),
                Some(self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)),
            )
            .await
        {
            Closed::Clean
        } else {
            Closed::TimedOut
        };
        *self.cleanup.lock().unwrap().get_or_insert(outcome.and(*self.outcome.lock().unwrap()))
    }
}
