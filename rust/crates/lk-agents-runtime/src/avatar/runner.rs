//! The avatar worker's runner (`AvatarRunner`): host I/O and ownership around the synchronous
//! RunnerCore, implementing its ABI class directly.

use crate::telemetry;
use lk_agents_abi::voice::avatar::{
    self, AvatarAudioReceiver, AvatarMedia, AvatarVideoGenerator, RunnerError,
};
use lk_agents_abi::voice::room_owner::SessionRoom;
use lk_agents_abi::voice::rtc::{RoomEvent, RtcError, RtcRoom};
use lk_agents_abi::{Clock, Spawner, SpawnerExt};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TaskSet;
use lk_agents_voice::avatar::effects::AvatarEffect;
pub use lk_agents_voice::avatar::machine::runner::AvatarEvent;
use lk_agents_voice::avatar::machine::runner::{AvatarIoOperation, CallId, RunnerCore};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

// ----------------------------------------------------------------------------- the I/O layer

/// The host objects the runner's ops run against.
#[derive(Clone)]
pub struct AvatarIo {
    pub receiver: Arc<dyn AvatarAudioReceiver>,
    pub generator: Arc<dyn AvatarVideoGenerator>,
    pub media: Arc<dyn AvatarMedia>,
}

impl AvatarIo {
    /// Run one op; its answer.
    pub async fn perform(&self, op: AvatarIoOperation) -> AvatarEvent {
        match op {
            AvatarIoOperation::OpenAudio => {
                AvatarEvent::AudioOpened(self.receiver.open_audio(&telemetry::Cx::NONE).await)
            }
            AvatarIoOperation::NextAudio => {
                AvatarEvent::Audio(self.receiver.next_audio(&telemetry::Cx::NONE).await)
            }
            AvatarIoOperation::PushAudio(frame) => AvatarEvent::AudioPushed(
                self.generator.push_audio(&telemetry::Cx::NONE, frame).await,
            ),
            AvatarIoOperation::OpenFrames => {
                AvatarEvent::FramesOpened(self.generator.open_frames(&telemetry::Cx::NONE).await)
            }
            AvatarIoOperation::NextFrame => {
                AvatarEvent::Frame(self.generator.next_frame(&telemetry::Cx::NONE).await)
            }
            AvatarIoOperation::Publish => AvatarEvent::Published(self.media.publish().await),
            AvatarIoOperation::Push(frame) => AvatarEvent::Pushed(self.media.push(frame).await),
            AvatarIoOperation::ReportStarted { segment } => AvatarEvent::Reported {
                finished: false,
                result: self.receiver.notify_playback_started(&telemetry::Cx::NONE, segment).await,
            },
            AvatarIoOperation::ReportFinished { segment, position, interrupted } => {
                AvatarEvent::Reported {
                    finished: true,
                    result: self
                        .receiver
                        .notify_playback_finished(
                            &telemetry::Cx::NONE,
                            segment,
                            position,
                            interrupted,
                        )
                        .await,
                }
            }
            AvatarIoOperation::ClearGenerator { clear } => AvatarEvent::GeneratorCleared {
                clear,
                result: self.generator.clear_buffer(&telemetry::Cx::NONE).await,
            },
            AvatarIoOperation::CloseReceiver => {
                AvatarEvent::ReceiverClosed(self.receiver.aclose(&telemetry::Cx::NONE).await)
            }
            AvatarIoOperation::CloseMedia => AvatarEvent::MediaClosed(self.media.aclose().await),
        }
    }
}
type Answers = Arc<Mutex<HashMap<CallId, oneshot::Sender<Result<(), RunnerError>>>>>;

/// `AvatarRunner`: the queue owner of one [`RunnerCore`]. Host calls and the room become events,
/// each effect runs as a task whose answer is the next event.
pub struct Runner {
    room: Arc<dyn RtcRoom>,
    events: mpsc::UnboundedSender<AvatarEvent>,
    answers: Answers,
    next_call: std::sync::atomic::AtomicU64,
    spawner: Arc<dyn Spawner>,
    io_tasks: Arc<TaskSet>,
    /// The queue owner's loop; cancelled with the runner.
    _owner: lk_agents_abi::OwnedTask,
}

impl Drop for Runner {
    fn drop(&mut self) {
        self.io_tasks.cancel_all();
    }
}

impl Runner {
    /// The runner, its ready events taken in the order `order(ready)` picks (an index into
    /// `ready`): arrival order in production, the simulation's own choice under test.
    pub fn scheduled(
        room: Arc<dyn RtcRoom>,
        io: AvatarIo,
        lazy_publish: bool,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        mut order: impl FnMut(&std::collections::VecDeque<AvatarEvent>) -> usize + Send + 'static,
    ) -> Self {
        let (events, mut queue) = mpsc::unbounded_channel();
        let answers: Answers = Arc::default();
        let io_tasks: Arc<TaskSet> = Arc::default();
        let owner = {
            let (events, answers, io_tasks, spawner) =
                (events.clone(), Arc::clone(&answers), Arc::clone(&io_tasks), Arc::clone(&spawner));
            spawner.clone().spawn(async move {
                let mut core = RunnerCore::new(lazy_publish);
                let mut ready = std::collections::VecDeque::new();
                loop {
                    if ready.is_empty() {
                        match queue.recv().await {
                            Some(event) => ready.push_back(event),
                            None => return,
                        }
                    }
                    while let Ok(event) = queue.try_recv() {
                        ready.push_back(event);
                    }
                    let pick = order(&ready).min(ready.len() - 1);
                    let Some(event) = ready.remove(pick) else { continue };
                    for effect in core.handle(event, clock.now()) {
                        match effect {
                            AvatarEffect::Io(op) => {
                                let (io, events) = (io.clone(), events.clone());
                                io_tasks.spawn(&*spawner, "avatar runner op", async move {
                                    let _ = events.send(io.perform(op).await);
                                });
                            }
                            AvatarEffect::CancelIo => {
                                let (io_tasks, events, clock) =
                                    (Arc::clone(&io_tasks), events.clone(), Arc::clone(&clock));
                                // Detached: it ends once the cancelled ops settled, and only
                                // reports to the loop, which ignores it once gone.
                                spawner.spawn_detached(async move {
                                    io_tasks.cancel_and_wait(&*clock).await;
                                    let _ = events.send(AvatarEvent::IoCancelled);
                                });
                            }
                            AvatarEffect::Answer { call, result } => {
                                let tx = answers.lock().remove(&call);
                                if let Some(tx) = tx {
                                    let _ = tx.send(result);
                                }
                            }
                        }
                    }
                }
            })
        };
        Runner {
            room,
            events,
            answers,
            next_call: Default::default(),
            spawner,
            io_tasks,
            _owner: owner,
        }
    }

    /// Whether an op, a report or the room watch is still running: none once closed.
    pub fn io_in_flight(&self) -> bool {
        !self.io_tasks.is_empty()
    }

    /// Hand the core one host call and wait for its answer.
    async fn call(&self, event: impl FnOnce(CallId) -> AvatarEvent) -> Result<(), RunnerError> {
        let call = self.next_call.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.answers.lock().insert(call, tx);
        let _ = self.events.send(event(call));
        rx.await.unwrap_or(Err(RunnerError::Closed))
    }
}
/// The exported handle always owns its synchronizer; the executor also runs over port fakes.
pub struct AvatarRunner {
    runner: Runner,
    synchronizer: Arc<crate::rtc::synchronizer::AVSynchronizer>,
}

impl avatar::AvatarRunner for AvatarRunner {
    type Env = ProcessEnv;
    type Synchronizer = crate::rtc::synchronizer::AVSynchronizer;

    /// `room` is the avatar worker's `rtc.Room`; `media` the SDK half (`rtc.AVSynchronizer` and
    /// the tracks); `lazy_publish` holds the tracks back until the first frame.
    fn new(
        env: &ProcessEnv,
        room: Arc<dyn SessionRoom>,
        receiver: Arc<dyn AvatarAudioReceiver>,
        generator: Arc<dyn AvatarVideoGenerator>,
        options: avatar::AvatarOptions,
        queue_size_ms: u32,
        lazy_publish: bool,
    ) -> Result<Self, RtcError> {
        let room = room.rtc().ok_or(RtcError::NotConnected)?;
        let spawner = (env.spawner)().map_err(RtcError::host)?;
        let plan = lk_agents_voice::av_sync::MediaPlan::new(options, queue_size_ms)
            .map_err(RtcError::host)?;
        let clock = env.clock.clone();
        let media = super::media::Media::new(room.clone(), plan, clock.clone(), spawner.clone())?;
        let synchronizer = media.synchronizer.clone();
        let io = AvatarIo { receiver, generator, media };
        let runner = Runner::scheduled(room, io, lazy_publish, clock, spawner, |_| 0);
        Ok(Self { runner, synchronizer })
    }

    fn av_sync(&self) -> Arc<Self::Synchronizer> {
        self.synchronizer.clone()
    }

    async fn start(&self) -> Result<(), RunnerError> {
        self.runner.start().await
    }
    async fn wait_for_complete(&self) -> Result<(), RunnerError> {
        self.runner.wait_for_complete().await
    }
    fn clear_buffer(&self, epoch: u64) {
        self.runner.clear_buffer(epoch);
    }
    async fn aclose(&self) -> Result<(), RunnerError> {
        self.runner.aclose().await
    }
}

impl Runner {
    /// `start()`, once the receiver has started (`await audio_recv.start()`).
    pub async fn start(&self) -> Result<(), RunnerError> {
        // subscribed before the state is read, so no change falls between the two
        let mut room_events = self.room.events();
        let connected = self.room.connection_state()?.is_connected();
        let events = self.events.clone();
        self.io_tasks.spawn(&*self.spawner, "avatar room connection", async move {
            loop {
                match room_events.next().await {
                    Ok(Some(RoomEvent::ConnectionStateChanged { state })) => {
                        let _ = events
                            .send(AvatarEvent::Connection { connected: state.is_connected() });
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => return,
                    Err(error) => {
                        let _ = events.send(AvatarEvent::RoomFailed(error));
                        return;
                    }
                }
            }
        });
        self.call(|call| AvatarEvent::Start { call, connected }).await
    }

    /// Both loops ran out; the first failure as soon as one fails; `CancelledError` once closed.
    pub async fn wait_for_complete(&self) -> Result<(), RunnerError> {
        self.call(|call| AvatarEvent::WaitForComplete { call }).await
    }

    /// The receiver's `clear_buffer` event.
    pub fn clear_buffer(&self, epoch: u64) {
        let _ = self.events.send(AvatarEvent::ClearBuffer { epoch });
    }

    /// `aclose()`: the receiver, then the loops and what is in flight, then the media.
    pub async fn aclose(&self) -> Result<(), RunnerError> {
        self.call(|call| AvatarEvent::Close { call }).await
    }
}
