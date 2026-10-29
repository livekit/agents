//! The avatar worker's runner (`AvatarRunner`): host I/O and ownership around the synchronous
//! RunnerCore, implementing its ABI class directly.

use lk_agents_abi::voice::avatar::{
    self, AvatarAudioReceiver, AvatarMedia, AvatarVideoGenerator, RunnerError,
};
use lk_agents_abi::voice::room_owner::RoomSession;
use lk_agents_abi::voice::rtc::{RoomEvent, RtcError, RtcRoom};
use lk_agents_abi::{Clock, Spawner, SpawnerExt, TaskSet};
use lk_agents_net::env::ProcessEnv;
pub use lk_agents_voice::avatar_runner::Event;
use lk_agents_voice::avatar_runner::{CallId, Effect, IoOp, RunnerCore};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
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
    pub async fn perform(&self, op: IoOp) -> Event {
        match op {
            IoOp::OpenAudio => Event::AudioOpened(self.receiver.open_audio().await),
            IoOp::NextAudio => Event::Audio(self.receiver.next_audio().await),
            IoOp::PushAudio(frame) => Event::AudioPushed(self.generator.push_audio(frame).await),
            IoOp::OpenFrames => Event::FramesOpened(self.generator.open_frames().await),
            IoOp::NextFrame => Event::Frame(self.generator.next_frame().await),
            IoOp::Publish => Event::Published(self.media.publish().await),
            IoOp::Push(frame) => Event::Pushed(self.media.push(frame).await),
            IoOp::ReportStarted { segment } => Event::Reported {
                finished: false,
                result: self.receiver.notify_playback_started(segment).await,
            },
            IoOp::ReportFinished { segment, position, interrupted } => Event::Reported {
                finished: true,
                result: self
                    .receiver
                    .notify_playback_finished(segment, position, interrupted)
                    .await,
            },
            IoOp::ClearGenerator { clear } => {
                Event::GeneratorCleared { clear, result: self.generator.clear_buffer().await }
            }
            IoOp::CloseReceiver => Event::ReceiverClosed(self.receiver.aclose().await),
            IoOp::CloseMedia => Event::MediaClosed(self.media.aclose().await),
        }
    }
}
type Answers = Arc<Mutex<HashMap<CallId, oneshot::Sender<Result<(), RunnerError>>>>>;

/// `AvatarRunner`: the queue owner of one [`RunnerCore`]. Host calls and the room become events,
/// each effect runs as a task whose answer is the next event.
pub struct AvatarRunner {
    room: Arc<dyn RtcRoom>,
    events: mpsc::UnboundedSender<Event>,
    answers: Answers,
    next_call: std::sync::atomic::AtomicU64,
    spawner: Arc<dyn Spawner>,
    io_tasks: Arc<TaskSet>,
    owner: lk_agents_abi::AbortHandle,
}

impl Drop for AvatarRunner {
    fn drop(&mut self) {
        self.owner.abort();
        self.io_tasks.abort_all();
    }
}

impl AvatarRunner {
    /// The runner, its ready events taken in the order `order(ready)` picks (an index into
    /// `ready`): arrival order in production, the simulation's own choice under test.
    pub fn scheduled(
        room: Arc<dyn RtcRoom>,
        io: AvatarIo,
        lazy_publish: bool,
        clock: Arc<dyn Clock>,
        spawner: Arc<dyn Spawner>,
        mut order: impl FnMut(&std::collections::VecDeque<Event>) -> usize + Send + 'static,
    ) -> Self {
        let (events, mut queue) = mpsc::unbounded_channel();
        let answers: Answers = Arc::default();
        let io_tasks: Arc<TaskSet> = Arc::default();
        let owner = {
            let (events, answers, io_tasks, spawner) =
                (events.clone(), Arc::clone(&answers), Arc::clone(&io_tasks), Arc::clone(&spawner));
            spawner.clone().spawn_unboxed(async move {
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
                            Effect::Io(op) => {
                                let (io, events) = (io.clone(), events.clone());
                                io_tasks.spawn(&*spawner, "avatar runner op", async move {
                                    let _ = events.send(io.perform(op).await);
                                });
                            }
                            Effect::CancelIo => {
                                let (io_tasks, events, clock) =
                                    (Arc::clone(&io_tasks), events.clone(), Arc::clone(&clock));
                                spawner.spawn_unboxed(async move {
                                    io_tasks.cancel_and_wait(&*clock).await;
                                    let _ = events.send(Event::IoCancelled);
                                });
                            }
                            Effect::Answer { call, result } => {
                                let tx =
                                    answers.lock().unwrap_or_else(|p| p.into_inner()).remove(&call);
                                if let Some(tx) = tx {
                                    let _ = tx.send(result);
                                }
                            }
                        }
                    }
                }
            })
        };
        AvatarRunner {
            room,
            events,
            answers,
            next_call: Default::default(),
            spawner,
            io_tasks,
            owner,
        }
    }

    /// Whether an op, a report or the room watch is still running: none once closed.
    pub fn io_in_flight(&self) -> bool {
        !self.io_tasks.is_empty()
    }

    /// Hand the core one host call and wait for its answer.
    async fn call(&self, event: impl FnOnce(CallId) -> Event) -> Result<(), RunnerError> {
        let call = self.next_call.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.answers.lock().unwrap_or_else(|p| p.into_inner()).insert(call, tx);
        let _ = self.events.send(event(call));
        rx.await.unwrap_or(Err(RunnerError::Closed))
    }
}
impl avatar::AvatarRunner for AvatarRunner {
    type Env = ProcessEnv;

    /// `room` is the avatar worker's `rtc.Room`; `media` the SDK half (`rtc.AVSynchronizer` and
    /// the tracks); `lazy_publish` holds the tracks back until the first frame.
    fn new(
        env: &ProcessEnv,
        room: RoomSession,
        receiver: Arc<dyn AvatarAudioReceiver>,
        generator: Arc<dyn AvatarVideoGenerator>,
        media: Arc<dyn AvatarMedia>,
        lazy_publish: bool,
    ) -> Result<Self, RtcError> {
        let room = room.0.rtc().ok_or(RtcError::NotConnected)?;
        let spawner = (env.spawner)().map_err(RtcError::host)?;
        let io = AvatarIo { receiver, generator, media };
        let clock = env.clock.clone();
        Ok(AvatarRunner::scheduled(room, io, lazy_publish, clock, spawner, |_| 0))
    }

    /// `start()`, once the receiver has started (`await audio_recv.start()`).
    async fn start(&self) -> Result<(), RunnerError> {
        // subscribed before the state is read, so no change falls between the two
        let mut room_events = self.room.events();
        let connected = self.room.connection_state()?.is_connected();
        let events = self.events.clone();
        self.io_tasks.spawn(&*self.spawner, "avatar room connection", async move {
            loop {
                match room_events.next().await {
                    Ok(Some(RoomEvent::ConnectionStateChanged { state })) => {
                        let _ = events.send(Event::Connection { connected: state.is_connected() });
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => return,
                    Err(error) => {
                        let _ = events.send(Event::RoomFailed(error));
                        return;
                    }
                }
            }
        });
        self.call(|call| Event::Start { call, connected }).await
    }

    /// Both loops ran out; the first failure as soon as one fails; `CancelledError` once closed.
    async fn wait_for_complete(&self) -> Result<(), RunnerError> {
        self.call(|call| Event::WaitForComplete { call }).await
    }

    /// The receiver's `clear_buffer` event.
    fn clear_buffer(&self, epoch: u64) {
        let _ = self.events.send(Event::ClearBuffer { epoch });
    }

    /// `aclose()`: the receiver, then the loops and what is in flight, then the media.
    async fn aclose(&self) -> Result<(), RunnerError> {
        self.call(|call| Event::Close { call }).await
    }
}

#[cfg(test)]
#[path = "runner_tests.rs"]
mod tests;
