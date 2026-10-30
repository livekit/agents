//! Gives the real runtime queue owner to the deterministic harness, which runs its own loop
//! ([`Session`]): `Runtime::serve` and `Runtime::close`, over the same injected provider/SDK ports
//! and session Clock/Spawner as production. The harness only chooses what reaches the loop's queue
//! and when.

use lk_agents_abi::{AudioFrame, BoxFuture, Spawner, Stt, Vad};
use lk_agents_runtime::voice::resources::Resources;
use lk_agents_runtime::voice::session::{
    OpeningLegs, QueueOwner, SessionCores, SessionExecutor, Timebase,
};
use lk_agents_runtime::voice::{Executor, Runtime};
use lk_agents_voice::agent_session::{DriverError, Event, Now};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use tokio::sync::{mpsc, oneshot};

pub(crate) struct Handover {
    factory: Executor,
    run: Option<oneshot::Sender<Runtime>>,
}

impl Handover {
    pub(crate) fn new(
        run: oneshot::Sender<Runtime>,
        time: Timebase,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        Self { factory: Executor::new(time, spawner), run: Some(run) }
    }
}

impl SessionExecutor for Handover {
    fn stt_adapter(&self) -> lk_agents_runtime::voice::session::SttAdapter {
        self.factory.stt_adapter()
    }
    fn tts_adapter(&self) -> lk_agents_runtime::voice::session::TtsAdapter {
        self.factory.tts_adapter()
    }
    fn open_legs<'a>(
        &'a mut self,
        resources: &'a Resources,
        stt: Option<Arc<dyn Stt>>,
        vad: Option<Arc<dyn Vad>>,
    ) -> BoxFuture<'a, Result<OpeningLegs, DriverError>> {
        self.factory.open_legs(resources, stt, vad)
    }
    fn open_realtime<'a>(
        &'a mut self,
        resources: &'a lk_agents_runtime::voice::resources::Resources,
        model: Arc<dyn lk_agents_abi::Realtime>,
        disabled: bool,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<
            (Arc<dyn lk_agents_abi::RealtimeSession>, lk_agents_abi::resources::ResourceId),
            DriverError,
        >,
    > {
        self.factory.open_realtime(resources, model, disabled)
    }

    fn own(self: Box<Self>, cores: SessionCores) -> Box<dyn QueueOwner> {
        let Self { factory, run } = *self;
        Box::new(Running { runtime: Some(factory.into_runtime(cores)), run })
    }
}

struct Running {
    runtime: Option<Runtime>,
    run: Option<oneshot::Sender<Runtime>>,
}

impl QueueOwner for Running {
    fn open(&mut self, now: Now) {
        if let Some(runtime) = &mut self.runtime {
            runtime.open(now);
        }
    }
    fn run(mut self: Box<Self>) -> Option<BoxFuture<'static, ()>> {
        if let (Some(runtime), Some(run)) = (self.runtime.take(), self.run.take()) {
            let _ = run.send(runtime);
        }
        None
    }
}

/// The session's real loop, polled by the run. Everything the session's sources send lands in
/// `intake` (and the microphone's frames in `mic`), the channels the runtime was built with; the
/// run's queue schedules it and hands the step it chose to the runtime's own queue, then polls the
/// loop at once. Once the decision core stopped, what the run still held is handed over in arrival
/// order, as it would have sat in the production queue, and the runtime closes: it finishes that
/// queue and drains its tasks against the cleanup bound.
pub(crate) struct Session {
    stage: Stage,
    pub intake: mpsc::UnboundedReceiver<Event>,
    pub mic: mpsc::Receiver<AudioFrame>,
    events: mpsc::UnboundedSender<Event>,
    frames: mpsc::Sender<AudioFrame>,
    woken: Arc<Woken>,
}

enum Stage {
    Serving(BoxFuture<'static, Box<Runtime>>),
    Closing(BoxFuture<'static, Vec<String>>),
    Closed,
}

/// What one poll of the loop came to.
pub(crate) enum Polled {
    Running,
    /// The decision core stopped: [`Session::close`] next.
    Stopped(Box<Runtime>),
    /// The close returned: the tasks it cancelled at the cleanup bound.
    Closed(Vec<String>),
}

impl Session {
    pub(crate) fn new(mut runtime: Runtime) -> Self {
        let (events, queue) = mpsc::unbounded_channel();
        let (frames, mic) = mpsc::channel(tokio::sync::Semaphore::MAX_PERMITS);
        let intake = std::mem::replace(&mut runtime.queue, queue);
        let mic = std::mem::replace(&mut runtime.frames, mic);
        let mut runtime = Box::new(runtime);
        let serve = Box::pin(async move {
            runtime.serve().await;
            runtime
        });
        let woken = Arc::new(Woken { woken: AtomicBool::new(true), run: Mutex::default() });
        Session { stage: Stage::Serving(serve), intake, mic, events, frames, woken }
    }

    pub(crate) fn serving(&self) -> bool {
        matches!(self.stage, Stage::Serving(_))
    }

    pub(crate) fn deliver(&mut self, event: Event) {
        let _ = self.events.send(event);
        self.woken.woken.store(true, Ordering::Relaxed);
    }

    pub(crate) fn deliver_frames(&mut self, frames: Vec<AudioFrame>) {
        for frame in frames {
            self.frames.try_send(frame).expect("the delivery channel holds any batch");
        }
        self.woken.woken.store(true, Ordering::Relaxed);
    }

    /// The loop stopped: the queue closes to its senders, and what reached it, in arrival order,
    /// goes to the runtime's close. The microphone's channel stays as production leaves it, unread.
    pub(crate) fn close(&mut self, runtime: Box<Runtime>, held: Vec<Event>) {
        self.intake.close();
        let late = std::iter::from_fn(|| self.intake.try_recv().ok());
        for event in held.into_iter().chain(late) {
            let _ = self.events.send(event);
        }
        let mut runtime = runtime;
        self.stage = Stage::Closing(Box::pin(async move { runtime.close().await }));
        self.woken.woken.store(true, Ordering::Relaxed);
    }

    /// Polls the loop if anything woke it since its last poll, or it was handed a step.
    pub(crate) fn poll(&mut self) -> Polled {
        if !self.woken.woken.swap(false, Ordering::Relaxed) {
            return Polled::Running;
        }
        let waker = Waker::from(Arc::clone(&self.woken));
        let mut cx = Context::from_waker(&waker);
        match &mut self.stage {
            Stage::Serving(serve) => match serve.as_mut().poll(&mut cx) {
                Poll::Pending => Polled::Running,
                Poll::Ready(runtime) => {
                    self.stage = Stage::Closed;
                    Polled::Stopped(runtime)
                }
            },
            Stage::Closing(close) => match close.as_mut().poll(&mut cx) {
                Poll::Pending => Polled::Running,
                Poll::Ready(late) => {
                    self.stage = Stage::Closed;
                    Polled::Closed(late)
                }
            },
            Stage::Closed => Polled::Running,
        }
    }

    /// Resolves once the loop was woken since its last poll.
    pub(crate) fn woken(&self) -> impl Future<Output = ()> + use<> {
        let woken = Arc::clone(&self.woken);
        std::future::poll_fn(move |cx| {
            *woken.run.lock().unwrap() = Some(cx.waker().clone());
            if woken.woken.load(Ordering::Relaxed) { Poll::Ready(()) } else { Poll::Pending }
        })
    }
}

/// The loop's waker: marks it to be polled, and wakes the run waiting on it.
struct Woken {
    woken: AtomicBool,
    run: Mutex<Option<Waker>>,
}

impl Wake for Woken {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.woken.store(true, Ordering::Relaxed);
        if let Some(run) = self.run.lock().unwrap().take() {
            run.wake();
        }
    }
}
