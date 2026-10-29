//! The driver's queue, owned by the case: what every source sent, delivered one event at a time in
//! the order the run's policy chooses.
//!
//! In production the queue is a channel and the order is arrival order. Here the case collects
//! what reached the channel and picks what runs next, so an interleaving the channel would only
//! produce under load is one a seed can choose. The choice is the shared scheduler's
//! ([`crate::sched`]): each source keeps its own order, and BasicPOS samples the rest. The loop
//! takes its own due wakes, before anything handed to it. The microphone's frames are a source of their own: the frames that reached the run since its last
//! step are one batch, delivered as one step in their order, racing the events under POS and
//! ahead of them in arrival order (the production loop takes frames before its queue). The
//! driver's ready tasks are a process of the scheduler, and so is each bridged task.
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::mem::Discriminant;
use std::pin::Pin;
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll, Wake, Waker};

use fastrand::Rng;
use lk_agents_abi::{AudioFrame, BoxFuture, resources::ResourceId};
use lk_agents_voice::agent_session::{Event, Input};

use crate::sched::{Sampling, Sched, Step};

/// How the next step is chosen, drawn per seed and carried by the case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Policy {
    /// The channel's own order: the driver drains what is queued, and the tasks it spawned run
    /// when the queue is empty.
    #[default]
    Arrival,
    /// Basic partial-order sampling ([`Sampling::BasicPos`]). The processes are each event source
    /// and the pool of tasks the driver spawned, which takes one turn when chosen.
    BasicPos,
    /// BasicPOS with the task bridge ([`Gate`]): every task the run's spawner spawns is a process
    /// of its own, with its own priority, rather than one turn the pool shares, so the order of
    /// two tasks is sampled too; after its turn a task draws a new priority. Transitional: it
    /// schedules the voice policy tasks that are not core machines yet, and goes with the last
    /// of them.
    TaskBridge,
}

impl Policy {
    pub const ALL: [Policy; 3] = [Policy::Arrival, Policy::BasicPos, Policy::TaskBridge];

    /// `VOICE_POLICY=BasicPos`: one policy by its name.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| format!("{p:?}") == name)
    }

    /// A seed's own policy: the task bridge for half the seeds, BasicPOS for a quarter and
    /// arrival order for a quarter, which is still what reaches some bugs first (the paper and
    /// gosim found the same). `bench/known_bugs.sh` measures each over the known bugs.
    pub fn of(schedule: u64) -> Self {
        match Rng::with_seed(schedule ^ POLICY_STREAM).u8(0..4) {
            0 => Policy::Arrival,
            1 => Policy::BasicPos,
            _ => Policy::TaskBridge,
        }
    }

    fn sampling(self) -> Sampling {
        match self {
            Policy::Arrival => Sampling::Arrival,
            Policy::BasicPos | Policy::TaskBridge => Sampling::BasicPos,
        }
    }
}

/// The policy's own random stream off the schedule seed: the bytes `policy`.
const POLICY_STREAM: u64 = 0x0070_6f6c_6963_79;

/// The pool of the driver's ready tasks, as a process of the scheduler (bridged tasks count from
/// one).
const POOL: u64 = u64::MAX;

/// What runs next.
pub enum Next {
    Event(Event),
    /// The microphone's frames that arrived since the last step, in order.
    Frames(Vec<AudioFrame>),
    /// One turn for the driver's ready tasks.
    Tasks,
    /// One turn for one bridged task.
    Task(u64),
}

pub struct Queue {
    sched: Sched<Source, Item>,
    /// What was dispatched, in order, folded: two runs that dispatched the same steps in the same
    /// order agree on it.
    order: u64,
    trail: Arc<Mutex<Trail>>,
}

/// The last steps dispatched and what is pending, by kind, shared with the run's watchdog: what a
/// hang that blocks the run's thread is reported with, and what a stopped run's ending reads.
#[derive(Default)]
pub struct Trail {
    /// The last [`TRAIL`] dispatches: their number and kind.
    pub last: VecDeque<(u64, &'static str)>,
    /// What is pending, by arrival.
    pub pending: BTreeMap<u64, &'static str>,
}

/// Dispatches the trail keeps.
const TRAIL: usize = 64;

thread_local! {
    /// The trail of the simulation running on this thread (`drive::simulate_with`).
    static CURRENT: RefCell<Arc<Mutex<Trail>>> = RefCell::default();
}

impl Trail {
    /// A fresh trail for the simulation about to run on this thread.
    pub fn install() -> Arc<Mutex<Trail>> {
        let trail = Arc::default();
        CURRENT.set(Arc::clone(&trail));
        trail
    }

    /// The smallest period, of at most a quarter of the trail, that the whole trail repeats:
    /// the cycle a run that stopped at its step bound was spinning in. `None` when the trail is
    /// not full or does not repeat, which is no evidence of a livelock.
    pub fn cycle(&self) -> Option<Vec<&'static str>> {
        let kinds: Vec<&str> = self.last.iter().map(|(_, kind)| *kind).collect();
        if kinds.len() < TRAIL {
            return None;
        }
        (1..=TRAIL / 4)
            .find(|&p| (p..kinds.len()).all(|i| kinds[i] == kinds[i - p]))
            .map(|p| kinds[kinds.len() - p..].to_vec())
    }

    pub fn dump(&self) -> String {
        let last: Vec<String> = self.last.iter().map(|(n, k)| format!("{n}:{k}")).collect();
        let pending: Vec<String> = self.pending.iter().map(|(a, k)| format!("{a}:{k}")).collect();
        format!(
            "last {} dispatches: {}\npending: {}",
            last.len(),
            last.join(" "),
            if pending.is_empty() { "nothing".into() } else { pending.join(" ") }
        )
    }
}

/// An event's kind, without formatting it: what the trail keeps.
fn label(event: &Event) -> &'static str {
    match event {
        Event::Wake => "Wake",
        Event::HostRequests => "HostRequests",
        Event::Forwarder(_) => "Forwarder",
        Event::LegOpened(_) => "StreamOpened",
        Event::Input(input) | Event::Awaited(input, _) => match input {
            Input::Host(sent) => sent.command.kind(),
            Input::Stt(..) => "Stt",
            Input::Vad(_) => "Vad",
            Input::Room(_) => "Room",
            Input::RoomIo(_) => "RoomIo",
            Input::Gen(_) => "Gen",
            Input::Realtime(_) => "Realtime",
            Input::Eot(..) => "Eot",
            Input::Overlap(_) => "Overlap",
            Input::ChannelEnded(_) => "ChannelEnded",
            Input::SttRetried(_) => "SttRetried",
            Input::StreamOpened { .. } => "StreamOpened",
            Input::Video(_) => "Video",
            Input::InputStarted { .. } => "InputStarted",
            Input::GateSilenced(_) => "GateSilenced",
            Input::InputCensus(_) => "InputCensus",
            Input::StorageReloaded(_) => "StorageReloaded",
            Input::StorageSettled(_) => "StorageSettled",
            Input::StorageAbandoned(_) => "StorageAbandoned",
            Input::StorageLost(_) => "StorageLost",
        },
    }
}

enum Item {
    Event(Event),
    Frames(Vec<AudioFrame>),
}

/// Who sent an event: one host origin, one recognition stream, one detector leg, one kind of
/// input, or the decision core's commands to the data plane. A leg's end comes from the drain that
/// sent its items, after them, so it is that leg's.
#[derive(PartialEq, Eq)]
enum Source {
    Host(String),
    Stt(ResourceId),
    Leg(&'static str),
    Kind(Discriminant<Input>),
    Event(Discriminant<Event>),
    /// The microphone's frames channel.
    Frames,
}

impl crate::sched::Source for Source {
    fn polled_first(&self) -> bool {
        *self == Source::Frames
    }
}

impl Source {
    /// An awaited input comes from the source the same input would without its reply.
    fn of(event: &Event) -> Self {
        let input = match event {
            Event::Input(input) | Event::Awaited(input, _) => input,
            other => return Source::Event(std::mem::discriminant(other)),
        };
        match input {
            Input::Host(sent) => Source::Host(format!("{:?}", sent.origin)),
            Input::Stt(tagged, _) => Source::Stt(tagged.stream),
            Input::Vad(_) => Source::Leg("vad"),
            Input::Eot(..) => Source::Leg("eot"),
            Input::Overlap(_) => Source::Leg("interruption"),
            Input::ChannelEnded(end) => match (end.leg(), end.sink()) {
                ("stt", Some(stream)) => Source::Stt(stream),
                (leg @ ("vad" | "eot" | "interruption"), _) => Source::Leg(leg),
                _ => Source::Kind(std::mem::discriminant(input)),
            },
            other => Source::Kind(std::mem::discriminant(other)),
        }
    }
}

impl Queue {
    pub fn new(schedule: u64, policy: Policy) -> Self {
        let rng = Rng::with_seed(schedule ^ POLICY_STREAM.rotate_left(32));
        Queue {
            sched: Sched::new(policy.sampling(), rng, &[POOL]),
            order: ORDER_SEED,
            trail: CURRENT.with_borrow(Arc::clone),
        }
    }

    pub fn push(&mut self, event: Event) {
        let kind = label(&event);
        let arrival = self.sched.push(Source::of(&event), Item::Event(event));
        self.trail.lock().unwrap().pending.insert(arrival, kind);
    }

    /// A microphone frame: it joins the pending batch, or starts one.
    pub fn push_frame(&mut self, frame: AudioFrame) {
        if let Some(Item::Frames(batch)) = self.sched.pending_mut(&Source::Frames) {
            batch.push(frame);
            return;
        }
        let arrival = self.sched.push(Source::Frames, Item::Frames(vec![frame]));
        self.trail.lock().unwrap().pending.insert(arrival, "Frames");
    }

    /// The next step: an event, a bridged task's turn (`ready`, under [`Policy::TaskBridge`]), a
    /// turn for the pool when `tasks_ready`, or nothing.
    pub fn next(&mut self, tasks_ready: bool, ready: &[u64]) -> Option<Next> {
        let processes: Vec<u64> =
            tasks_ready.then_some(POOL).into_iter().chain(ready.iter().copied()).collect();
        match self.sched.next(&processes)? {
            Step::Item { arrival, item } => {
                self.fold(1, arrival);
                Some(match item {
                    Item::Event(event) => {
                        self.note(Some(arrival), label(&event));
                        Next::Event(event)
                    }
                    Item::Frames(frames) => {
                        self.note(Some(arrival), "Frames");
                        Next::Frames(frames)
                    }
                })
            }
            Step::Process(POOL) => {
                self.fold(3, 0);
                self.note(None, "Tasks");
                Some(Next::Tasks)
            }
            Step::Process(id) => {
                self.fold(2, id);
                self.note(None, "Task");
                Some(Next::Task(id))
            }
        }
    }

    /// Everything still pending, in arrival order: the frames, and the events.
    pub fn drain(&mut self) -> (Vec<AudioFrame>, Vec<Event>) {
        self.trail.lock().unwrap().pending.clear();
        let (mut frames, mut events) = (Vec::new(), Vec::new());
        for item in self.sched.drain() {
            match item {
                Item::Event(event) => events.push(event),
                Item::Frames(batch) => frames.extend(batch),
            }
        }
        (frames, events)
    }

    /// The dispatch order so far, folded into one number: what replaying a seed must reproduce.
    pub fn order(&self) -> u64 {
        self.order
    }

    /// The trail this queue writes.
    pub fn trail(&self) -> Arc<Mutex<Trail>> {
        Arc::clone(&self.trail)
    }

    /// Whether nothing is pending.
    pub fn idle(&self) -> bool {
        self.sched.idle()
    }

    fn note(&self, arrival: Option<u64>, kind: &'static str) {
        let mut trail = self.trail.lock().unwrap();
        if let Some(arrival) = arrival {
            trail.pending.remove(&arrival);
        }
        if trail.last.len() == TRAIL {
            trail.last.pop_front();
        }
        trail.last.push_back((self.sched.dispatches(), kind));
    }

    /// FNV-1a over the dispatch's kind and identity (an event's arrival, a task's id).
    fn fold(&mut self, tag: u8, id: u64) {
        for byte in std::iter::once(tag).chain(id.to_le_bytes()) {
            self.order = (self.order ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }
}

/// FNV-1a's offset basis.
const ORDER_SEED: u64 = 0xcbf2_9ce4_8422_2325;

/// The task bridge: under [`Policy::TaskBridge`] every future the run's spawner spawns runs
/// behind a gate, polled only when the queue gives that task its turn. Its own waker marks it
/// ready, and wakes the run, which may be waiting on time; tokio only ever sees a task woken to
/// take its granted turn. The gate is shut only while the run steps the session: when the run
/// awaits something itself (a host call, the session's start, a player's close) or the session
/// is gone, what is ready runs as tokio schedules it, as the host's loop outlives the session's.
///
/// Transitional: it exists because voice policy tasks still run as tasks; it
/// is deleted with the last of them.
pub struct Gate {
    state: Mutex<GateState>,
    woken: tokio::sync::Notify,
}

#[derive(Default)]
struct GateState {
    next: u64,
    open: bool,
    granted: Option<u64>,
    slots: BTreeMap<u64, Slot>,
}

struct Slot {
    ready: bool,
    /// The tokio task's own waker, as its last poll left it.
    outer: Option<Waker>,
}

impl Gate {
    pub fn new() -> Self {
        let state = GateState { open: true, ..GateState::default() };
        Gate { state: Mutex::new(state), woken: tokio::sync::Notify::new() }
    }

    /// Shut until the returned guard drops: only granted turns run meanwhile.
    pub fn shut(self: &Arc<Self>) -> Shut {
        self.state.lock().unwrap().open = false;
        Shut(Arc::clone(self))
    }

    pub fn wrap(self: &Arc<Self>, inner: BoxFuture<'static, ()>) -> BoxFuture<'static, ()> {
        let id = {
            let mut state = self.state.lock().unwrap();
            state.next += 1;
            let id = state.next;
            state.slots.insert(id, Slot { ready: true, outer: None });
            id
        };
        Box::pin(Gated { id, gate: Arc::clone(self), inner })
    }

    /// The tasks a turn would move, in spawn order.
    pub fn ready(&self) -> Vec<u64> {
        let state = self.state.lock().unwrap();
        if state.open {
            return Vec::new();
        }
        state.slots.iter().filter(|(_, slot)| slot.ready).map(|(&id, _)| id).collect()
    }

    /// Give `id` its turn: it runs at its tokio task's next poll.
    pub fn grant(&self, id: u64) {
        let outer = {
            let mut state = self.state.lock().unwrap();
            state.granted = Some(id);
            state.slots.get(&id).and_then(|slot| slot.outer.clone())
        };
        if let Some(outer) = outer {
            outer.wake();
        }
    }

    /// The granted turn is not taken yet.
    pub fn granting(&self) -> bool {
        self.state.lock().unwrap().granted.is_some()
    }

    pub fn open(&self) {
        let woken: Vec<Waker> = {
            let mut state = self.state.lock().unwrap();
            if std::mem::replace(&mut state.open, true) {
                return;
            }
            state.slots.values().filter(|s| s.ready).filter_map(|s| s.outer.clone()).collect()
        };
        for outer in woken {
            outer.wake();
        }
    }

    /// Resolves once a task became ready since the last time it did.
    pub async fn woken(&self) {
        self.woken.notified().await
    }
}

pub struct Shut(Arc<Gate>);

impl Drop for Shut {
    fn drop(&mut self) {
        self.0.open();
    }
}

struct Gated {
    id: u64,
    gate: Arc<Gate>,
    inner: BoxFuture<'static, ()>,
}

impl Future for Gated {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        {
            let mut state = this.gate.state.lock().unwrap();
            let open = state.open;
            let granted = state.granted == Some(this.id);
            let Some(slot) = state.slots.get_mut(&this.id) else { return Poll::Ready(()) };
            slot.outer = Some(cx.waker().clone());
            if !open && !granted {
                return Poll::Pending;
            }
            slot.ready = false;
            if granted {
                state.granted = None;
            }
        }
        let waker =
            Waker::from(Arc::new(TaskWaker { gate: Arc::downgrade(&this.gate), id: this.id }));
        this.inner.as_mut().poll(&mut Context::from_waker(&waker))
    }
}

impl Drop for Gated {
    fn drop(&mut self) {
        let mut state = self.gate.state.lock().unwrap();
        state.slots.remove(&self.id);
        if state.granted == Some(self.id) {
            state.granted = None;
        }
    }
}

struct TaskWaker {
    gate: Weak<Gate>,
    id: u64,
}

impl Wake for TaskWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let Some(gate) = self.gate.upgrade() else { return };
        let outer = {
            let mut state = gate.state.lock().unwrap();
            let open = state.open;
            let Some(slot) = state.slots.get_mut(&self.id) else { return };
            slot.ready = true;
            if open { slot.outer.clone() } else { None }
        };
        match outer {
            Some(outer) => outer.wake(),
            None => gate.woken.notify_one(),
        }
    }
}

/// The scheduler against small programs whose legal orders are known (RC "Partial-order
/// sampling": scheduler correctness is tested on small programs; benchmarks do not establish it).
/// Each program is a few sources' events; what is checked is the order the queue delivers them.
#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use lk_agents_voice::agent_session::{Command, Origin, Sent};

    use super::*;

    /// An event from `origin`'s source, named `name`.
    fn event(origin: Origin, name: &str) -> Event {
        Event::Input(Input::Host(Sent {
            origin,
            command: Command::SetStaticKeyterms(vec![name.to_owned()]),
        }))
    }

    fn name(next: &Next) -> String {
        match next {
            Next::Event(Event::Input(Input::Host(Sent {
                command: Command::SetStaticKeyterms(names),
                ..
            }))) => names[0].clone(),
            Next::Frames(frames) => {
                let ids: Vec<String> =
                    frames.iter().map(|f| format!("f{}", f.samples_per_channel)).collect();
                ids.join("+")
            }
            Next::Tasks => "pool".into(),
            Next::Task(id) => format!("task{id}"),
            Next::Event(_) => unreachable!("only the program's events are queued"),
        }
    }

    fn tool(call: &str) -> Origin {
        Origin::Tool { call_id: call.to_owned() }
    }

    /// Two sources of two events each: `a1 a2` from the host, `b1 b2` from a tool.
    fn two_by_two(queue: &mut Queue) {
        for (origin, name) in
            [(Origin::Host, "a1"), (tool("b"), "b1"), (Origin::Host, "a2"), (tool("b"), "b2")]
        {
            queue.push(event(origin, name));
        }
    }

    /// Every event delivered, the pool offered `tasks` turns along the way.
    fn deliver_all(queue: &mut Queue, mut tasks: u32) -> Vec<String> {
        let mut out = Vec::new();
        while let Some(next) = queue.next(tasks > 0, &[]) {
            if matches!(next, Next::Tasks) {
                tasks -= 1;
            }
            out.push(name(&next));
        }
        out
    }

    /// A microphone frame, told apart by its sample count.
    fn frame(n: u32) -> AudioFrame {
        AudioFrame {
            data: Arc::from(vec![0u8; 2 * n as usize]),
            sample_rate: 48_000,
            num_channels: 1,
            samples_per_channel: n,
        }
    }

    /// The frames that arrive between two steps are one step, in their order; a frame after it
    /// is the next batch, never ahead of it. Under POS the batch races the events; in arrival
    /// order it goes first, as the production loop takes frames before its queue.
    #[test]
    fn frames_are_scheduled_batches_in_order() {
        let mut positions = BTreeSet::new();
        for policy in Policy::ALL {
            for seed in 0..256 {
                let mut queue = Queue::new(seed, policy);
                queue.push(event(Origin::Host, "a1"));
                queue.push_frame(frame(1));
                queue.push(event(tool("b"), "b1"));
                queue.push_frame(frame(2));
                let mut order = Vec::new();
                while order.len() < 3 {
                    order.push(name(&queue.next(false, &[]).unwrap()));
                }
                queue.push_frame(frame(3));
                order.push(name(&queue.next(false, &[]).unwrap()));
                assert!(order.contains(&"f1+f2".to_owned()), "{policy:?} seed {seed}: {order:?}");
                assert_eq!(order[3], "f3", "{policy:?} seed {seed}: a later frame went first");
                let at = order.iter().position(|o| o == "f1+f2").unwrap();
                if policy == Policy::Arrival {
                    assert_eq!(at, 0, "seed {seed}: arrival order put events before frames");
                } else {
                    positions.insert(at);
                }
            }
        }
        assert_eq!(positions, BTreeSet::from([0, 1, 2]), "POS never raced frames with events");
    }

    /// A seed's dispatch sequence replays exactly, and its folded order says so; another seed's
    /// differs.
    #[test]
    fn a_seed_replays_its_dispatches() {
        let program = |seed| {
            let mut queue = Queue::new(seed, Policy::TaskBridge);
            two_by_two(&mut queue);
            let mut order = Vec::new();
            for step in 0..6 {
                let ready: &[u64] = if step % 2 == 0 { &[1, 2] } else { &[] };
                if let Some(next) = queue.next(step == 3, ready) {
                    order.push(name(&next));
                }
            }
            order.extend(deliver_all(&mut queue, 0));
            (order, queue.order())
        };
        for seed in 0..64 {
            assert_eq!(program(seed), program(seed), "seed {seed} did not replay");
        }
        let distinct: BTreeSet<u64> = (0..64).map(|seed| program(seed).1).collect();
        assert!(distinct.len() > 1, "every seed dispatched alike");
    }

    /// A full trail that repeats one cycle names it; one that does not, or is not full yet, is
    /// no evidence of a livelock (the run's ending is then `Budget`).
    #[test]
    fn the_trail_names_a_cycle_only_when_it_repeats() {
        let trail = |kinds: &[&'static str], n: usize| Trail {
            last: (0..n as u64).map(|i| (i, kinds[i as usize % kinds.len()])).collect(),
            pending: BTreeMap::new(),
        };
        assert_eq!(trail(&["Tasks"], TRAIL).cycle(), Some(vec!["Tasks"]));
        assert_eq!(trail(&["Gen", "Tasks", "Wake"], TRAIL).cycle().map(|c| c.len()), Some(3));
        assert_eq!(trail(&["Tasks"], TRAIL - 1).cycle(), None, "a trail not yet full");
        let mut broken = trail(&["Gen", "Tasks"], TRAIL);
        broken.last[3].1 = "Wake";
        assert_eq!(broken.cycle(), None, "one step out of the cycle");
    }

    /// Transitional (retires with the bridge): a granted task takes exactly its turn, and a
    /// gated task that is dropped (cancelled) leaves the ready set and any grant it held.
    #[test]
    fn bridge_grants_one_turn_and_forgets_cancelled_tasks() {
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        runtime.block_on(async {
            let gate = Arc::new(Gate::new());
            let polls = Arc::new(Mutex::new(Vec::new()));
            let task = |id: u32| {
                let polls = Arc::clone(&polls);
                let mut turns = 0;
                gate.wrap(Box::pin(std::future::poll_fn(move |cx| {
                    turns += 1;
                    polls.lock().unwrap().push(id);
                    cx.waker().wake_by_ref();
                    if turns == 3 { Poll::Ready(()) } else { Poll::Pending }
                })))
            };
            let _shut = gate.shut();
            let one = tokio::spawn(task(1));
            let two = tokio::spawn(task(2));
            tokio::task::yield_now().await;
            assert_eq!(gate.ready(), [1, 2]);
            assert!(polls.lock().unwrap().is_empty(), "a shut gate ran a task");
            gate.grant(2);
            while gate.granting() {
                tokio::task::yield_now().await;
            }
            assert_eq!(*polls.lock().unwrap(), [2], "the grant ran another task or two turns");
            gate.grant(1);
            one.abort();
            let _ = one.await;
            assert!(!gate.granting(), "a cancelled task kept its grant");
            assert_eq!(gate.ready(), [2]);
            drop(two);
        });
    }
}
