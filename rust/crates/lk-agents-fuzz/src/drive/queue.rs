//! The driver's queue, owned by the case: what every source sent, delivered one event at a time in
//! the order the run's policy chooses.
//!
//! In production the queue is a channel and the order is arrival order. Here the case collects
//! what reached the channel and picks what runs next, so an interleaving the channel would only
//! produce under load is one a seed can choose. Each source keeps its own order: a reordering no
//! single sender could produce is never delivered.
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::future::Future;
use std::mem::Discriminant;
use std::pin::Pin;
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll, Wake, Waker};

use fastrand::Rng;
use lk_agents_abi::{BoxFuture, resources::ResourceId};
use lk_agents_voice::agent_session::{Event, Input};

use super::objects::{Access, accesses, races};

/// How the next step is chosen, drawn per seed and carried by the case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Policy {
    /// The channel's own order: the driver drains what is queued, and the tasks it spawned run
    /// when the queue is empty.
    #[default]
    Arrival,
    /// Basic partial-order sampling (Yuan, Yang and Gu, CAV 2018, Algorithm 2), as gosim
    /// implements it (livekit/gosim#28). The processes are each event source and the pool of
    /// tasks the driver spawned. Every event draws an independent priority; the enabled step with
    /// the highest priority runs: a source's oldest pending event, or one task turn, after which
    /// the pool draws a new priority.
    BasicPos,
    /// Partial-order sampling (Algorithm 4 with POS*'s read relaxation): BasicPOS, and after each
    /// step every pending head racing with it draws a new priority. Racing is per object touched
    /// ([`accesses`]), not per process: one core handles every event, but two events of
    /// different speeches, or a data-plane command and a speech's tool start, touch nothing in
    /// common. The task pool's steps are unknown, so the pool races with every event.
    FullPos,
    /// Full POS with the task bridge ([`Gate`]): every task the driver spawns is a process of its
    /// own, with its own priority, rather than one turn the pool shares, so the order of two
    /// tasks is sampled too. A task touches what it reports: the accesses of the events it sent
    /// in its turns; before its first report it races with everything. What the fakes spawn
    /// outside the session's spawner is the pool.
    TaskPos,
}

impl Policy {
    pub const ALL: [Policy; 4] =
        [Policy::Arrival, Policy::BasicPos, Policy::FullPos, Policy::TaskPos];

    /// `VOICE_POLICY=FullPos`: one policy by its name.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| format!("{p:?}") == name)
    }

    /// A seed's own policy: the task bridge for half the seeds, full POS without it for a
    /// quarter, and arrival order for a quarter, which is still what reaches some bugs first (the
    /// paper and gosim found the same). `tests/pos.rs bench` over the known bugs: on a failing
    /// case's schedules the geometric-mean hit ratio is 0.07 under arrival, 0.30 under BasicPOS,
    /// 0.33 under full POS and 0.66 with the bridge, for about a quarter more time per seed.
    pub fn of(schedule: u64) -> Self {
        match Rng::with_seed(schedule ^ POLICY_STREAM).u8(0..4) {
            0 => Policy::Arrival,
            1 => Policy::FullPos,
            _ => Policy::TaskPos,
        }
    }
}

/// The policy's own random stream off the schedule seed: the bytes `policy`.
const POLICY_STREAM: u64 = 0x0070_6f6c_6963_79;

/// gosim's starvation bound: a step that held one priority for this many dispatches draws a new
/// one, so a source that keeps winning cannot hold another back forever.
const STARVE: u64 = 10_000;

/// What runs next.
pub enum Next {
    Event(Event),
    /// One turn for the driver's ready tasks.
    Tasks,
    /// One turn for one bridged task.
    Task(u64),
}

#[derive(Clone, Copy)]
enum Pick {
    Event(usize),
    Task(u64),
    Pool,
}

/// A bridged task: its priority and what it reported touching.
struct Proc {
    priority: Priority,
    accesses: Option<Vec<Access>>,
}

pub struct Queue {
    policy: Policy,
    rng: Rng,
    pending: Vec<Queued>,
    arrivals: u64,
    dispatches: u64,
    /// The task pool's priority.
    tasks: Priority,
    /// Events delivered, and the commutation check's pair: the event owed next, and whether the
    /// last one delivered belongs to it.
    delivered: u64,
    owed: Option<u64>,
    paired: bool,
    /// The bridged tasks, the one whose turn's reports are being drained, and the one whose
    /// turn is still to be reassigned for.
    procs: BTreeMap<u64, Proc>,
    reporter: Option<u64>,
    stepped: Option<u64>,
}

#[derive(Clone, Copy)]
struct Priority {
    value: u64,
    drawn_at: u64,
}

struct Queued {
    source: Source,
    arrival: u64,
    priority: Priority,
    accesses: Vec<Access>,
    event: Event,
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
}

impl Source {
    fn of(event: &Event) -> Self {
        match event {
            Event::Input(Input::Host(sent)) => Source::Host(format!("{:?}", sent.origin)),
            Event::Input(Input::Stt(tagged)) => Source::Stt(tagged.stream),
            Event::Input(Input::Vad(_)) => Source::Leg("vad"),
            Event::Input(Input::Eot(..)) => Source::Leg("eot"),
            Event::Input(Input::Overlap(_)) => Source::Leg("interruption"),
            Event::Input(input @ Input::ChannelEnded(end)) => match (end.leg(), end.sink()) {
                ("stt", Some(stream)) => Source::Stt(stream),
                (leg @ ("vad" | "eot" | "interruption"), _) => Source::Leg(leg),
                _ => Source::Kind(std::mem::discriminant(input)),
            },
            Event::Input(other) => Source::Kind(std::mem::discriminant(other)),
            other => Source::Event(std::mem::discriminant(other)),
        }
    }
}

impl Queue {
    pub fn new(schedule: u64, policy: Policy) -> Self {
        let mut rng = Rng::with_seed(schedule ^ POLICY_STREAM.rotate_left(32));
        let tasks = Priority { value: rng.u64(..), drawn_at: 0 };
        Queue {
            policy,
            rng,
            pending: Vec::new(),
            arrivals: 0,
            dispatches: 0,
            tasks,
            delivered: 0,
            owed: None,
            paired: false,
            procs: BTreeMap::new(),
            reporter: None,
            stepped: None,
        }
    }

    pub fn push(&mut self, event: Event) {
        let arrival = self.arrivals;
        self.arrivals += 1;
        let priority = self.draw();
        let (source, accesses) = (Source::of(&event), accesses(&event));
        if let Some(proc) = self.reporter.and_then(|id| self.procs.get_mut(&id)) {
            let theirs = proc.accesses.get_or_insert_with(Vec::new);
            theirs.extend(accesses.iter().copied());
            theirs.sort();
            theirs.dedup();
        }
        self.pending.push(Queued { source, arrival, priority, accesses, event });
    }

    /// What the bridged task last given a turn reported, as the run drains the channel after it:
    /// what it touches.
    pub fn reported(&mut self) {
        self.reporter = None;
    }

    /// The next step: an event, a bridged task's turn (`ready`, under [`Policy::TaskPos`]), a turn
    /// for the pool when `tasks_ready`, or nothing.
    pub fn next(&mut self, tasks_ready: bool, ready: &[u64]) -> Option<Next> {
        self.dispatches += 1;
        self.paired = false;
        self.reporter = None;
        if let Some(owed) = self.owed.take()
            && let Some(i) = self.pending.iter().position(|q| q.arrival == owed)
        {
            return Some(self.deliver(i, true));
        }
        let heads: Vec<usize> = (0..self.pending.len())
            .filter(|&i| {
                let q = &self.pending[i];
                !self.pending.iter().any(|o| o.source == q.source && o.arrival < q.arrival)
            })
            .collect();
        for &id in ready {
            if !self.procs.contains_key(&id) {
                let priority = self.draw();
                self.procs.insert(id, Proc { priority, accesses: None });
            }
        }
        // A task's turn is reassigned for once its reports are in.
        if let Some(id) = self.stepped.take() {
            let theirs = self.procs.get(&id).and_then(|p| p.accesses.clone());
            self.reassign(theirs.as_deref(), &heads, ready, Some(id));
        }
        let pick = match self.policy {
            Policy::Arrival => heads
                .iter()
                .copied()
                .min_by_key(|&i| self.pending[i].arrival)
                .map(Pick::Event)
                .or(tasks_ready.then_some(Pick::Pool)),
            Policy::BasicPos | Policy::FullPos | Policy::TaskPos => {
                for &i in &heads {
                    if self.starved(self.pending[i].priority) {
                        self.pending[i].priority = self.draw();
                    }
                }
                if tasks_ready && self.starved(self.tasks) {
                    self.tasks = self.draw();
                }
                for &id in ready {
                    if self.starved(self.procs[&id].priority) {
                        let priority = self.draw();
                        self.procs.get_mut(&id).unwrap().priority = priority;
                    }
                }
                let event = heads.iter().copied().max_by_key(|&i| self.pending[i].priority.value);
                let mut best = event.map(|i| (self.pending[i].priority.value, Pick::Event(i)));
                for &id in ready {
                    let value = self.procs[&id].priority.value;
                    if best.is_none_or(|(b, _)| value > b) {
                        best = Some((value, Pick::Task(id)));
                    }
                }
                if tasks_ready && best.is_none_or(|(b, _)| self.tasks.value >= b) {
                    best = Some((self.tasks.value, Pick::Pool));
                }
                best.map(|(_, pick)| pick)
            }
        };
        match pick? {
            Pick::Event(i) => {
                let i = self.commutation_pair(i, &heads);
                if self.reassigns() {
                    let accesses = std::mem::take(&mut self.pending[i].accesses);
                    self.reassign(Some(&accesses), &heads, ready, None);
                }
                Some(self.deliver(i, self.owed.is_some()))
            }
            Pick::Task(id) => {
                // Its next step is a new event: a new priority, as a goroutine's in gosim.
                let priority = self.draw();
                self.procs.get_mut(&id).unwrap().priority = priority;
                (self.stepped, self.reporter) = (Some(id), Some(id));
                Some(Next::Task(id))
            }
            Pick::Pool => {
                // The pool took its step: like a goroutine in gosim, it draws its next priority,
                // and what it did is unknown, so under full POS everything races with it.
                if self.reassigns() {
                    self.reassign(None, &heads, ready, None);
                } else {
                    self.tasks = self.draw();
                }
                Some(Next::Tasks)
            }
        }
    }

    fn reassigns(&self) -> bool {
        matches!(self.policy, Policy::FullPos | Policy::TaskPos)
    }

    /// Algorithm 4's reassignment: every pending step racing with the one just taken (`None`:
    /// touching what nobody knows, so racing with everything) draws a new priority, so the next
    /// race is ordered afresh rather than by what the step outran.
    fn reassign(
        &mut self,
        theirs: Option<&[Access]>,
        heads: &[usize],
        ready: &[u64],
        except: Option<u64>,
    ) {
        for &j in heads {
            if theirs.is_none_or(|a| races(a, &self.pending[j].accesses)) {
                self.pending[j].priority = self.draw();
            }
        }
        for &id in ready.iter().filter(|&&id| Some(id) != except) {
            let racing = match (theirs, &self.procs[&id].accesses) {
                (Some(a), Some(b)) => races(a, b),
                _ => true,
            };
            if racing {
                let priority = self.draw();
                self.procs.get_mut(&id).unwrap().priority = priority;
            }
        }
        // The pool's next step is unknown: it races.
        self.tasks = self.draw();
    }

    fn deliver(&mut self, i: usize, paired: bool) -> Next {
        self.delivered += 1;
        self.paired = paired;
        let event = self.pending.remove(i).event;
        PROBE.with_borrow_mut(|p| {
            p.delivered = self.delivered;
            if paired {
                p.pair.push(kind(&event));
            }
        });
        Next::Event(event)
    }

    /// Under the commutation check, at its chosen delivery: the policy's pick `i` and the oldest
    /// pending head declared independent of it go out back to back, in the order the check says.
    fn commutation_pair(&mut self, i: usize, heads: &[usize]) -> usize {
        let Some(swap) = PROBE.with_borrow(|p| p.swap) else { return i };
        if self.delivered != swap.at {
            return i;
        }
        let picked = self.pending[i].accesses.clone();
        let others: Vec<(usize, bool)> = heads
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| (j, races(&picked, &self.pending[j].accesses)))
            .collect();
        PROBE.with_borrow_mut(|p| {
            p.heads = others.len();
            p.independent = others.iter().filter(|(_, racing)| !racing).count();
        });
        let partner = others
            .iter()
            .filter(|&&(_, racing)| racing == swap.racing)
            .map(|&(j, _)| j)
            .min_by_key(|&j| self.pending[j].arrival);
        let Some(j) = partner else { return i };
        let (first, second) = if swap.flip { (j, i) } else { (i, j) };
        self.owed = Some(self.pending[second].arrival);
        first
    }

    /// Whether the event just delivered is one of the commutation check's pair.
    pub fn paired(&self) -> bool {
        self.paired
    }

    /// Everything still pending, in arrival order.
    pub fn drain(&mut self) -> Vec<Event> {
        self.pending.sort_by_key(|q| q.arrival);
        self.pending.drain(..).map(|q| q.event).collect()
    }

    fn draw(&mut self) -> Priority {
        Priority { value: self.rng.u64(..), drawn_at: self.dispatches }
    }

    fn starved(&self, priority: Priority) -> bool {
        self.dispatches - priority.drawn_at >= STARVE
    }
}

/// The task bridge: under [`Policy::TaskPos`] every future the session's spawner spawns runs
/// behind a gate, polled only when the queue gives that task its turn. Its own waker marks it
/// ready, and wakes the run, which may be waiting on time; tokio only ever sees a task woken to
/// take its granted turn. The gate is shut only while the run steps the session: when the run
/// awaits something itself (a host call, the session's start, a player's close) or the session
/// is gone, what is ready runs as tokio schedules it, as the host's loop outlives the session's.
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

/// The commutation check (`tests/pos.rs`): two events the conflict relation declares independent
/// must leave the same session whichever goes first. Armed per thread, as a simulation runs on
/// the thread that started it.
#[derive(Clone, Copy, Debug)]
pub struct Swap {
    /// The delivery, counted from zero, at which the pair goes out.
    pub at: u64,
    /// The partner first.
    pub flip: bool,
    /// The control: a partner the relation says races instead.
    pub racing: bool,
}

/// What the armed check saw.
#[derive(Default, Debug)]
pub struct Probe {
    pub swap: Option<Swap>,
    pub delivered: u64,
    /// At the chosen delivery: the other pending heads, and how many the relation calls
    /// independent of the policy's pick.
    pub heads: usize,
    pub independent: usize,
    /// The pair's kinds, in delivery order, and the journal steps their two steps wrote.
    pub pair: Vec<String>,
    pub pair_steps: Vec<String>,
    /// The whole journal and the failures, sorted, at the run's end.
    pub journal: Vec<String>,
}

thread_local! {
    static PROBE: RefCell<Probe> = RefCell::default();
}

/// Arm the check for the next run on this thread; `None` only counts deliveries.
pub fn arm(swap: Option<Swap>) {
    PROBE.set(Probe { swap, ..Probe::default() });
}

pub(super) fn armed() -> bool {
    PROBE.with_borrow(|p| p.swap.is_some())
}

/// What the check saw, disarming it.
pub fn take() -> Probe {
    PROBE.take()
}

pub(super) fn note_pair_steps(steps: impl Iterator<Item = String>) {
    PROBE.with_borrow_mut(|p| p.pair_steps.extend(steps));
}

pub(super) fn note_journal(journal: Vec<String>) {
    PROBE.with_borrow_mut(|p| p.journal = journal);
}

/// An event's kind, for the check's report.
pub fn kind(event: &Event) -> String {
    fn head(debug: String) -> String {
        debug.split([' ', '(', '{']).next().unwrap_or_default().to_owned()
    }
    match event {
        Event::Wake => "Wake".into(),
        Event::Forwarder(command) => format!("Forwarder::{}", head(format!("{command:?}"))),
        Event::Input(input) => match input {
            Input::Host(sent) => sent.command.kind().into(),
            Input::Stt(tagged) => format!("Stt::{:?}", tagged.item.kind),
            Input::Vad(ev) => format!("Vad::{:?}", ev.kind),
            Input::Room(_) => "Room".into(),
            Input::RoomIo(ev) => format!("RoomIo::{}", head(format!("{ev:?}"))),
            Input::Gen(ev) => format!("Gen::{}", head(format!("{ev:?}"))),
            Input::Realtime(wake) => {
                use lk_agents_voice::agent_activity::realtime::RealtimeWake;
                match wake {
                    RealtimeWake::Event(ev) => format!("Realtime::{}", head(format!("{ev:?}"))),
                    RealtimeWake::Ended => "Realtime::Ended".into(),
                    RealtimeWake::Closed => "Realtime::Closed".into(),
                    RealtimeWake::ToolsDone { .. } => "Realtime::ToolsDone".into(),
                    RealtimeWake::Heard { .. } => "Realtime::Heard".into(),
                }
            }
            Input::Eot(..) => "Eot".into(),
            Input::Overlap(_) => "Overlap".into(),
            Input::ChannelEnded(_) => "ChannelEnded".into(),
            Input::SttRetried(_) => "SttRetried".into(),
            Input::StreamOpened(_) => "StreamOpened".into(),
            Input::Video(_) => "Video".into(),
            Input::InputStarted { .. } => "InputStarted".into(),
            Input::GateSilenced(_) => "GateSilenced".into(),
            Input::InputCensus(_) => "InputCensus".into(),
        },
    }
}
