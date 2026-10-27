//! The driver's queue, owned by the case: what every source sent, delivered one event at a time in
//! the order the run's policy chooses.
//!
//! In production the queue is a channel and the order is arrival order. Here the case collects
//! what reached the channel and picks what runs next, so an interleaving the channel would only
//! produce under load is one a seed can choose. Each source keeps its own order: a reordering no
//! single sender could produce is never delivered.
use std::cell::RefCell;
use std::mem::Discriminant;

use fastrand::Rng;
use lk_agents_abi::resources::ResourceId;
use lk_agents_voice::agent_session::{Event, Input, objects::races};

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
    /// the pool draws a new priority. Full POS re-draws the priorities of events racing with the
    /// one just run; every event here races, since all of them reach one core, so it would
    /// degenerate to a random walk.
    Pos,
}

impl Policy {
    pub fn of(schedule: u64) -> Self {
        if Rng::with_seed(schedule ^ POLICY_STREAM).bool() { Policy::Pos } else { Policy::Arrival }
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
    event: Event,
}

/// Who sent an event: one host origin, one recognition stream, one kind of input, or the decision
/// core's commands to the data plane.
#[derive(PartialEq, Eq)]
enum Source {
    Host(String),
    Stt(ResourceId),
    Kind(Discriminant<Input>),
    Event(Discriminant<Event>),
}

impl Source {
    fn of(event: &Event) -> Self {
        match event {
            Event::Input(Input::Host(sent)) => Source::Host(format!("{:?}", sent.origin)),
            Event::Input(Input::Stt(tagged)) => Source::Stt(tagged.stream),
            Event::Input(other) => Source::Kind(std::mem::discriminant(other)),
            other => Source::Event(std::mem::discriminant(other)),
        }
    }
}

impl Queue {
    pub fn new(schedule: u64, policy: Policy) -> Self {
        let mut rng = Rng::with_seed(schedule ^ POLICY_STREAM.rotate_left(32));
        let tasks = Priority { value: rng.u64(..), drawn_at: 0 };
        let (pending, delivered, owed, paired) = (Vec::new(), 0, None, false);
        Queue { policy, rng, pending, arrivals: 0, dispatches: 0, tasks, delivered, owed, paired }
    }

    pub fn push(&mut self, event: Event) {
        let arrival = self.arrivals;
        self.arrivals += 1;
        let priority = self.draw();
        self.pending.push(Queued { source: Source::of(&event), arrival, priority, event });
    }

    /// The next step: an event, a task turn when `tasks_ready`, or nothing.
    pub fn next(&mut self, tasks_ready: bool) -> Option<Next> {
        self.dispatches += 1;
        self.paired = false;
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
        let event = match self.policy {
            Policy::Arrival => heads.iter().copied().min_by_key(|&i| self.pending[i].arrival),
            Policy::Pos => {
                for &i in &heads {
                    if self.starved(self.pending[i].priority) {
                        self.pending[i].priority = self.draw();
                    }
                }
                if tasks_ready && self.starved(self.tasks) {
                    self.tasks = self.draw();
                }
                let best = heads.iter().copied().max_by_key(|&i| self.pending[i].priority.value);
                best.filter(|&i| !tasks_ready || self.pending[i].priority.value > self.tasks.value)
            }
        };
        if let Some(i) = event {
            let i = self.commutation_pair(i, &heads);
            return Some(self.deliver(i, self.owed.is_some()));
        }
        if !tasks_ready {
            return None;
        }
        // The pool took its step: like a goroutine in gosim, it draws its next priority.
        self.tasks = self.draw();
        Some(Next::Tasks)
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
        let picked = self.pending[i].event.accesses();
        let others: Vec<(usize, bool)> = heads
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| (j, races(&picked, &self.pending[j].event.accesses())))
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
