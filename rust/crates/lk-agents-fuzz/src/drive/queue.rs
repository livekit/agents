//! The driver's queue, owned by the case: what every source sent, delivered one event at a time in
//! the order the run's policy chooses.
//!
//! In production the queue is a channel and the order is arrival order. Here the case collects
//! what reached the channel and picks what runs next, so an interleaving the channel would only
//! produce under load is one a seed can choose. Each source keeps its own order: a reordering no
//! single sender could produce is never delivered.
use std::mem::Discriminant;

use fastrand::Rng;
use lk_agents_voice::agent_session::Input;

/// How the next step is chosen, drawn per seed and saved with the case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
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
    Event(Input),
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
    input: Input,
}

/// Who sent an event: one host origin, one recognition stream, or one kind of input.
#[derive(PartialEq, Eq)]
enum Source {
    Host(String),
    Stt(u32),
    Kind(Discriminant<Input>),
}

impl Source {
    fn of(input: &Input) -> Self {
        match input {
            Input::Host(sent) => Source::Host(format!("{:?}", sent.origin)),
            Input::Stt(tagged) => Source::Stt(tagged.open),
            other => Source::Kind(std::mem::discriminant(other)),
        }
    }
}

impl Queue {
    pub fn new(schedule: u64, policy: Policy) -> Self {
        let mut rng = Rng::with_seed(schedule ^ POLICY_STREAM.rotate_left(32));
        let tasks = Priority { value: rng.u64(..), drawn_at: 0 };
        Queue { policy, rng, pending: Vec::new(), arrivals: 0, dispatches: 0, tasks }
    }

    pub fn push(&mut self, input: Input) {
        let arrival = self.arrivals;
        self.arrivals += 1;
        let priority = self.draw();
        self.pending.push(Queued { source: Source::of(&input), arrival, priority, input });
    }

    /// The next step: an event, a task turn when `tasks_ready`, or nothing.
    pub fn next(&mut self, tasks_ready: bool) -> Option<Next> {
        self.dispatches += 1;
        let heads: Vec<usize> = (0..self.pending.len())
            .filter(|&i| {
                let q = &self.pending[i];
                !self.pending.iter().any(|o| o.source == q.source && o.arrival < q.arrival)
            })
            .collect();
        let event = match self.policy {
            Policy::Arrival => heads.into_iter().min_by_key(|&i| self.pending[i].arrival),
            Policy::Pos => {
                for &i in &heads {
                    if self.starved(self.pending[i].priority) {
                        self.pending[i].priority = self.draw();
                    }
                }
                if tasks_ready && self.starved(self.tasks) {
                    self.tasks = self.draw();
                }
                let best = heads.into_iter().max_by_key(|&i| self.pending[i].priority.value);
                best.filter(|&i| !tasks_ready || self.pending[i].priority.value > self.tasks.value)
            }
        };
        if let Some(i) = event {
            return Some(Next::Event(self.pending.remove(i).input));
        }
        if !tasks_ready {
            return None;
        }
        // The pool took its step: like a goroutine in gosim, it draws its next priority.
        self.tasks = self.draw();
        Some(Next::Tasks)
    }

    /// Everything still pending, in arrival order.
    pub fn drain(&mut self) -> Vec<Input> {
        self.pending.sort_by_key(|q| q.arrival);
        self.pending.drain(..).map(|q| q.input).collect()
    }

    fn draw(&mut self) -> Priority {
        Priority { value: self.rng.u64(..), drawn_at: self.dispatches }
    }

    fn starved(&self, priority: Priority) -> bool {
        self.dispatches - priority.drawn_at >= STARVE
    }
}
