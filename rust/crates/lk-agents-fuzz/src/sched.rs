//! The scheduler every simulation shares: what is pending, and which enabled step runs next.
//!
//! Two kinds of steps compete. A queued item comes from a source and keeps that source's order:
//! a reordering no single sender could produce is never delivered. A process (a task pool, a
//! worker, a client) is enabled when its caller says it is ready, and steps as often as it is
//! chosen. The scheduler knows the sources and processes and nothing of what a step reads or
//! writes.
//!
//! [`Sampling::Arrival`] is the production order: the oldest queued item, a source the loop polls
//! first ahead of the rest, then the first ready process. [`Sampling::BasicPos`] is basic
//! partial-order sampling (Yuan, Yang and Gu, CAV 2018, Algorithm 2), as gosim implements it
//! (livekit/gosim#28): every queued item and every process draws an independent priority, the
//! enabled step with the highest runs, and a process that ran draws a new one; no footprints and
//! no race-based reassignment. Its tests are the small programs at the
//! bottom.
use std::collections::BTreeMap;
use std::convert::Infallible;

use fastrand::Rng;

/// How the next step is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampling {
    Arrival,
    BasicPos,
}

/// Who sent a queued item.
pub trait Source: PartialEq {
    /// The production loop polls this source before the others: in arrival order its oldest item
    /// goes before anything that arrived earlier.
    fn polled_first(&self) -> bool {
        false
    }
}

/// A simulation of processes only: nothing is queued.
impl Source for Infallible {}

/// What runs next.
pub enum Step<T> {
    /// A queued item, with its arrival number.
    Item { arrival: u64, item: T },
    /// One step of a ready process.
    Process(u64),
}

/// What a run does next ([`Sched::tick`]).
pub enum Tick<T> {
    /// An enabled step.
    Step(Step<T>),
    /// Nothing is enabled: virtual time moves to this instant, the earliest anything is due.
    Advance(u64),
    /// Nothing is enabled and nothing will be due: the run is over.
    Quiet,
}

/// gosim's starvation bound: a step that held one priority for this many dispatches draws a new
/// one, so a source that keeps winning cannot hold another back forever.
const STARVE: u64 = 10_000;

#[derive(Clone, Copy)]
struct Priority {
    value: u64,
    drawn_at: u64,
}

struct Queued<S, T> {
    source: S,
    arrival: u64,
    priority: Priority,
    item: T,
}

pub struct Sched<S, T> {
    sampling: Sampling,
    rng: Rng,
    pending: Vec<Queued<S, T>>,
    arrivals: u64,
    dispatches: u64,
    processes: BTreeMap<u64, Priority>,
}

impl<S: Source, T> Sched<S, T> {
    /// `processes` draw their priorities now, in order; any other process draws one the first
    /// time it is ready.
    pub fn new(sampling: Sampling, rng: Rng, processes: &[u64]) -> Self {
        let mut sched = Sched {
            sampling,
            rng,
            pending: Vec::new(),
            arrivals: 0,
            dispatches: 0,
            processes: BTreeMap::new(),
        };
        for &id in processes {
            let priority = sched.draw();
            sched.processes.insert(id, priority);
        }
        sched
    }

    /// Queues `item` from `source`; its arrival number.
    pub fn push(&mut self, source: S, item: T) -> u64 {
        let arrival = self.arrivals;
        self.arrivals += 1;
        let priority = self.draw();
        self.pending.push(Queued { source, arrival, priority, item });
        arrival
    }

    /// The oldest item still pending from `source`, to add to.
    pub fn pending_mut(&mut self, source: &S) -> Option<&mut T> {
        self.pending.iter_mut().find(|q| q.source == *source).map(|q| &mut q.item)
    }

    /// The next step among the pending items' heads and the `ready` processes (in the caller's
    /// order, which arrival order follows), or nothing when neither has one.
    pub fn next(&mut self, ready: &[u64]) -> Option<Step<T>> {
        self.dispatches += 1;
        let heads: Vec<usize> = (0..self.pending.len())
            .filter(|&i| {
                let q = &self.pending[i];
                !self.pending.iter().any(|o| o.source == q.source && o.arrival < q.arrival)
            })
            .collect();
        for &id in ready {
            if !self.processes.contains_key(&id) {
                let priority = self.draw();
                self.processes.insert(id, priority);
            }
        }
        let pick = match self.sampling {
            Sampling::Arrival => heads
                .iter()
                .copied()
                .min_by_key(|&i| (!self.pending[i].source.polled_first(), self.pending[i].arrival))
                .map(Ok)
                .or(ready.first().map(|&id| Err(id))),
            Sampling::BasicPos => {
                for &i in &heads {
                    if self.starved(self.pending[i].priority) {
                        self.pending[i].priority = self.draw();
                    }
                }
                for &id in ready {
                    if self.starved(self.processes[&id]) {
                        let priority = self.draw();
                        self.processes.insert(id, priority);
                    }
                }
                let item = heads.iter().copied().max_by_key(|&i| self.pending[i].priority.value);
                let mut best = item.map(|i| (self.pending[i].priority.value, Ok(i)));
                for &id in ready {
                    let value = self.processes[&id].value;
                    if best.is_none_or(|(b, _)| value > b) {
                        best = Some((value, Err(id)));
                    }
                }
                best.map(|(_, pick)| pick)
            }
        };
        Some(match pick? {
            Ok(i) => {
                let queued = self.pending.remove(i);
                Step::Item { arrival: queued.arrival, item: queued.item }
            }
            Err(id) => {
                // Its next step is a new event: a new priority, as a goroutine's in gosim.
                let priority = self.draw();
                self.processes.insert(id, priority);
                Step::Process(id)
            }
        })
    }

    /// The next step, or where virtual time goes when none is enabled: virtual time moves only
    /// when no step is enabled, and then exactly to the earliest instant something is `due` (a
    /// fake's next report, a core's wake), never past it (RC "Partial-order sampling"; K-11).
    pub fn tick(&mut self, now: u64, ready: &[u64], due: impl IntoIterator<Item = u64>) -> Tick<T> {
        match self.next(ready) {
            Some(step) => Tick::Step(step),
            None => match due.into_iter().min() {
                // what was due by now was delivered already: the next instant is after it
                Some(at) => Tick::Advance(at.max(now + 1)),
                None => Tick::Quiet,
            },
        }
    }

    /// Everything still pending, in arrival order.
    pub fn drain(&mut self) -> Vec<T> {
        self.pending.sort_by_key(|q| q.arrival);
        self.pending.drain(..).map(|q| q.item).collect()
    }

    /// How many steps were chosen so far.
    pub fn dispatches(&self) -> u64 {
        self.dispatches
    }

    /// Whether nothing is queued.
    pub fn idle(&self) -> bool {
        self.pending.is_empty()
    }

    fn draw(&mut self) -> Priority {
        Priority { value: self.rng.u64(..), drawn_at: self.dispatches }
    }

    fn starved(&self, priority: Priority) -> bool {
        self.dispatches - priority.drawn_at >= STARVE
    }
}

/// The scheduler against small programs whose legal orders are known (RC "Partial-order
/// sampling": scheduler correctness is tested on small programs; benchmarks do not establish it).
/// What the fakes will report, and when: an item reaches the scheduler once it is due.
pub struct Timeline<S, E> {
    items: Vec<(u64, u64, S, E)>,
    seq: u64,
}

impl<S: Source, E> Default for Timeline<S, E> {
    fn default() -> Self {
        Timeline { items: Vec::new(), seq: 0 }
    }
}

impl<S: Source, E> Timeline<S, E> {
    /// `event` from `source` is reported at `due` ms.
    pub fn at(&mut self, due: u64, source: S, event: E) {
        self.items.push((due, self.seq, source, event));
        self.seq += 1;
    }

    /// Everything due by `now` reaches the scheduler, in the order it fell due.
    pub fn release(&mut self, now: u64, sched: &mut Sched<S, E>) {
        let mut due: Vec<(u64, u64, S, E)> = Vec::new();
        let mut i = 0;
        while i < self.items.len() {
            if self.items[i].0 <= now {
                due.push(self.items.swap_remove(i));
            } else {
                i += 1;
            }
        }
        due.sort_by_key(|(at, seq, ..)| (*at, *seq));
        for (_, _, source, event) in due {
            sched.push(source, event);
        }
    }

    /// When the next report falls due.
    pub fn next_due(&self) -> Option<u64> {
        self.items.iter().map(|(at, ..)| *at).min()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[derive(PartialEq)]
    struct From(&'static str);

    impl Source for From {
        fn polled_first(&self) -> bool {
            self.0 == "first"
        }
    }

    type Program = Sched<From, &'static str>;

    fn program(seed: u64, sampling: Sampling) -> Program {
        Sched::new(sampling, Rng::with_seed(seed), &[])
    }

    /// Two sources of two items each: `a1 a2` from `a`, `b1 b2` from `b`.
    fn two_by_two(sched: &mut Program) {
        for (source, item) in [("a", "a1"), ("b", "b1"), ("a", "a2"), ("b", "b2")] {
            sched.push(From(source), item);
        }
    }

    /// Every step until nothing is enabled, the processes in `ready` each enabled for `turns`
    /// steps.
    fn run(sched: &mut Program, ready: &[u64], turns: u32) -> Vec<String> {
        let mut left: BTreeMap<u64, u32> = ready.iter().map(|&id| (id, turns)).collect();
        let mut out = Vec::new();
        loop {
            let enabled: Vec<u64> = ready.iter().copied().filter(|id| left[id] > 0).collect();
            match sched.next(&enabled) {
                Some(Step::Item { item, .. }) => out.push(item.to_owned()),
                Some(Step::Process(id)) => {
                    *left.get_mut(&id).unwrap() -= 1;
                    out.push(format!("p{id}"));
                }
                None => return out,
            }
        }
    }

    /// The legal orders of `two_by_two`: each source's items in its own order.
    fn legal(order: &[String]) -> bool {
        let position = |n: &str| order.iter().position(|o| o == n).unwrap();
        position("a1") < position("a2") && position("b1") < position("b2")
    }

    const BOTH: [Sampling; 2] = [Sampling::Arrival, Sampling::BasicPos];

    #[test]
    fn each_source_keeps_its_order() {
        for sampling in BOTH {
            for seed in 0..512 {
                let mut sched = program(seed, sampling);
                two_by_two(&mut sched);
                let order = run(&mut sched, &[], 0);
                assert_eq!(order.len(), 4, "{sampling:?} seed {seed}: {order:?}");
                assert!(legal(&order), "{sampling:?} seed {seed} reordered a source: {order:?}");
            }
        }
    }

    /// BasicPOS reaches every legal interleaving of two independent sources; arrival order only
    /// the one the channel produces.
    #[test]
    fn independent_orders_are_all_reached() {
        let reached = |sampling| -> BTreeSet<Vec<String>> {
            (0..512)
                .map(|seed| {
                    let mut sched = program(seed, sampling);
                    two_by_two(&mut sched);
                    run(&mut sched, &[], 0)
                })
                .collect()
        };
        // a1 a2 b1 b2, a1 b1 a2 b2, a1 b1 b2 a2, b1 a1 a2 b2, b1 a1 b2 a2, b1 b2 a1 a2
        assert_eq!(reached(Sampling::BasicPos).len(), 6);
        let arrival = reached(Sampling::Arrival);
        assert_eq!(arrival.into_iter().collect::<Vec<_>>(), [["a1", "b1", "a2", "b2"]]);
    }

    /// A source the loop polls first goes ahead in arrival order, whatever arrived before it.
    #[test]
    fn arrival_polls_its_first_source_first() {
        let mut sched = program(0, Sampling::Arrival);
        sched.push(From("a"), "a1");
        sched.push(From("first"), "f1");
        assert_eq!(run(&mut sched, &[], 0), ["f1", "a1"]);
    }

    /// A process's step races the pending items: under BasicPOS it runs before, between and after
    /// them across seeds; in arrival order only once nothing is queued.
    #[test]
    fn process_steps_race_pending_items() {
        let positions = |sampling| -> BTreeSet<usize> {
            (0..512)
                .map(|seed| {
                    let mut sched = program(seed, sampling);
                    two_by_two(&mut sched);
                    let order = run(&mut sched, &[7], 1);
                    assert!(legal(&order), "{sampling:?} seed {seed}: {order:?}");
                    order.iter().position(|o| o == "p7").unwrap()
                })
                .collect()
        };
        assert_eq!(positions(Sampling::BasicPos), BTreeSet::from([0, 1, 2, 3, 4]));
        assert_eq!(positions(Sampling::Arrival), BTreeSet::from([4]));
    }

    /// Only a ready process steps: one that stopped being ready (cancelled, done, not due) is
    /// never picked, and nothing is when nothing is enabled.
    #[test]
    fn only_ready_processes_step() {
        for sampling in BOTH {
            for seed in 0..256 {
                let mut sched = program(seed, sampling);
                sched.push(From("a"), "a1");
                let mut ready: &[u64] = &[1, 2];
                loop {
                    match sched.next(ready) {
                        Some(Step::Process(id)) => {
                            assert!(ready.contains(&id), "{sampling:?} seed {seed}: p{id}");
                        }
                        Some(Step::Item { .. }) => break,
                        None => unreachable!("p2 is ready"),
                    }
                    // p1 is cancelled after the first step
                    ready = &[2];
                }
                assert!(sched.next(&[]).is_none(), "a step with nothing enabled");
            }
        }
    }

    /// Both orders of two processes are reached under BasicPOS; arrival order follows the
    /// caller's order of the ready ones.
    #[test]
    fn processes_are_ordered_by_the_seed() {
        let firsts = |sampling| -> BTreeSet<u64> {
            (0..128)
                .filter_map(|seed| match program(seed, sampling).next(&[1, 2]) {
                    Some(Step::Process(id)) => Some(id),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(firsts(Sampling::BasicPos), BTreeSet::from([1, 2]));
        assert_eq!(firsts(Sampling::Arrival), BTreeSet::from([1]));
    }

    /// A process whose priority lost to another's every redraw draws a new one after
    /// [`STARVE`] dispatches, so an always-ready process is never held back for good: on every
    /// seed it steps within twice the bound.
    #[test]
    fn a_ready_process_steps_within_the_starvation_bound() {
        for seed in 0..64 {
            let mut sched = program(seed, Sampling::BasicPos);
            let mut last = BTreeMap::<u64, u64>::new();
            let mut longest = 0;
            for step in 0..3 * STARVE {
                if let Some(Step::Process(id)) = sched.next(&[1, 2]) {
                    longest = longest.max(step - last.get(&id).copied().unwrap_or(0));
                    last.insert(id, step);
                }
            }
            assert!(longest < 2 * STARVE, "seed {seed}: a process waited {longest} dispatches");
            assert_eq!(last.len(), 2, "seed {seed}: {last:?}");
        }
    }

    /// `drain` returns what is pending in arrival order, and leaves nothing.
    type Ticker = fn(&mut Program, u64, &[u64], &[u64]) -> Tick<&'static str>;

    /// What each time-advance case breaks under `tick`: ready work with a future instant due
    /// (queued, or a ready process), an idle run with an arrival before the wake, a wake before
    /// the arrival, both at one instant, nothing due.
    fn time_violations(tick: Ticker) -> Vec<String> {
        let mut out = Vec::new();
        for sampling in BOTH {
            for seed in 0..64 {
                let mut queued = program(seed, sampling);
                queued.push(From("a"), "a1");
                if !matches!(tick(&mut queued, 10, &[], &[40]), Tick::Step(_)) {
                    out.push(format!("{sampling:?} {seed}: time moved over a queued item"));
                }
                let mut process = program(seed, sampling);
                if !matches!(tick(&mut process, 10, &[7], &[40]), Tick::Step(_)) {
                    out.push(format!("{sampling:?} {seed}: time moved over a ready process"));
                }
            }
        }
        for (due, earliest) in [([25, 90], 25), ([90, 25], 25), ([30, 30], 30)] {
            let mut idle = program(0, Sampling::BasicPos);
            match tick(&mut idle, 10, &[], &due) {
                Tick::Advance(at) if at == earliest => {}
                Tick::Advance(at) => out.push(format!("{due:?}: advanced to {at}, not {earliest}")),
                _ => out.push(format!("{due:?}: an idle run did not advance")),
            }
        }
        if !matches!(tick(&mut program(0, Sampling::Arrival), 10, &[], &[]), Tick::Quiet) {
            out.push("nothing due, yet the run went on".into());
        }
        out
    }

    #[test]
    fn enabled_step_blocks_time_advance_and_idle_advances_to_next_event() {
        let real: Ticker = |s, now, ready, due| s.tick(now, ready, due.iter().copied());
        assert_eq!(time_violations(real), Vec::<String>::new());
        // mutants: each must be told apart from the scheduler
        let jumps_while_enabled: Ticker = |s, now, ready, due| match due.iter().min() {
            Some(&at) => Tick::Advance(at.max(now + 1)),
            None => s.tick(now, ready, []),
        };
        let skips_the_earliest: Ticker =
            |s, now, ready, due| s.tick(now, ready, due.iter().max().copied());
        let overshoots: Ticker = |s, now, ready, due| s.tick(now, ready, due.iter().map(|d| d + 1));
        let never_ends: Ticker = |s, now, ready, due| match s.tick(now, ready, due.iter().copied())
        {
            Tick::Quiet => Tick::Advance(now + 1),
            other => other,
        };
        for (name, mutant) in [
            ("jumps while enabled", jumps_while_enabled),
            ("skips the earliest", skips_the_earliest),
            ("overshoots", overshoots),
            ("never ends", never_ends),
        ] {
            assert!(!time_violations(mutant).is_empty(), "the mutant that {name} passed");
        }
    }

    #[test]
    fn drain_is_in_arrival_order() {
        let mut sched = program(3, Sampling::BasicPos);
        two_by_two(&mut sched);
        let first = sched.next(&[]);
        assert!(matches!(first, Some(Step::Item { .. })));
        let rest = sched.drain();
        assert_eq!(rest.len(), 3);
        assert!(sched.idle());
        let all = ["a1", "b1", "a2", "b2"];
        let expected: Vec<&str> = all.into_iter().filter(|i| rest.contains(i)).collect();
        assert_eq!(rest, expected);
    }
}
