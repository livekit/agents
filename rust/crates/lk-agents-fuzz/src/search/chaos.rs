//! Chaos mode: the table, uniformly.
//!
//! Every legal row is as likely as every other, with two judgements only: reach for a kind the
//! run is not holding, and let time pass between steps -- aimed, half the time, at a deadline the
//! driver has armed, because boundary races are where a driver breaks and uniform delays never
//! land on one. Shallow everywhere by construction: this mode is the check that every row applies
//! and every declared open is real, not the search. The search is conversation mode.
use std::{collections::BTreeMap, time::Instant};

use fastrand::Rng;
use lk_agents_abi::Ts;
use lk_agents_voice::{
    agent_session::cycle::TurnTimer,
    trace::{CoreStep, Step},
};

use crate::{
    drive::{Case, Outcome, Run, Setup},
    input::{DELAYS_MS, Input, Millis, Sampler},
    record::Recording,
};

/// How often to prefer an input that starts a kind the run is not holding.
///
/// Uniform choice keeps hitting the second half of chains it never started, so a long run comes
/// out wide and shallow. The only judgement the search currently makes.
const REACH_FOR_NEW: u8 = 55;

/// How often an `Advance` is aimed at a deadline the driver has armed, rather than a stock delay.
///
/// The driver's own timers are a map of where its behaviour changes. Landing an action one
/// millisecond before, exactly on, and one after each armed deadline is how a search finds the
/// conflicts that only exist at a boundary, and uniform delays almost never land there.
const AIM_AT_TIMER: u8 = 50;

/// How often a step is followed by time passing, independently of which input was chosen.
///
/// `Advance` is one row in a table of sixty four, so left to the table time passes about twice per
/// run, and otherwise only when an await gives up after its whole budget. Every deadline the driver
/// arms is then crossed by a single thirty-second leap and never approached. Time is a dimension of
/// the space, not one input among many, so it is sampled between steps rather than competing with
/// them.
const TICK: u8 = 50;

/// Time passing between two steps: aimed at an armed deadline when there is one, a stock delay
/// otherwise.
pub fn tick(record: &Recording, rng: &mut Rng) -> Input {
    let mut sampler = Sampler::new(record.live(), rng);
    let ms = aim_at_armed_timer(record, &mut sampler)
        .unwrap_or_else(|| Millis(*sampler.pick(DELAYS_MS)));
    Input::Advance { ms }
}

/// Sample the next input to send, or `None` when the run can send nothing at all.
///
/// [`Input::available`] has already asked every row whether its references resolve, so the legal
/// moves arrive ready-made and no availability rule is written twice.
pub fn sample_next(record: &Recording, rng: &mut Rng) -> Option<Input> {
    let live = record.live();
    let mut sampler = Sampler::new(live, rng);
    let sendable = Input::available(&mut sampler);
    if sendable.is_empty() {
        return None;
    }
    // The ones that start something the run is not holding yet.
    let fresh: Vec<&Input> =
        sendable.iter().filter(|i| i.outcomes().iter().any(|k| live.of(*k).is_empty())).collect();
    let mut chosen = if !fresh.is_empty() && sampler.chance(REACH_FOR_NEW) {
        (*sampler.pick(&fresh)).clone()
    } else {
        sampler.pick(&sendable).clone()
    };
    if let Input::Advance { ms } = &mut chosen
        && sampler.chance(AIM_AT_TIMER)
        && let Some(aimed) = aim_at_armed_timer(record, &mut sampler)
    {
        *ms = aimed;
    }
    Some(chosen)
}

/// A delay that lands just before, on, or just after a deadline the driver still has armed.
fn aim_at_armed_timer(record: &Recording, sampler: &mut Sampler<'_>) -> Option<Millis> {
    let mut armed: BTreeMap<TurnTimer, Ts> = BTreeMap::new();
    for (_, step) in record.journal() {
        match step {
            Step::Core(CoreStep::TimerArmed { timer, at }) => {
                armed.insert(*timer, *at);
            }
            Step::Core(CoreStep::TimerFired(timer, _) | CoreStep::TimerCancelled(timer)) => {
                armed.remove(timer);
            }
            _ => {}
        }
    }
    let now_ms = record.now().at_ms;
    let pending: Vec<u64> = armed
        .values()
        .map(|at| at.as_nanos() / 1_000_000)
        .filter(|deadline| *deadline > now_ms)
        .collect();
    if pending.is_empty() {
        return None;
    }
    let deadline = *sampler.pick(&pending);
    let offset = *sampler.pick(&[-1i64, 0, 1]);
    Some(Millis(((deadline - now_ms) as i64 + offset).max(1) as u64))
}

/// Explore from a seed: sample inputs until the budget runs out, then close the session down.
///
/// The case is built as it is applied, because an input is only legal against what the run holds
/// at that moment, which is unknowable without having run the earlier ones.
pub fn explore(setup: Setup, seed: u64, budget: usize) -> Outcome {
    crate::drive::simulate_with(seed, explore_in(setup, seed, budget))
}

async fn explore_in(setup: Setup, seed: u64, budget: usize) -> Outcome {
    let started = Instant::now();
    let mut rng = Rng::with_seed(seed);
    let mut run = Run::start(&setup).await;
    let mut outcome =
        Outcome { case: Case { setup, inputs: Vec::new(), schedule: seed }, ..Outcome::default() };

    for _ in 0..budget {
        let Some(input) = sample_next(&run.record, &mut rng) else { break };
        // Recorded before it is applied, so a case that ends in a panic still says what caused it.
        outcome.case.inputs.push(input.clone());
        run.apply(&input).await;
        if let Some(found) = crate::drive::guard(&run, &mut outcome, started) {
            return found;
        }
        if rng.u8(0..100) < TICK {
            let advance = tick(&run.record, &mut rng);
            outcome.case.inputs.push(advance.clone());
            run.apply(&advance).await;
            if let Some(found) = crate::drive::guard(&run, &mut outcome, started) {
                return found;
            }
        }
    }
    crate::drive::finish(run, outcome, started).await
}
