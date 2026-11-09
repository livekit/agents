use std::collections::BTreeMap;

use fastrand::Rng;
use lk_agents_abi::Ts;
use lk_agents_voice::trace::{CoreStep, Step, TimerName};

use crate::{
    input::{DELAYS_MS, Input, Millis, RoomIn, Sampler},
    record::Recording,
};

// A uniform move: any legal row of the table, no actor behind it.

/// How often to prefer an input that starts a kind the run is not holding, since uniform choice
/// keeps hitting the second half of chains it never started.
const REACH_FOR_NEW: u8 = 55;

/// How often an `Advance` is aimed just before, on or after an armed deadline: boundary races
/// are where a driver breaks, and uniform delays almost never land there.
const AIM_AT_TIMER: u8 = 50;

/// How often a step is followed by time passing. Time is sampled between steps rather than as
/// one table row, or deadlines are only ever crossed by one long leap.
pub const TICK: u8 = 50;

/// Time passing between two steps: aimed at an armed deadline when there is one, a stock delay
/// otherwise.
pub fn tick(record: &Recording, rng: &mut Rng) -> Input {
    let mut sampler = Sampler::new(record.live(), rng);
    let ms = aim_at_armed_timer(record, &mut sampler)
        .unwrap_or_else(|| Millis(*sampler.pick(DELAYS_MS)));
    Input::Advance { ms }
}

/// Sample the next input to send, or `None` when the run can send nothing at all.
pub fn sample_next(record: &Recording, rng: &mut Rng) -> Option<Input> {
    let live = record.live();
    let mut sampler = Sampler::new(live, rng);
    // How the call ends is its own dial; a uniform move never ends it.
    let sendable: Vec<Input> = Input::available(&mut sampler)
        .into_iter()
        .filter(|i| {
            !matches!(
                i,
                Input::Close { .. } | Input::Room { event: RoomIn::Closed | RoomIn::Unlinked }
            )
        })
        .collect();
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
pub fn aim_at_armed_timer(record: &Recording, sampler: &mut Sampler<'_>) -> Option<Millis> {
    let mut armed: BTreeMap<TimerName, Ts> = BTreeMap::new();
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
