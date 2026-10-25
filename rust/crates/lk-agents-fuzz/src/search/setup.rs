//! The setup: what a session is built with, and the arms that learn which switches pay.
use std::collections::BTreeMap;

use fastrand::Rng;

use crate::{
    drive::{RealtimeSetup, RoomSetup, Setup},
    input::{AGENT_TURNS, DELAYS_MS, NAMES, TurnMode},
};

/// Sample a session to run the case against, with nothing learned: every switch a coin, every
/// three-way switch a die. What `SetupArms` starts from, and what a single run draws from.
pub fn sample_setup(rng: &mut Rng) -> Setup {
    SetupArms::new().sample(rng).0
}

/// Thompson sampling over the setup's switches: the seed-scheduling scheme of the paper, one
/// level up, where the odds of reaching anything actually live.
///
/// Each value a switch can take is an arm with a `Beta(1, 1)` prior. A seed that reaches a
/// journal variant the campaign had not is a success for every arm it drew, any other seed a
/// failure, and the next seed's switches are set by drawing from each arm's posterior and taking
/// the value that drew highest. The arms lean toward the values that have kept finding new ground
/// and never stop exploring, because a Beta draw is never zero. Nothing here names a variant or a
/// switch: the same rule is applied to every switch, and a campaign that has found everything it
/// can drifts back toward uniform as every arm's successes stop.
#[derive(Clone, Debug, Default)]
pub struct SetupArms {
    /// `(switch, value) -> (successes + 1, failures + 1)`: the Beta posterior's two parameters.
    arms: BTreeMap<(&'static str, &'static str), (f64, f64)>,
}

/// The arms one setup was drawn from, for the campaign to reward once the seed has run.
#[derive(Clone, Debug, Default)]
pub struct Drawn(Vec<(&'static str, &'static str)>);

impl SetupArms {
    pub fn new() -> Self {
        Self::default()
    }

    /// One value of a switch: the arm whose posterior draw is highest.
    fn pick(
        &self,
        rng: &mut Rng,
        switch: &'static str,
        values: &[&'static str],
        drawn: &mut Drawn,
    ) -> usize {
        let mut best = (0, f64::MIN);
        for (i, value) in values.iter().enumerate() {
            let (a, b) = self.arms.get(&(switch, *value)).copied().unwrap_or((1.0, 1.0));
            let draw = beta(rng, a, b);
            if draw > best.1 {
                best = (i, draw);
            }
        }
        drawn.0.push((switch, values[best.0]));
        best.0
    }

    fn on(&self, rng: &mut Rng, switch: &'static str, drawn: &mut Drawn) -> bool {
        self.pick(rng, switch, &["off", "on"], drawn) == 1
    }

    /// A setup, and the arms it came from.
    pub fn sample(&self, rng: &mut Rng) -> (Setup, Drawn) {
        let mut d = Drawn::default();
        let names = |rng: &mut Rng| -> Vec<String> {
            (0..rng.usize(0..4)).map(|_| NAMES[rng.usize(0..NAMES.len())].to_owned()).collect()
        };
        let ms = |rng: &mut Rng| DELAYS_MS[rng.usize(0..DELAYS_MS.len())];
        let turn = match self.pick(rng, "turn", &["default", "stt", "manual"], &mut d) {
            0 => TurnMode::Default,
            1 => TurnMode::Stt,
            _ => TurnMode::Manual,
        };
        let setup = Setup {
            stt: self.on(rng, "stt", &mut d),
            vad: self.on(rng, "vad", &mut d),
            eot: self.on(rng, "eot", &mut d),
            interruption: self.on(rng, "interruption", &mut d),
            dynamic_endpointing: self.on(rng, "dynamic_endpointing", &mut d),
            turn,
            preemptive: self.on(rng, "preemptive", &mut d),
            tools: if self.on(rng, "tools", &mut d) { names(rng) } else { vec![] },
            max_tool_steps: self.on(rng, "max_tool_steps", &mut d).then(|| rng.u32(0..4)),
            instructions: AGENT_TURNS[rng.usize(0..AGENT_TURNS.len())].to_owned(),
            max_unrecoverable_errors: rng.u32(0..4),
            hook_bound_ms: self.on(rng, "hooks", &mut d).then(|| ms(rng)),
            realtime: self
                .on(rng, "realtime", &mut d)
                .then(|| RealtimeSetup { turn_detection: self.on(rng, "server_turns", &mut d) }),
            // Three ways: no navigator, one the host started, one navigating from the start.
            ivr: match self.pick(rng, "ivr", &["none", "started", "detecting"], &mut d) {
                0 => None,
                1 => Some(false),
                _ => Some(true),
            },
            amd: self.on(rng, "amd", &mut d),
            keyterms: if self.on(rng, "keyterms", &mut d) { names(rng) } else { vec![] },
            room: self.on(rng, "room", &mut d).then(|| RoomSetup {
                close_on_disconnect: rng.bool(),
                delete_room_on_close: rng.bool(),
            }),
            tts: self.on(rng, "tts", &mut d),
            connect_ms: self.on(rng, "connect", &mut d).then(|| ms(rng)),
            video: self.on(rng, "video", &mut d),
            sync_transcription: self.on(rng, "sync_transcription", &mut d),
            pausable_output: self.on(rng, "pausable_output", &mut d),
            speech_gap_ms: self.on(rng, "speech_gap", &mut d).then(|| ms(rng)),
            user_turn_max_words: self.on(rng, "turn_words", &mut d).then(|| rng.u32(1..12)),
            user_turn_max_ms: self.on(rng, "turn_duration", &mut d).then(|| ms(rng)),
            transcription_timeout_ms: self
                .on(rng, "transcription_timeout", &mut d)
                .then(|| ms(rng)),
            backchannel_threshold: self.on(rng, "backchannel", &mut d).then(|| rng.u8(0..=100)),
        };
        (setup, d)
    }

    /// The seed ran: what it drew found new ground, or did not.
    pub fn learn(&mut self, drawn: &Drawn, found_new: bool) {
        for arm in &drawn.0 {
            let (a, b) = self.arms.entry(*arm).or_insert((1.0, 1.0));
            if found_new {
                *a += 1.0;
            } else {
                *b += 1.0;
            }
        }
    }

    /// Where the arms lean, for reading a campaign: `(switch, value, posterior mean)`, the
    /// switches whose values it has come to prefer first.
    pub fn leanings(&self) -> Vec<(&'static str, &'static str, f64)> {
        let mut out: Vec<_> =
            self.arms.iter().map(|((s, v), (a, b))| (*s, *v, a / (a + b))).collect();
        out.sort_by(|x, y| y.2.total_cmp(&x.2));
        out
    }
}

/// A draw from `Beta(a, b)`, as the ratio of two Gamma draws.
fn beta(rng: &mut Rng, a: f64, b: f64) -> f64 {
    let x = gamma(rng, a);
    let y = gamma(rng, b);
    if x + y == 0.0 { 0.5 } else { x / (x + y) }
}

/// A draw from `Gamma(shape, 1)`: Marsaglia and Tsang for a shape of one or more, boosted for less.
fn gamma(rng: &mut Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        let u = rng.f64().max(f64::MIN_POSITIVE);
        return gamma(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let n = normal(rng);
        let v = (1.0 + c * n).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.f64().max(f64::MIN_POSITIVE);
        if u.ln() < 0.5 * n * n + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// A standard normal draw, by Box and Muller.
fn normal(rng: &mut Rng) -> f64 {
    let u1 = rng.f64().max(f64::MIN_POSITIVE);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
