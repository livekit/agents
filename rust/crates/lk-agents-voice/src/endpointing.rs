use crate::agent_activity::options::EndpointingMode;
use lk_agents_abi::{ExpFilter, Maybe, Ts};
use std::time::Duration;

/// User speech within this of the agent starting isn't a backchannel: TTS leading silence means
/// they may have heard nothing yet.
const AGENT_SPEECH_LEADING_SILENCE_GRACE: Duration = Duration::from_millis(250);

/// Retro-dates a VAD interruption to just before the agent started; only "strictly before" matters.
const OVERLAP_BACKDATE: Duration = Duration::from_millis(1);

#[derive(Debug, Clone)]
pub enum Endpointing {
    Fixed(Fixed),
    Dynamic(Dynamic),
}

impl Endpointing {
    pub fn new(cfg: &crate::turn::Endpointing) -> Self {
        match cfg.mode {
            EndpointingMode::Dynamic => {
                Endpointing::Dynamic(Dynamic::new(cfg.min_delay, cfg.max_delay, cfg.alpha))
            }
            EndpointingMode::Fixed => Endpointing::Fixed(Fixed::new(cfg.min_delay, cfg.max_delay)),
        }
    }

    /// The learned floor for `Dynamic`, the configured one for `Fixed`.
    pub fn min_delay(&self) -> Duration {
        match self {
            Endpointing::Fixed(f) => f.min_delay,
            Endpointing::Dynamic(d) => d.min_delay(),
        }
    }

    pub fn max_delay(&self) -> Duration {
        match self {
            Endpointing::Fixed(f) => f.max_delay,
            Endpointing::Dynamic(d) => d.base.max_delay,
        }
    }

    pub fn on_start_of_speech(&mut self, at: Ts, overlapping: bool) {
        match self {
            Endpointing::Fixed(f) => f.overlapping = overlapping,
            Endpointing::Dynamic(d) => d.on_start_of_speech(at, overlapping),
        }
    }

    /// `Some(min)` when a `Dynamic` estimator's floor moved.
    pub fn on_end_of_speech(&mut self, at: Ts, should_ignore: bool) -> Option<Duration> {
        match self {
            Endpointing::Fixed(f) => {
                f.overlapping = false;
                None
            }
            Endpointing::Dynamic(d) => d.on_end_of_speech(at, should_ignore),
        }
    }

    pub fn on_start_of_agent_speech(&mut self, at: Ts) {
        if let Endpointing::Dynamic(d) = self {
            d.on_start_of_agent_speech(at)
        }
    }

    pub fn on_end_of_agent_speech(&mut self, at: Ts) {
        if let Endpointing::Dynamic(d) = self {
            d.on_end_of_agent_speech(at)
        }
    }
}

/// `BaseEndpointing`: fixed delays.
#[derive(Debug, Clone)]
#[lk_agents_abi::class(name = "FixedEndpointing", mutable)]
pub struct Fixed {
    min_delay: Duration,
    max_delay: Duration,
    overlapping: bool,
}

impl Fixed {
    fn new(min_delay: Duration, max_delay: Duration) -> Self {
        Fixed { min_delay, max_delay, overlapping: false }
    }
}

/// Seconds on the host's clock (`time.time()`, or whatever its caller stamps with) as a `Ts`.
fn ts_of(secs: f64) -> Ts {
    Ts::from_nanos((secs.max(0.0) * 1e9).round() as u64)
}

fn secs_of(ts: Ts) -> f64 {
    ts.as_nanos() as f64 / 1e9
}

fn delay_of(secs: f64) -> Duration {
    Duration::try_from_secs_f64(secs).unwrap_or(Duration::ZERO)
}

/// `voice.endpointing.BaseEndpointing`'s state, for its host class.
#[lk_agents_abi::host(object(name = "FixedEndpointing", pid = false))]
impl Fixed {
    #[host(new)]
    pub fn for_host(min_delay: f64, max_delay: f64) -> Self {
        Fixed::new(delay_of(min_delay), delay_of(max_delay))
    }

    #[host(getter(name = "min_delay"))]
    pub fn min_delay_secs(&self) -> f64 {
        self.min_delay.as_secs_f64()
    }

    #[host(getter(name = "max_delay"))]
    pub fn max_delay_secs(&self) -> f64 {
        self.max_delay.as_secs_f64()
    }

    #[host(getter)]
    pub fn overlapping(&self) -> bool {
        self.overlapping
    }

    /// `update_options(min_delay=, max_delay=)`: each only when given.
    #[host(method(kw_only))]
    pub fn update_options(&mut self, min_delay: lk_agents_abi::Maybe<f64>, max_delay: lk_agents_abi::Maybe<f64>) {
        if let lk_agents_abi::Maybe::Given(d) = min_delay {
            self.min_delay = delay_of(d);
        }
        if let lk_agents_abi::Maybe::Given(d) = max_delay {
            self.max_delay = delay_of(d);
        }
    }

    #[host(method(name = "on_start_of_speech"))]
    pub fn on_start_of_speech_for_host(&mut self, _started_at: f64, overlapping: bool) {
        self.overlapping = overlapping;
    }

    #[host(method(name = "on_end_of_speech"))]
    pub fn on_end_of_speech_for_host(&mut self, _ended_at: f64, _should_ignore: bool) {
        self.overlapping = false;
    }
}

/// Learns `min_delay` from how the user pauses (`max_delay` only clamps). Learned: an intra-user
/// pause with no agent turn between, and a pause before an immediate interruption.
#[derive(Debug, Clone)]
#[lk_agents_abi::class(name = "DynamicEndpointing", mutable)]
pub struct Dynamic {
    base: Fixed,
    /// Seconds. The filter is clamped to `[min_delay, max_delay]` at construction.
    utterance_pause: ExpFilter,
    utterance_started_at: Option<Ts>,
    utterance_ended_at: Option<Ts>,
    agent_started_at: Option<Ts>,
    agent_ended_at: Option<Ts>,
    /// `_speaking`: between the user's start and end of speech.
    speaking: bool,
}

/// `voice.endpointing.DynamicEndpointing`'s state, for its host class: seconds in, seconds out.
#[lk_agents_abi::host(object(name = "DynamicEndpointing", pid = false))]
impl Dynamic {
    #[host(new(defaults(alpha = 0.9)))]
    pub fn for_host(min_delay: f64, max_delay: f64, alpha: f64) -> Result<Self, lk_agents_abi::FilterError> {
        let (min, max) = (delay_of(min_delay), delay_of(max_delay));
        let f = ExpFilter::new(alpha, Maybe::Given(max_delay), Maybe::Given(min_delay), Maybe::Given(min_delay))?;
        Ok(Dynamic { utterance_pause: f, ..Dynamic::new(min, max, 0.5) })
    }

    /// The learned floor, clamped by the ceiling.
    #[host(getter(name = "min_delay"))]
    pub fn min_delay_secs(&self) -> f64 {
        let learned = self.utterance_pause.value().unwrap_or(self.base.min_delay.as_secs_f64());
        learned.min(self.base.max_delay.as_secs_f64())
    }

    #[host(getter(name = "max_delay"))]
    pub fn max_delay_secs(&self) -> f64 {
        self.base.max_delay.as_secs_f64()
    }

    /// `_min_delay`: the configured floor, not the learned one.
    #[host(getter)]
    pub fn configured_min_delay(&self) -> f64 {
        self.base.min_delay.as_secs_f64()
    }

    #[host(getter)]
    pub fn overlapping(&self) -> bool {
        self.base.overlapping
    }

    #[host(getter)]
    pub fn speaking(&self) -> bool {
        self.speaking
    }

    #[host(getter)]
    pub fn utterance_started_at(&self) -> Option<f64> {
        self.utterance_started_at.map(secs_of)
    }

    #[host(getter)]
    pub fn utterance_ended_at(&self) -> Option<f64> {
        self.utterance_ended_at.map(secs_of)
    }

    #[host(getter)]
    pub fn agent_speech_started_at(&self) -> Option<f64> {
        self.agent_started_at.map(secs_of)
    }

    #[host(getter)]
    pub fn agent_speech_ended_at(&self) -> Option<f64> {
        self.agent_ended_at.map(secs_of)
    }

    #[host(getter)]
    pub fn between_utterance_delay(&self) -> f64 {
        self.between_utterance().as_secs_f64()
    }

    #[host(getter)]
    pub fn between_turn_delay(&self) -> f64 {
        self.between_turn().as_secs_f64()
    }

    #[host(getter)]
    pub fn immediate_interruption_delay(&self) -> (f64, f64) {
        let (turn, interruption) = self.immediate_interruption();
        (turn.as_secs_f64(), interruption.as_secs_f64())
    }

    #[host(method(name = "on_start_of_agent_speech"))]
    pub fn on_start_of_agent_speech_for_host(&mut self, started_at: f64) {
        self.on_start_of_agent_speech(ts_of(started_at));
    }

    #[host(method(name = "on_end_of_agent_speech"))]
    pub fn on_end_of_agent_speech_for_host(&mut self, ended_at: f64) {
        self.on_end_of_agent_speech(ts_of(ended_at));
    }

    #[host(method(name = "on_start_of_speech"))]
    pub fn on_start_of_speech_for_host(&mut self, started_at: f64, overlapping: bool) {
        self.on_start_of_speech(ts_of(started_at), overlapping);
    }

    /// The learned floor when it moved, for the host's log line.
    #[host(method(name = "on_end_of_speech"))]
    pub fn on_end_of_speech_for_host(&mut self, ended_at: f64, should_ignore: bool) -> Option<f64> {
        self.on_end_of_speech(ts_of(ended_at), should_ignore).map(|_| self.min_delay_secs())
    }

    /// `update_options`: a new floor resets the filter to it (and clamps from below), a new
    /// ceiling clamps from above, a new `alpha` keeps the learned value.
    #[host(method(kw_only))]
    pub fn update_options(
        &mut self,
        min_delay: Maybe<f64>,
        max_delay: Maybe<f64>,
        alpha: Maybe<f64>,
    ) -> Result<(), lk_agents_abi::FilterError> {
        if let Maybe::Given(d) = min_delay {
            self.base.min_delay = delay_of(d);
            self.utterance_pause.reset(Maybe::NotGiven, Maybe::Given(d), Maybe::Given(d), Maybe::NotGiven)?;
        }
        if let Maybe::Given(d) = max_delay {
            self.base.max_delay = delay_of(d);
            self.utterance_pause.reset(Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven, Maybe::Given(d))?;
        }
        if let Maybe::Given(a) = alpha {
            self.utterance_pause.reset(Maybe::Given(a), Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven)?;
        }
        Ok(())
    }
}

impl Dynamic {
    fn new(min_delay: Duration, max_delay: Duration, alpha: f64) -> Self {
        let f = ExpFilter::new(
            alpha,
            Maybe::Given(max_delay.as_secs_f64()),
            Maybe::Given(min_delay.as_secs_f64()),
            Maybe::Given(min_delay.as_secs_f64()),
        )
        .expect("alpha validated by the option resolver");
        Dynamic {
            base: Fixed::new(min_delay, max_delay),
            utterance_pause: f,
            utterance_started_at: None,
            utterance_ended_at: None,
            agent_started_at: None,
            agent_ended_at: None,
            speaking: false,
        }
    }

    /// Clamped on read: `update_options` can lower the ceiling below a learned floor.
    pub fn min_delay(&self) -> Duration {
        let learned = self
            .utterance_pause
            .value()
            .map(Duration::from_secs_f64)
            .unwrap_or(self.base.min_delay);
        learned.min(self.base.max_delay)
    }

    /// Gap between the previous utterance ending and this one starting, saturating.
    fn between_utterance(&self) -> Duration {
        match (self.utterance_started_at, self.utterance_ended_at) {
            (Some(start), Some(end)) => start.saturating_duration_since(end),
            (Some(_), None) | (None, Some(_)) | (None, None) => Duration::ZERO,
        }
    }

    /// Gap between the previous utterance ending and the agent starting to speak.
    fn between_turn(&self) -> Duration {
        match (self.agent_started_at, self.utterance_ended_at) {
            (Some(agent), Some(end)) => agent.saturating_duration_since(end),
            (Some(_), None) | (None, Some(_)) | (None, None) => Duration::ZERO,
        }
    }

    /// `[utterance] [a][b] [immediate interruption]`: `a` up to the agent starting, `b` to the cut.
    /// The early return's `||` -> `&&` is an equivalent mutant (caught downstream).
    fn immediate_interruption(&self) -> (Duration, Duration) {
        if self.utterance_started_at.is_none() || self.agent_started_at.is_none() {
            return (Duration::ZERO, Duration::ZERO);
        }
        let turn = self.between_turn();
        (turn, self.between_utterance().abs_diff(turn))
    }

    fn learn(&mut self, pause: Duration) -> Option<Duration> {
        let before = self.min_delay();
        let _ = self.utterance_pause.apply(1.0, Maybe::Given(pause.as_secs_f64()));
        let after = self.min_delay();
        (after != before).then_some(after)
    }

    fn on_start_of_agent_speech(&mut self, at: Ts) {
        self.agent_started_at = Some(at);
        self.agent_ended_at = None;
        self.base.overlapping = false;
    }

    fn on_end_of_agent_speech(&mut self, at: Ts) {
        // Agent timestamps survive until the next user utterance ends; the ordering check absorbs
        // the duplicate call.
        let stale = self.agent_ended_at.is_none_or(|e| Some(e) < self.agent_started_at);
        if self.agent_started_at.is_some() && stale {
            self.agent_ended_at = Some(at);
        }
        self.base.overlapping = false;
    }

    fn on_start_of_speech(&mut self, at: Ts, overlapping: bool) {
        if self.base.overlapping {
            // Duplicate: the interruption path and the VAD path both call this.
            return;
        }

        // A VAD interruption fires before end-of-speech; retro-date the open utterance.
        if overlapping
            && let (Some(start), Some(end), Some(agent)) =
                (self.utterance_started_at, self.utterance_ended_at, self.agent_started_at)
            && end < start
        {
            self.utterance_ended_at = Some(agent.saturating_sub(OVERLAP_BACKDATE));
        }

        self.utterance_started_at = Some(at);
        self.base.overlapping = overlapping;
        self.speaking = true;
    }

    fn on_end_of_speech(&mut self, at: Ts, should_ignore: bool) -> Option<Duration> {
        if should_ignore && self.base.overlapping && !self.within_grace_of_agent_speech() {
            // A genuine backchannel: learn nothing and forget the utterance entirely.
            self.base.overlapping = false;
            self.speaking = false;
            self.utterance_started_at = None;
            self.utterance_ended_at = None;
            return None;
        }

        let agent_still_speaking = self.agent_started_at.is_some() && self.agent_ended_at.is_none();
        let mut learned = None;

        if self.base.overlapping || agent_still_speaking {
            let (turn, interruption) = self.immediate_interruption();
            let pause = self.between_utterance();
            let immediate = interruption > Duration::ZERO
                && interruption <= self.min_delay()
                && turn > Duration::ZERO
                && turn <= self.base.max_delay;
            if immediate && pause > Duration::ZERO {
                learned = self.learn(pause); // case 2
            }
        } else {
            let pause = self.between_utterance();
            // `&&` -> `||` is an equivalent mutant: `(None, Some)` is unreachable.
            let no_agent_turn_between =
                self.agent_started_at.is_none() && self.agent_ended_at.is_none();
            if pause > Duration::ZERO && no_agent_turn_between {
                learned = self.learn(pause); // case 1
            }
        }

        self.utterance_ended_at = Some(at);
        self.agent_started_at = None;
        self.agent_ended_at = None;
        self.speaking = false;
        self.base.overlapping = false;
        learned
    }

    fn within_grace_of_agent_speech(&self) -> bool {
        match (self.utterance_started_at, self.agent_started_at) {
            (Some(u), Some(a)) => {
                u.as_nanos().abs_diff(a.as_nanos())
                    < AGENT_SPEECH_LEADING_SILENCE_GRACE.as_nanos() as u64
            }
            (Some(_), None) | (None, Some(_)) | (None, None) => false,
        }
    }
}
