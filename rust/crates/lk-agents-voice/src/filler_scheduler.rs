use lk_agents_abi::Ts;
use std::time::Duration;

/// One `with_filler` scope, as the host numbers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FillerId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillerAction {
    /// `await session.wait_for_idle()`, then [`on_idle`](FillerScheduler::on_idle).
    WaitForIdle,
    /// Race [`on_busy`](FillerScheduler::on_busy) against this deadline ([`tick`](FillerScheduler::tick)).
    Dwell(Ts),
    /// Sleep until this instant, uninterruptibly, then [`tick`](FillerScheduler::tick).
    CoolDown(Ts),
    /// Produce the source for `step`, then report via [`fired`](FillerScheduler::fired).
    Fire { step: u32 },
    /// Nothing further. The context manager's `aclose` is the only thing left.
    Done,
}

/// What `_FillerScheduler.__init__` raises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[lk_agents_abi::host(error(kind = "ValueError"))]
pub enum FillerError {
    #[error("delay must be non-negative")]
    NegativeDelay,
    #[error("interval must be non-negative when set")]
    NegativeInterval,
}

impl FillerError {
    /// The reference's messages, verbatim -- they are what a user sees from `with_filler`.
    pub fn message(self) -> &'static str {
        match self {
            FillerError::NegativeDelay => "delay must be non-negative",
            FillerError::NegativeInterval => "interval must be non-negative when set",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    WaitingForIdle,
    Dwelling { deadline: Ts },
    CoolingDown { deadline: Ts },
    Done,
}

/// `_FillerScheduler`'s decisions; its host class keeps the waits (`wait_for_idle`, the dwell
/// race, the cooldown sleep) and the `say`.
#[derive(Debug)]
#[lk_agents_abi::class(name = "FillerScheduler", mutable)]
pub struct FillerScheduler {
    delay: Duration,
    /// `None` means "fire at most once", not "no cooldown".
    interval: Option<Duration>,
    max_steps: Option<u32>,
    /// `len(self._created_speeches)`.
    steps: u32,
    state: State,
}

impl FillerScheduler {
    pub fn new(
        delay: f64,
        interval: Option<f64>,
        max_steps: Option<u32>,
    ) -> Result<Self, FillerError> {
        if delay < 0.0 || delay.is_nan() {
            return Err(FillerError::NegativeDelay);
        }
        if let Some(i) = interval {
            if i < 0.0 || i.is_nan() {
                return Err(FillerError::NegativeInterval);
            }
        }
        Ok(FillerScheduler {
            delay: Duration::from_secs_f64(delay),
            interval: interval.map(Duration::from_secs_f64),
            max_steps,
            steps: 0,
            state: State::WaitingForIdle,
        })
    }

    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// What the scheduler waits on now, without moving it.
    pub fn waiting(&self) -> FillerAction {
        match self.state {
            State::WaitingForIdle => FillerAction::WaitForIdle,
            State::Dwelling { deadline } => FillerAction::Dwell(deadline),
            State::CoolingDown { deadline } => FillerAction::CoolDown(deadline),
            State::Done => FillerAction::Done,
        }
    }

    /// The session reported idle: the dwell begins now.
    pub fn on_idle(&mut self, now: Ts) -> FillerAction {
        if self.state == State::Done {
            return FillerAction::Done;
        }
        let deadline = now.saturating_add(self.delay);
        self.state = State::Dwelling { deadline };
        FillerAction::Dwell(deadline)
    }

    /// A busy signal (speech, thinking, or `reset_dwell`); only one inside the dwell restarts it.
    pub fn on_busy(&mut self, _now: Ts) -> FillerAction {
        match self.state {
            State::Dwelling { .. } => {
                self.state = State::WaitingForIdle;
                FillerAction::WaitForIdle
            }
            State::CoolingDown { deadline } => FillerAction::CoolDown(deadline),
            State::WaitingForIdle => FillerAction::WaitForIdle,
            State::Done => FillerAction::Done,
        }
    }

    /// A deadline arrived; not yet due answers the same wait.
    pub fn tick(&mut self, now: Ts) -> FillerAction {
        match self.state {
            State::Dwelling { deadline } if now >= deadline => {
                FillerAction::Fire { step: self.steps }
            }
            State::Dwelling { deadline } => FillerAction::Dwell(deadline),
            State::CoolingDown { deadline } if now >= deadline => {
                self.state = State::WaitingForIdle;
                FillerAction::WaitForIdle
            }
            State::CoolingDown { deadline } => FillerAction::CoolDown(deadline),
            State::WaitingForIdle => FillerAction::WaitForIdle,
            State::Done => FillerAction::Done,
        }
    }

    /// The host produced the source, or declined (`None`), which doesn't advance the step.
    pub fn fired(&mut self, now: Ts, produced: bool) -> FillerAction {
        if self.state == State::Done {
            return FillerAction::Done;
        }
        if produced {
            self.steps += 1;
        }
        // `interval is None` short-circuits before `max_steps`: one fire.
        let Some(interval) = self.interval else {
            self.state = State::Done;
            return FillerAction::Done;
        };
        if self.max_steps.is_some_and(|m| self.steps >= m) {
            self.state = State::Done;
            return FillerAction::Done;
        }
        let deadline = now.saturating_add(interval);
        self.state = State::CoolingDown { deadline };
        FillerAction::CoolDown(deadline)
    }
}

/// The next wait or step, for the host: `("wait_for_idle", None)`, `("dwell", deadline)`,
/// `("cool_down", deadline)`, `("fire", step)` or `("done", None)`; instants in the host's seconds.
fn host_action(action: FillerAction) -> (&'static str, Option<f64>) {
    let secs = |ts: Ts| ts.as_nanos() as f64 / 1e9;
    match action {
        FillerAction::WaitForIdle => ("wait_for_idle", None),
        FillerAction::Dwell(at) => ("dwell", Some(secs(at))),
        FillerAction::CoolDown(at) => ("cool_down", Some(secs(at))),
        FillerAction::Fire { step } => ("fire", Some(f64::from(step))),
        FillerAction::Done => ("done", None),
    }
}

fn host_ts(secs: f64) -> Ts {
    Ts::from_nanos((secs.max(0.0) * 1e9).round() as u64)
}

/// `voice.filler_scheduler._FillerScheduler`'s machine, stamped with the host loop's seconds.
#[lk_agents_abi::host(object(name = "FillerScheduler", pid = false))]
impl FillerScheduler {
    #[host(new(kw_only = "delay"))]
    pub fn for_host(delay: f64, interval: Option<f64>, max_steps: Option<u32>) -> Result<Self, FillerError> {
        FillerScheduler::new(delay, interval, max_steps)
    }

    #[host(getter(name = "steps"))]
    pub fn steps_for_host(&self) -> u32 {
        self.steps
    }

    #[host(method(name = "on_idle"))]
    pub fn on_idle_for_host(&mut self, now: f64) -> (&'static str, Option<f64>) {
        host_action(self.on_idle(host_ts(now)))
    }

    #[host(method(name = "on_busy"))]
    pub fn on_busy_for_host(&mut self, now: f64) -> (&'static str, Option<f64>) {
        host_action(self.on_busy(host_ts(now)))
    }

    #[host(method(name = "tick"))]
    pub fn tick_for_host(&mut self, now: f64) -> (&'static str, Option<f64>) {
        host_action(self.tick(host_ts(now)))
    }

    #[host(method(name = "fired"))]
    pub fn fired_for_host(&mut self, now: f64, produced: bool) -> (&'static str, Option<f64>) {
        host_action(self.fired(host_ts(now), produced))
    }
}
