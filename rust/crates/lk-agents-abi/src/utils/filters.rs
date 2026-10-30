//! Export contracts for numeric filters; implementations live in the media core.

use crate::Maybe;

/// The reference raises `ValueError` in `__init__` and asserts in `reset`; the binding picks.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FilterError {
    #[error("alpha must be in (0, 1].")]
    Alpha,
    #[error("sample or initial value must be given.")]
    NoSample,
}

/// A reset retains the reference's assertion failure for an invalid alpha.
#[crate::host(error(kind = "AssertionError"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct FilterResetError(pub FilterError);

#[crate::host(class(
    name = "ExpFilter",
    module = "livekit.agents.utils.exp_filter",
    pid = false,
    subclass
))]
pub trait ExpFilter: Send + Sized + 'static {
    #[host(new)]
    fn new(
        alpha: f64,
        max_val: Maybe<f64>,
        min_val: Maybe<f64>,
        initial: Maybe<f64>,
    ) -> Result<Self, FilterError>;
    #[host(call)]
    fn reset(
        &mut self,
        alpha: Maybe<f64>,
        initial: Maybe<f64>,
        min_val: Maybe<f64>,
        max_val: Maybe<f64>,
    ) -> Result<(), FilterResetError>;
    #[host(call)]
    fn apply(&mut self, exp: f64, sample: Maybe<f64>) -> Result<f64, FilterError>;
    // Read per audio frame; a field of a Python-owned value, under no lock.
    #[host(getter, attached)]
    fn value(&self) -> Option<f64>;
    #[host(call)]
    fn update_base(&mut self, alpha: f64);
}

#[crate::host(class(
    name = "MovingAverage",
    module = "livekit.agents.utils.moving_average",
    pid = false,
    subclass
))]
pub trait MovingAverage: Send + Sized + 'static {
    #[host(new)]
    fn new(window_size: usize) -> Self;
    #[host(call)]
    fn add_sample(&mut self, sample: f64);
    #[host(call)]
    fn get_avg(&self) -> f64;
    #[host(call)]
    fn reset(&mut self);
    #[host(call)]
    fn size(&self) -> usize;
}
