//! Numeric filters: public API (`livekit.agents.utils.{ExpFilter, MovingAverage}`).

/// The reference raises `ValueError` in `__init__` and asserts in `reset`; the binding picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FilterError {
    #[error("alpha must be in (0, 1].")]
    Alpha,
    #[error("sample or initial value must be given.")]
    NoSample,
}

use crate::Maybe;

/// Exponential moving average with optional clamping.
#[derive(Debug, Clone)]
#[crate::class(name = "ExpFilter", module = "livekit.agents.utils.exp_filter", mutable, subclass)]
pub struct ExpFilter {
    alpha: f64,
    filtered: Option<f64>,
    max_val: Option<f64>,
    min_val: Option<f64>,
}

#[crate::host(object(
    pid = false,
    module = "livekit.agents.utils.exp_filter"
))]
impl ExpFilter {
    /// Positional argument order is part of the contract (`ExpFilter(0.5, 2.0)`).
    #[host(new)]
    pub fn new(
        alpha: f64,
        max_val: Maybe<f64>,
        min_val: Maybe<f64>,
        initial: Maybe<f64>,
    ) -> Result<Self, FilterError> {
        if !(alpha > 0.0 && alpha <= 1.0) {
            return Err(FilterError::Alpha);
        }
        Ok(ExpFilter { alpha, filtered: initial.given(), max_val: max_val.given(), min_val: min_val.given() })
    }

    /// Each argument is optional: absent means "leave as is" (`NotGivenOr`).
    #[host(method(err = "AssertionError"))]
    pub fn reset(
        &mut self,
        alpha: Maybe<f64>,
        initial: Maybe<f64>,
        min_val: Maybe<f64>,
        max_val: Maybe<f64>,
    ) -> Result<(), FilterError> {
        if let Maybe::Given(a) = alpha {
            if !(a > 0.0 && a <= 1.0) {
                return Err(FilterError::Alpha);
            }
            self.alpha = a;
        }
        if let Maybe::Given(v) = initial {
            self.filtered = Some(v);
        }
        if let Maybe::Given(v) = min_val {
            self.min_val = Some(v);
        }
        if let Maybe::Given(v) = max_val {
            self.max_val = Some(v);
        }
        Ok(())
    }

    /// `sample = None` re-applies to the current value, which still re-clamps.
    #[host(method)]
    pub fn apply(&mut self, exp: f64, sample: Maybe<f64>) -> Result<f64, FilterError> {
        let sample = sample.given().or(self.filtered);

        match (sample, self.filtered) {
            (Some(s), None) => self.filtered = Some(s),
            (Some(s), Some(f)) => {
                let a = self.alpha.powf(exp);
                self.filtered = Some(a * f + (1.0 - a) * s);
            }
            (None, _) => {}
        }

        let mut v = self.filtered.ok_or(FilterError::NoSample)?;
        if let Some(max) = self.max_val {
            v = v.min(max);
        }
        if let Some(min) = self.min_val {
            v = v.max(min);
        }
        self.filtered = Some(v);
        Ok(v)
    }

    /// `None` until a sample or an initial value has been seen.
    #[host(getter)]
    pub fn value(&self) -> Option<f64> {
        self.filtered
    }

    #[host(method)]
    pub fn update_base(&mut self, alpha: f64) {
        self.alpha = alpha;
    }
}

/// Mean of the last `window` samples, as a plain ring buffer.
#[derive(Debug, Clone)]
#[crate::class(name = "MovingAverage", module = "livekit.agents.utils.moving_average", mutable, subclass)]
pub struct MovingAverage {
    hist: Vec<f64>,
    sum: f64,
    count: usize,
}

#[crate::host(object(
    pid = false,
    module = "livekit.agents.utils.moving_average"
))]
impl MovingAverage {
    #[host(new)]
    pub fn new(window_size: usize) -> Self {
        MovingAverage { hist: vec![0.0; window_size], sum: 0.0, count: 0 }
    }

    #[host(method)]
    pub fn add_sample(&mut self, sample: f64) {
        let n = self.hist.len();
        if n == 0 {
            return;
        }
        let idx = self.count % n;
        if self.count >= n {
            self.sum -= self.hist[idx];
        }
        self.sum += sample;
        self.hist[idx] = sample;
        self.count += 1;
    }

    /// Zero on an empty window, matching the reference -- callers divide by it without guarding.
    #[host(method)]
    pub fn get_avg(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum / self.size() as f64 }
    }

    #[host(method)]
    pub fn reset(&mut self) {
        self.hist.iter_mut().for_each(|v| *v = 0.0);
        self.sum = 0.0;
        self.count = 0;
    }

    #[host(method)]
    pub fn size(&self) -> usize {
        self.count.min(self.hist.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: clamping only on write, so an out-of-range initial reads back unclamped.
    #[test]
    fn apply_clamps_to_both_bounds() {
        let mut f = ExpFilter::new(0.5, Maybe::Given(2.0), Maybe::Given(0.5), Maybe::Given(10.0)).unwrap();
        assert_eq!(f.apply(1.0, Maybe::Given(10.0)).unwrap(), 2.0, "clamped down to max");
        // 0.5*2.0 + 0.5*(-5.0) = -1.5, clamped up to min.
        assert_eq!(f.apply(1.0, Maybe::Given(-5.0)).unwrap(), 0.5, "clamped up to min");
    }

    // Pins a surprise: `alpha` weights history, so `alpha == 1.0` freezes the filter. Don't
    // "fix" it into 1 - alpha.
    #[test]
    fn alpha_of_one_freezes_the_filter() {
        let mut f = ExpFilter::new(1.0, Maybe::NotGiven, Maybe::NotGiven, Maybe::Given(3.0)).unwrap();
        assert_eq!(f.apply(1.0, Maybe::Given(100.0)).unwrap(), 3.0);
        assert_eq!(f.apply(1.0, Maybe::Given(-100.0)).unwrap(), 3.0);
    }

    // Catches: the first sample averaged against zero instead of seeding the filter.
    #[test]
    fn first_sample_seeds_rather_than_averages() {
        let mut f = ExpFilter::new(0.5, Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven).unwrap();
        assert_eq!(f.value(), None);
        assert_eq!(f.apply(1.0, Maybe::Given(4.0)).unwrap(), 4.0, "seeded, not 0.5*0 + 0.5*4");
        assert_eq!(f.value(), Some(4.0));
        assert_eq!(f.apply(1.0, Maybe::Given(0.0)).unwrap(), 2.0, "now it averages");
        assert_eq!(f.value(), Some(2.0), "it tracks, rather than latching the first sample");
    }

    // Catches: `alpha^exp` being ignored.
    #[test]
    fn exp_weights_the_alpha() {
        let mut a = ExpFilter::new(0.5, Maybe::NotGiven, Maybe::NotGiven, Maybe::Given(0.0)).unwrap();
        let mut b = ExpFilter::new(0.5, Maybe::NotGiven, Maybe::NotGiven, Maybe::Given(0.0)).unwrap();
        let one = a.apply(1.0, Maybe::Given(1.0)).unwrap(); // a = 0.5   -> 0.5
        let two = b.apply(2.0, Maybe::Given(1.0)).unwrap(); // a = 0.25  -> 0.75
        assert_eq!((one, two), (0.5, 0.75));
    }

    // Catches: `apply` with no sample skipping the re-clamp.
    #[test]
    fn apply_without_a_sample_still_reclamps() {
        let mut f = ExpFilter::new(0.9, Maybe::NotGiven, Maybe::NotGiven, Maybe::Given(1.0)).unwrap();
        f.reset(Maybe::NotGiven, Maybe::NotGiven, Maybe::Given(5.0), Maybe::NotGiven).unwrap();
        assert_eq!(f.apply(1.0, Maybe::NotGiven).unwrap(), 5.0);
    }


    // Catches: applying with neither sample nor initial returning 0.0 instead of erroring (silero).
    #[test]
    fn apply_with_nothing_to_average_is_an_error() {
        let mut f = ExpFilter::new(0.35, Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven).unwrap();
        assert_eq!(f.apply(1.0, Maybe::NotGiven).unwrap_err(), FilterError::NoSample);
    }

    // Catches: an out-of-range alpha accepted by either entry point, or `reset`'s bound loosened
    // or inverted (a valid alpha must take effect, one above 1 must be refused).
    #[test]
    fn alpha_is_validated_on_both_paths() {
        for bad in [0.0, 1.5] {
            assert_eq!(ExpFilter::new(bad, Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven).unwrap_err(), FilterError::Alpha);
        }
        let mut f = ExpFilter::new(1.0, Maybe::NotGiven, Maybe::NotGiven, Maybe::Given(0.0)).unwrap();

        f.reset(Maybe::Given(0.5), Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven).unwrap();
        assert_eq!(f.apply(1.0, Maybe::Given(1.0)).unwrap(), 0.5, "the new alpha is the one in force");

        assert_eq!(f.reset(Maybe::Given(1.5), Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven).unwrap_err(), FilterError::Alpha);
        assert_eq!(f.reset(Maybe::Given(-0.1), Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven).unwrap_err(), FilterError::Alpha);
        assert!(f.reset(Maybe::Given(1.0), Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven).is_ok(), "1.0 is the inclusive upper bound");
        assert_eq!(
            f.apply(1.0, Maybe::Given(9.0)).unwrap(),
            0.5,
            "a rejected reset must leave the filter alone, and 1.0 freezes it"
        );
    }


    // Catches: `update_base` doing nothing (`utils/exp_filter.py:63`). Unvalidated, as in the
    // reference.
    #[test]
    fn update_base_replaces_the_alpha_and_skips_the_validation_reset_applies() {
        let mut f = ExpFilter::new(1.0, Maybe::NotGiven, Maybe::NotGiven, Maybe::Given(0.0)).unwrap();
        f.update_base(0.25);
        // 0.25*0 + 0.75*4 = 3.0; with the original alpha of 1.0 this would still be 0.0.
        assert_eq!(f.apply(1.0, Maybe::Given(4.0)).unwrap(), 3.0);

        f.update_base(2.0);
        assert_eq!(
            f.reset(Maybe::Given(2.0), Maybe::NotGiven, Maybe::NotGiven, Maybe::NotGiven).unwrap_err(),
            FilterError::Alpha,
            "`reset` rejects what `update_base` just accepted -- the reference's asymmetry"
        );
    }

    // Catches: the window evicting the wrong slot, so the average drifts once it wraps.
    #[test]
    fn moving_average_evicts_exactly_the_oldest() {
        let mut m = MovingAverage::new(3);
        for s in [1.0, 2.0, 3.0] {
            m.add_sample(s);
        }
        assert_eq!((m.size(), m.get_avg()), (3, 2.0));
        m.add_sample(4.0); // evicts 1.0 -> mean of 2,3,4
        assert_eq!((m.size(), m.get_avg()), (3, 3.0));
        m.add_sample(5.0); // evicts 2.0 -> mean of 3,4,5
        assert_eq!(m.get_avg(), 4.0);
    }

    // Catches: dividing by the window instead of the sample count before the window fills.
    #[test]
    fn moving_average_divides_by_count_until_full() {
        let mut m = MovingAverage::new(5);
        assert_eq!(m.get_avg(), 0.0, "empty must not divide by zero");
        m.add_sample(10.0);
        assert_eq!((m.size(), m.get_avg()), (1, 10.0), "not 10/5");
        m.reset();
        assert_eq!((m.size(), m.get_avg()), (0, 0.0));
    }
}
