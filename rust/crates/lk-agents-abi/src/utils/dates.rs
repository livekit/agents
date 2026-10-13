//! `datetime.date` and `datetime.time`, valid by construction, and the injectable `date.today()`.

use crate::Clock;
use chrono::{Datelike, NaiveDate, NaiveTime, Timelike};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date(NaiveDate);

/// Hour and minute only: the two fields `update_time` collects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Time(NaiveTime);

/// CPython's `datetime` wording, interpolated into the tool's reply. CPython 3.13 rewords the day
/// message; the pinned interpreter's older text is kept.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DateError {
    #[error("year {0} is out of range")]
    YearOutOfRange(i64),
    #[error("month must be in 1..12")]
    MonthOutOfRange,
    #[error("day is out of range for month")]
    DayOutOfRange,
    #[error("hour must be in 0..23")]
    HourOutOfRange,
    #[error("minute must be in 0..59")]
    MinuteOutOfRange,
}

impl Date {
    /// `i64` because the arguments are arbitrary JSON integers. Checked in CPython's order, so the
    /// first bad component is the one named.
    pub fn new(year: i64, month: i64, day: i64) -> Result<Date, DateError> {
        if !(1..=9999).contains(&year) {
            return Err(DateError::YearOutOfRange(year));
        }
        if !(1..=12).contains(&month) {
            return Err(DateError::MonthOutOfRange);
        }
        let day = u32::try_from(day).map_err(|_| DateError::DayOutOfRange)?;
        NaiveDate::from_ymd_opt(year as i32, month as u32, day).map(Date).ok_or(DateError::DayOutOfRange)
    }

    pub fn year(self) -> i32 {
        self.0.year()
    }
    pub fn month(self) -> u32 {
        self.0.month()
    }
    pub fn day(self) -> u32 {
        self.0.day()
    }

    /// `strftime("%B %d, %Y")` -- `January 15, 1990`. `%d` is zero-padded, so day 5 is `05`; the
    /// year is not (chrono's `%Y` would pad year 5 to `0005`).
    pub fn format_long(self) -> String {
        format!("{} {:02}, {}", self.0.format("%B"), self.day(), self.year())
    }
}

impl Time {
    pub fn new(hour: i64, minute: i64) -> Result<Time, DateError> {
        if !(0..=23).contains(&hour) {
            return Err(DateError::HourOutOfRange);
        }
        if !(0..=59).contains(&minute) {
            return Err(DateError::MinuteOutOfRange);
        }
        Ok(Time(NaiveTime::from_hms_opt(hour as u32, minute as u32, 0).expect("checked above")))
    }

    pub fn hour(self) -> u32 {
        self.0.hour()
    }
    pub fn minute(self) -> u32 {
        self.0.minute()
    }

    /// `strftime("%I:%M %p")` -- `03:05 PM`. Midnight is `12:00 AM`, noon is `12:00 PM`.
    pub fn format_12h(self) -> String {
        self.0.format("%I:%M %p").to_string()
    }
}

#[derive(Clone)]
pub struct TodaySource {
    clock: Arc<dyn Clock>,
    /// Seconds east of UTC.
    utc_offset_secs: i32,
}

impl std::fmt::Debug for TodaySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TodaySource")
            .field("today", &self.today())
            .field("utc_offset_secs", &self.utc_offset_secs)
            .finish()
    }
}

impl TodaySource {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        TodaySource { clock, utc_offset_secs: 0 }
    }

    pub fn with_utc_offset(clock: Arc<dyn Clock>, utc_offset_secs: i32) -> Self {
        TodaySource { clock, utc_offset_secs }
    }

    pub fn today(&self) -> Date {
        let t = self.clock.wall();
        let secs = match t.duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_secs() as i64,
            // Before 1970: negative rather than a panic on a host-supplied value.
            Err(e) => -(e.duration().as_secs() as i64),
        };
        let local = secs + i64::from(self.utc_offset_secs);
        Date(chrono::DateTime::from_timestamp(local, 0).unwrap_or_default().date_naive())
    }
}

impl Default for TodaySource {
    /// Tests pass a `VirtualClock`.
    fn default() -> Self {
        TodaySource::new(Arc::new(crate::RealClock::new()))
    }
}

#[cfg(feature = "python")]
mod host {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyDate, PyTime};

    use super::{Date, Time};

    impl<'py> IntoPyObject<'py> for Date {
        type Target = PyDate;
        type Output = Bound<'py, PyDate>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyDate>> {
            PyDate::new(py, self.year(), self.month() as u8, self.day() as u8)
        }
    }

    impl<'py> IntoPyObject<'py> for Time {
        type Target = PyTime;
        type Output = Bound<'py, PyTime>;
        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyTime>> {
            PyTime::new(py, self.hour() as u8, self.minute() as u8, 0, 0, None)
        }
    }

    /// Validated rather than saturated.
    impl<'a, 'py> FromPyObject<'a, 'py> for Date {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let part = |name: &str| -> PyResult<i64> { obj.getattr(name)?.extract() };
            Date::new(part("year")?, part("month")?, part("day")?).map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }

    /// To the minute.
    impl<'a, 'py> FromPyObject<'a, 'py> for Time {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let part = |name: &str| -> PyResult<i64> { obj.getattr(name)?.extract() };
            Time::new(part("hour")?, part("minute")?).map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }
}
