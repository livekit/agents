//! `datetime.date`, valid by construction.

use chrono::{Datelike, NaiveDate};

#[crate::host(handwritten(extract, emit, host = "datetime.date", py))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date(NaiveDate);

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
        NaiveDate::from_ymd_opt(year as i32, month as u32, day)
            .map(Date)
            .ok_or(DateError::DayOutOfRange)
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
