//! `beta/workflows` and `beta/tools`: the checks a capture tool runs on what the model heard
//! (card numbers and expiries, dates of birth, email addresses, phone numbers), DTMF reception and
//! sending, the warm transfer, and the call ended after its goodbye.

pub mod dtmf;
pub mod end_call;
pub mod instructions;
pub mod prompts;
pub mod twilio;
pub mod warm_transfer;

use crate::tokenize::{is_py_space, py_strip};
use lk_agents_abi::utils::dates::Date;
use lk_agents_abi::voice::workflows::{CaptureRefused, UnknownDtmfSymbol, WorkflowChecks};

/// `validate_card_number`: the Luhn checksum over an all-digit string.
pub fn validate_card_number(card_number: &str) -> bool {
    if card_number.is_empty() || !card_number.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let total: u32 = card_number
        .chars()
        .rev()
        .enumerate()
        .map(|(index, c)| {
            let digit = c.to_digit(10).expect("checked ascii above");
            if index % 2 == 1 {
                let doubled = digit * 2;
                if doubled > 9 { doubled - 9 } else { doubled }
            } else {
                digit
            }
        })
        .sum();
    total.is_multiple_of(10)
}

/// `is_expired(month, year)`: whether a two-digit expiry year and its month precede `today`'s
/// month.
pub fn card_expired(month: i64, year: i64, today: Date) -> bool {
    (2000 + year, month) < (i64::from(today.year()), i64::from(today.month()))
}

pub use lk_agents_abi::voice::workflows::DateOfBirthRefused;

/// `normalize_date_of_birth`: a two-digit year is this century's when it is not past this year's
/// two digits, else the last century's; an invalid or future date is refused.
pub fn normalize_date_of_birth(
    year: i64,
    month: i64,
    day: i64,
    today: Date,
) -> Result<Date, DateOfBirthRefused> {
    let year = match year {
        0..=99 if year <= i64::from(today.year()) % 100 => year + 2000,
        0..=99 => year + 1900,
        _ => year,
    };
    let value = Date::new(year, month, day)?;
    if value > today {
        return Err(DateOfBirthRefused::Future);
    }
    Ok(value)
}

/// `normalize_email_address`: stripped, and refused unless it reads `local@label.….tld` (the
/// reference's `EMAIL_REGEX`: a local part starting alphanumeric over `[A-Za-z0-9._%+-]`, labels
/// alphanumeric at both ends with inner hyphens, and an alphabetic TLD of two or more).
pub fn normalize_email_address(email: &str) -> Result<String, String> {
    let email = py_strip(email);
    let valid = email.split_once('@').is_some_and(|(local, domain)| {
        let alnum = |c: char| c.is_ascii_alphanumeric();
        let local_ok = local.chars().next().is_some_and(alnum)
            && local.chars().all(|c| alnum(c) || "._%+-".contains(c));
        let labels: Vec<&str> = domain.split('.').collect();
        let (tld, labels) = labels.split_last().expect("split yields one part at least");
        let label_ok = |l: &&str| {
            l.starts_with(alnum) && l.ends_with(alnum) && l.chars().all(|c| alnum(c) || c == '-')
        };
        local_ok
            && !labels.is_empty()
            && labels.iter().all(label_ok)
            && tld.len() >= 2
            && tld.chars().all(|c| c.is_ascii_alphabetic())
    });
    if valid {
        Ok(email.to_owned())
    } else {
        Err(format!("Invalid email address provided: {email}"))
    }
}

/// `normalize_phone_number`: whitespace, hyphens, parentheses and dots dropped, and refused unless
/// it reads `+?[1-9]` followed by six to fourteen digits (the reference's `PHONE_REGEX`).
pub fn normalize_phone_number(phone_number: &str) -> Result<String, String> {
    let cleaned: String = py_strip(phone_number)
        .chars()
        .filter(|c| !(is_py_space(*c) || "-().".contains(*c)))
        .collect();
    let digits = cleaned.strip_prefix('+').unwrap_or(&cleaned);
    let valid = digits.starts_with(|c: char| ('1'..='9').contains(&c))
        && (7..=15).contains(&digits.len())
        && digits.chars().all(|c| c.is_ascii_digit());
    if valid { Ok(cleaned) } else { Err(format!("Invalid phone number provided: {phone_number}")) }
}

/// The capture tools' checks and DTMF symbols, as the host calls them.
pub struct WorkflowRules;

impl WorkflowChecks for WorkflowRules {
    fn validate_card_number(card_number: String) -> bool {
        validate_card_number(&card_number)
    }

    fn card_expired(month: i64, year: i64, today: Date) -> bool {
        card_expired(month, year, today)
    }

    fn normalize_date_of_birth(
        year: i64,
        month: i64,
        day: i64,
        today: Date,
    ) -> Result<Date, DateOfBirthRefused> {
        normalize_date_of_birth(year, month, day, today)
    }

    fn normalize_email_address(email: String) -> Result<String, CaptureRefused> {
        normalize_email_address(&email).map_err(CaptureRefused)
    }

    fn normalize_phone_number(phone_number: String) -> Result<String, CaptureRefused> {
        normalize_phone_number(&phone_number).map_err(CaptureRefused)
    }

    fn dtmf_code(symbol: String) -> Result<u8, UnknownDtmfSymbol> {
        dtmf::dtmf_code(&symbol).map_err(|_| UnknownDtmfSymbol(symbol))
    }

    fn format_dtmf(symbols: Vec<String>) -> Result<String, UnknownDtmfSymbol> {
        // the space-joined digits that fill `prompts::dtmf`'s `{digits}` hole
        let keys = dtmf::parse_all(&symbols, str::to_owned).map_err(UnknownDtmfSymbol)?;
        Ok(dtmf::format_dtmf(&keys))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i64, m: i64, d: i64) -> Date {
        Date::new(y, m, d).unwrap()
    }

    #[test]
    fn email_addresses_read_as_the_reference_regex() {
        assert_eq!(normalize_email_address("  theo@livekit.io ").unwrap(), "theo@livekit.io");
        assert!(normalize_email_address("a.b+c@mail.example.co.uk").is_ok());
        for bad in [
            "@x.io",
            "theo@livekit",
            "theo@-x.io",
            "theo@x-.io",
            "theo@x.i",
            "a@b@c.io",
            ".a@x.io",
            "t@x..io",
        ] {
            assert!(normalize_email_address(bad).is_err(), "{bad}");
        }
        assert_eq!(
            normalize_email_address("nope").unwrap_err(),
            "Invalid email address provided: nope"
        );
    }

    #[test]
    fn phone_numbers_lose_their_punctuation() {
        assert_eq!(normalize_phone_number(" +1 (510) 555-0123 ").unwrap(), "+15105550123");
        assert_eq!(
            normalize_phone_number("06.12.34.56.78").unwrap_err(),
            "Invalid phone number provided: 06.12.34.56.78"
        );
        assert!(normalize_phone_number("123456").is_err(), "too short");
        assert!(normalize_phone_number("1234567890123456").is_err(), "too long");
    }

    #[test]
    fn two_digit_birth_years_pick_their_century() {
        let today = date(2026, 10, 26);
        assert_eq!(normalize_date_of_birth(26, 1, 1, today).unwrap(), date(2026, 1, 1));
        assert_eq!(normalize_date_of_birth(90, 5, 2, today).unwrap(), date(1990, 5, 2));
        assert_eq!(normalize_date_of_birth(2026, 12, 1, today), Err(DateOfBirthRefused::Future));
        assert_eq!(
            normalize_date_of_birth(1990, 2, 30, today).unwrap_err().to_string(),
            "day is out of range for month"
        );
    }

    #[test]
    fn cards_expire_after_their_month() {
        let today = date(2026, 10, 26);
        assert!(card_expired(9, 26, today));
        assert!(!card_expired(10, 26, today));
        assert!(validate_card_number("4242424242424242"));
        assert!(!validate_card_number("4242424242424241"));
        assert!(!validate_card_number("4242 4242"));
    }
}
