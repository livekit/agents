//! `beta/workflows` and `beta/tools`: the checks a capture tool runs on what the model heard
//! (card numbers and expiries, dates of birth, email addresses, phone numbers), DTMF reception and
//! sending, the warm transfer, and the call ended after its goodbye.

use lk_agents_abi::utils::dates::Date;
use lk_agents_abi::voice::workflows::{
    CaptureRefused, DateOfBirthRefused, UnknownDtmfSymbol, WorkflowChecks,
};
use lk_agents_core::tokenize::{is_py_space, py_strip};
use lk_agents_voice::generation::{Rendered, safe_render};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub mod dtmf;
pub mod effects;
pub mod end_call;
pub mod twilio;
pub mod warm_transfer;

/// The capture tools' checks and DTMF symbols, as the host calls them.
pub struct WorkflowRules;

impl WorkflowChecks for WorkflowRules {
    fn dtmf_capture_plan(confirmation: bool) -> lk_agents_abi::voice::workflows::CapturePlan {
        use lk_agents_abi::voice::workflows::{CapturePlan, prompts};
        let (name, instructions, tool, description) = if confirmation {
            (
                "get_dtmf_confirmed",
                prompts::DTMF_CONFIRMER_INSTRUCTIONS,
                "confirm_inputs",
                "Finalize the collected digit inputs after explicit user confirmation.\n\nUse this ONLY after the confirmation. You should confirm by verbally reading out the digits one by one and, once the user confirms they are correct, call this tool with the inputs. Do not use this tool to capture the initial digits.",
            )
        } else {
            (
                "get_dtmf",
                prompts::DTMF_RECORDER_INSTRUCTIONS,
                "record_inputs",
                "Record the collected digit inputs without additional confirmation.\n\nCall this tool as soon as a valid sequence of digits has been provided by the user (via DTMF or spoken).",
            )
        };
        CapturePlan {
            agent_name: name.into(),
            instructions: instructions.into(),
            tool_name: tool.into(),
            tool_description: description.into(),
        }
    }

    fn email_capture_plan() -> lk_agents_abi::voice::workflows::CapturePlan {
        lk_agents_abi::voice::workflows::CapturePlan {
            agent_name: "get_email".into(),
            instructions: lk_agents_abi::voice::workflows::prompts::EMAIL_COLLECTOR_INSTRUCTIONS
                .into(),
            tool_name: "confirm_email_address".into(),
            tool_description:
                "Record the caller's email address; they confirm it before it is kept.".into(),
        }
    }
    fn email_read_back(email_address: String) -> String {
        lk_agents_abi::voice::workflows::prompts::EMAIL_READ_BACK
            .replace("{email_address}", &email_address)
    }
    fn validate_card_number(card_number: String) -> bool {
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

    fn card_expired(month: i64, year: i64, today: Date) -> bool {
        (2000 + year, month) < (i64::from(today.year()), i64::from(today.month()))
    }

    fn normalize_date_of_birth(
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

    /// The reference's `EMAIL_REGEX`: a local part starting alphanumeric over
    /// `[A-Za-z0-9._%+-]`, labels alphanumeric at both ends with inner hyphens, and an alphabetic
    /// TLD of two or more.
    fn normalize_email_address(email: String) -> Result<String, CaptureRefused> {
        let email = py_strip(&email);
        let valid = email.split_once('@').is_some_and(|(local, domain)| {
            let alnum = |c: char| c.is_ascii_alphanumeric();
            let local_ok = local.chars().next().is_some_and(alnum)
                && local.chars().all(|c| alnum(c) || "._%+-".contains(c));
            let labels: Vec<&str> = domain.split('.').collect();
            let (tld, labels) = labels.split_last().expect("split yields one part at least");
            let label_ok = |l: &&str| {
                l.starts_with(alnum)
                    && l.ends_with(alnum)
                    && l.chars().all(|c| alnum(c) || c == '-')
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
            Err(CaptureRefused(format!("Invalid email address provided: {email}")))
        }
    }

    /// The reference's `PHONE_REGEX`.
    fn normalize_phone_number(phone_number: String) -> Result<String, CaptureRefused> {
        let cleaned: String = py_strip(&phone_number)
            .chars()
            .filter(|c| !(is_py_space(*c) || "-().".contains(*c)))
            .collect();
        let digits = cleaned.strip_prefix('+').unwrap_or(&cleaned);
        let valid = digits.starts_with(|c: char| ('1'..='9').contains(&c))
            && (7..=15).contains(&digits.len())
            && digits.chars().all(|c| c.is_ascii_digit());
        if valid {
            Ok(cleaned)
        } else {
            Err(CaptureRefused(format!("Invalid phone number provided: {phone_number}")))
        }
    }

    fn dtmf_code(symbol: String) -> Result<u8, UnknownDtmfSymbol> {
        dtmf::dtmf_code(&symbol).map_err(|_| UnknownDtmfSymbol(symbol))
    }

    fn format_dtmf(symbols: Vec<String>) -> Result<String, UnknownDtmfSymbol> {
        // the space-joined digits that fill the DTMF read-back's `{digits}` hole
        let keys = dtmf::parse_all(&symbols, str::to_owned).map_err(UnknownDtmfSymbol)?;
        Ok(dtmf::format_dtmf(&keys))
    }
}

/// `str.format` over a built-in template, dropping `missing` ([`render_checked`] proves there are none).
pub fn render(template: &str, data: &[(&str, &str)]) -> String {
    render_checked(template, data).text
}

pub fn render_checked(template: &str, data: &[(&str, &str)]) -> Rendered {
    let map: BTreeMap<&str, &str> = data.iter().copied().collect();
    let obj: Map<String, Value> =
        map.into_iter().map(|(k, v)| (k.to_owned(), Value::String(v.to_owned()))).collect();
    safe_render(template, &Value::Object(obj))
}

#[cfg(test)]
mod tests {
    use super::*;

    type R = WorkflowRules;

    fn date(y: i64, m: i64, d: i64) -> Date {
        Date::new(y, m, d).unwrap()
    }

    #[test]
    fn email_addresses_read_as_the_reference_regex() {
        assert_eq!(
            R::normalize_email_address("  theo@livekit.io ".into()).unwrap(),
            "theo@livekit.io"
        );
        assert!(R::normalize_email_address("a.b+c@mail.example.co.uk".into()).is_ok());
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
            assert!(R::normalize_email_address(bad.into()).is_err(), "{bad}");
        }
        assert_eq!(
            R::normalize_email_address("nope".into()).unwrap_err().to_string(),
            "Invalid email address provided: nope"
        );
    }

    #[test]
    fn phone_numbers_lose_their_punctuation() {
        assert_eq!(
            R::normalize_phone_number(" +1 (510) 555-0123 ".into()).unwrap(),
            "+15105550123"
        );
        assert_eq!(
            R::normalize_phone_number("06.12.34.56.78".into()).unwrap_err().to_string(),
            "Invalid phone number provided: 06.12.34.56.78"
        );
        assert!(R::normalize_phone_number("123456".into()).is_err(), "too short");
        assert!(R::normalize_phone_number("1234567890123456".into()).is_err(), "too long");
    }

    #[test]
    fn two_digit_birth_years_pick_their_century() {
        let today = date(2026, 10, 26);
        assert_eq!(R::normalize_date_of_birth(26, 1, 1, today).unwrap(), date(2026, 1, 1));
        assert_eq!(R::normalize_date_of_birth(90, 5, 2, today).unwrap(), date(1990, 5, 2));
        assert_eq!(R::normalize_date_of_birth(2026, 12, 1, today), Err(DateOfBirthRefused::Future));
        assert_eq!(
            R::normalize_date_of_birth(1990, 2, 30, today).unwrap_err().to_string(),
            "day is out of range for month"
        );
    }

    #[test]
    fn cards_expire_after_their_month() {
        let today = date(2026, 10, 26);
        assert!(R::card_expired(9, 26, today));
        assert!(!R::card_expired(10, 26, today));
        assert!(R::validate_card_number("4242424242424242".into()));
        assert!(!R::validate_card_number("4242424242424241".into()));
        assert!(!R::validate_card_number("4242 4242".into()));
    }
}
