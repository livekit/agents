use crate::workflows::ToolError;
use crate::workflows::builtins::instructions::render;
use crate::workflows::builtins::name::GetNameResult;
use crate::workflows::builtins::prompts::credit_card as p;
pub use lk_agents_abi::voice::workflows::{
    GetCardNumberResult, GetCreditCardResult, GetExpirationDateResult, GetSecurityCodeResult,
};
pub const CARD_ISSUERS: [(char, &str); 4] =
    [('3', "American Express"), ('4', "Visa"), ('5', "Mastercard"), ('6', "Discover")];
pub const CARD_CAPTURE_DECLINED: &str = "CardCaptureDeclinedError";
pub const CARD_COLLECTION_RESTART: &str = "CardCollectionRestartError";

pub fn issuer_for(card_number: &str) -> &'static str {
    let Some(first) = card_number.chars().next() else { return "Other" };
    CARD_ISSUERS.iter().find(|(d, _)| *d == first).map(|(_, n)| *n).unwrap_or("Other")
}

pub fn assemble_credit_card_result(
    cardholder: &GetNameResult,
    card_number: &GetCardNumberResult,
    security_code: &GetSecurityCodeResult,
    expiration_date: &GetExpirationDateResult,
) -> GetCreditCardResult {
    let first = cardholder.first_name.as_deref().unwrap_or("None");
    let last = cardholder.last_name.as_deref().unwrap_or("None");
    GetCreditCardResult {
        cardholder_name: format!("{first} {last}"),
        issuer: card_number.issuer.clone(),
        card_number: card_number.card_number.clone(),
        security_code: security_code.security_code.clone(),
        expiration_date: expiration_date.date.clone(),
    }
}

#[lk_agents_abi::host(export(pid = false))]
pub fn luhn_valid(card_number: &str) -> bool {
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

#[lk_agents_abi::host(export(pid = false))]
pub fn card_error_class_names() -> (&'static str, &'static str) {
    (CARD_CAPTURE_DECLINED, CARD_COLLECTION_RESTART)
}

#[lk_agents_abi::host(export(name = "assemble_credit_card", pid = false))]
pub fn assemble_credit_card(
    cardholder: &GetNameResult,
    card_number: &GetCardNumberResult,
    security_code: &GetSecurityCodeResult,
    expiration_date: &GetExpirationDateResult,
) -> GetCreditCardResult {
    assemble_credit_card_result(cardholder, card_number, security_code, expiration_date)
}

pub fn card_capture_declined(reason: String) -> ToolError {
    ToolError::subclass(CARD_CAPTURE_DECLINED, render(p::DECLINED, &[("reason", &reason)]), reason)
}

pub fn card_collection_restart(reason: String) -> ToolError {
    ToolError::subclass(
        CARD_COLLECTION_RESTART,
        render(p::RESTARTING, &[("reason", &reason)]),
        reason,
    )
}

pub fn digits_only(s: &str) -> String {
    s.chars().filter(char::is_ascii_digit).collect()
}
