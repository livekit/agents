//! What the Python `beta/workflows` tasks read from the core: the shared prompts, the DTMF
//! collector, the warm transfer plan, and the credit card helpers below.

pub mod dtmf;
pub mod instructions;
pub mod prompts;
pub mod warm_transfer;

use lk_agents_abi::voice::workflows::{
    GetCardNumberResult, GetCreditCardResult, GetExpirationDateResult, GetNameResult,
    GetSecurityCodeResult,
};

pub const CARD_ISSUERS: [(char, &str); 4] =
    [('3', "American Express"), ('4', "Visa"), ('5', "Mastercard"), ('6', "Discover")];
pub const CARD_CAPTURE_DECLINED: &str = "CardCaptureDeclinedError";
pub const CARD_COLLECTION_RESTART: &str = "CardCollectionRestartError";

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
