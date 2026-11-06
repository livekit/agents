//! The warm transfer's Twilio caller selection, reply decisions, and connector TwiML.
//! The runtime builds and sends HTTP requests and owns the cancel-on-drop guard.

use lk_agents_abi::voice::workflows::TwilioDial;

/// Twilio refused the caller id preserved from the original call (21210, 21212): no call exists.
const REJECTED_CALLER_IDS: [i64; 2] = [21210, 21212];
/// A cancel that lost the race with the callee answering (21220): the call is completed instead.
const NOT_CANCELLABLE: i64 = 21220;

/// Why a Twilio request did not do what the dial needed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TwilioError {
    #[error("Twilio request failed (HTTP {status}, code {})", code.map_or("unknown".to_owned(), |c| c.to_string()))]
    Status { status: u16, code: Option<i64> },
    #[error("Twilio call creation returned no call SID")]
    NoCallSid,
}

/// The first call's caller id: the original caller's number when a call token preserves it, else
/// the business number. The token goes with it.
pub fn first_caller(dial: &TwilioDial) -> (String, Option<String>) {
    match dial.call_token.clone().filter(|token| !token.is_empty()) {
        Some(token) => (dial.original_caller_number.clone().unwrap_or_default(), Some(token)),
        None => (dial.from_number.clone(), None),
    }
}

/// The caller id to retry with after `error`: the business number, when Twilio rejected the
/// preserved one.
pub fn retry_caller(dial: &TwilioDial, token: Option<&str>, error: &TwilioError) -> Option<String> {
    let rejected = matches!(error, TwilioError::Status { status: 400, code: Some(code) }
        if REJECTED_CALLER_IDS.contains(code));
    (rejected && token.is_some()).then(|| dial.from_number.clone())
}

/// Whether a failed cancel means the call was answered meanwhile and must be completed.
pub fn complete_instead(error: &TwilioError) -> bool {
    matches!(error, TwilioError::Status { status: 400, code: Some(NOT_CANCELLABLE) })
}

/// The call's TwiML: a stream to the connector's URL.
pub fn twiml(connect_url: &str) -> String {
    format!("<Response><Connect><Stream url={}/></Connect></Response>", quoteattr(connect_url))
}

/// `xml.sax.saxutils.quoteattr`: escaped and quoted, in single quotes when the value holds a
/// double quote but no single one.
fn quoteattr(s: &str) -> String {
    let escaped = s
        .replace('&', "&amp;")
        .replace('>', "&gt;")
        .replace('<', "&lt;")
        .replace('\n', "&#10;")
        .replace('\r', "&#13;")
        .replace('\t', "&#9;");
    match (escaped.contains('"'), escaped.contains('\'')) {
        (true, true) => format!("\"{}\"", escaped.replace('"', "&quot;")),
        (true, false) => format!("'{escaped}'"),
        _ => format!("\"{escaped}\""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twiml_attributes_are_quoted_as_saxutils() {
        assert_eq!(quoteattr("wss://x/y?a=1&b=2"), "\"wss://x/y?a=1&amp;b=2\"");
        assert_eq!(quoteattr("a\"b"), "'a\"b'");
        assert_eq!(quoteattr("a\"b'c"), "\"a&quot;b'c\"");
    }
}
