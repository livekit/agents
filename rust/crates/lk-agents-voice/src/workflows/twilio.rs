//! The warm transfer's Twilio dial: Calls API requests, how replies and error codes are read, and
//! the TwiML that streams the call into the LiveKit connector. The runtime sends the requests
//! through the host and owns the cancel-on-drop guard.

use lk_agents_abi::voice::workflows::TwilioDial;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

const API_BASE: &str = "https://api.twilio.com/2010-04-01";

/// Twilio refused the caller id preserved from the original call (21210, 21212): no call exists.
const REJECTED_CALLER_IDS: [i64; 2] = [21210, 21212];
/// A cancel that lost the race with the callee answering (21220): the call is completed instead.
const NOT_CANCELLABLE: i64 = 21220;

/// A form `POST` under the dial's account, with basic auth.
pub struct Request {
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub form: Vec<(String, String)>,
}

/// Why a Twilio request did not do what the dial needed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TwilioError {
    #[error("Twilio request failed (HTTP {status}, code {})", code.map_or("unknown".to_owned(), |c| c.to_string()))]
    Status { status: u16, code: Option<i64> },
    #[error("Twilio call creation returned no call SID")]
    NoCallSid,
}

fn request(dial: &TwilioDial, path: &str, form: Vec<(String, String)>) -> Request {
    use base64::Engine as _;
    let auth = base64::engine::general_purpose::STANDARD
        .encode(format!("{}:{}", dial.account_sid, dial.auth_token));
    Request {
        url: format!("{API_BASE}/Accounts/{}{path}", quote(&dial.account_sid)),
        headers: BTreeMap::from([("Authorization".to_owned(), format!("Basic {auth}"))]),
        form,
    }
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

/// Create the outbound call, streaming `twiml`.
pub fn create_call(dial: &TwilioDial, from: String, twiml: &str, token: Option<String>) -> Request {
    let mut form = vec![
        ("To".to_owned(), dial.phone_number.clone()),
        ("From".to_owned(), from),
        ("Twiml".to_owned(), twiml.to_owned()),
    ];
    form.extend(token.map(|token| ("CallToken".to_owned(), token)));
    request(dial, "/Calls.json", form)
}

/// Set the call's status: `canceled` to give it up, `completed` to end an answered one.
pub fn set_status(dial: &TwilioDial, sid: &str, status: &str) -> Request {
    let path = format!("/Calls/{}.json", quote(sid));
    request(dial, &path, vec![("Status".to_owned(), status.to_owned())])
}

/// Whether a failed cancel means the call was answered meanwhile and must be completed.
pub fn complete_instead(error: &TwilioError) -> bool {
    matches!(error, TwilioError::Status { status: 400, code: Some(NOT_CANCELLABLE) })
}

/// A reply's JSON object (`{}` when the body is not one), or its failure.
pub fn reply(status: u16, body: &str) -> Result<Map<String, Value>, TwilioError> {
    let body = match serde_json::from_str(body) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    };
    if (200..300).contains(&status) {
        Ok(body)
    } else {
        Err(TwilioError::Status { status, code: body.get("code").and_then(Value::as_i64) })
    }
}

/// The created call's SID.
pub fn call_sid(call: &Map<String, Value>) -> Result<String, TwilioError> {
    match call.get("sid").and_then(Value::as_str) {
        Some(sid) if !sid.is_empty() => Ok(sid.to_owned()),
        _ => Err(TwilioError::NoCallSid),
    }
}

/// The call's TwiML: a stream to the connector's URL.
pub fn twiml(connect_url: &str) -> String {
    format!("<Response><Connect><Stream url={}/></Connect></Response>", quoteattr(connect_url))
}

/// `urllib.parse.quote(s, safe="")`.
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
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
        assert_eq!(quote("AC12/3"), "AC12%2F3");
    }
}
