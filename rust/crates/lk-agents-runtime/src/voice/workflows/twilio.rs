//! Twilio Calls API HTTP requests. Transfer decisions remain in the voice core.

use lk_agents_abi::voice::workflows::{TwilioCallEnd, TwilioDial};
use lk_agents_voice::workflows::twilio::TwilioError;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

const API_BASE: &str = "https://api.twilio.com/2010-04-01";

pub struct Request {
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub form: Vec<(String, String)>,
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

pub fn create_call(dial: &TwilioDial, from: String, twiml: &str, token: Option<String>) -> Request {
    let mut form = vec![
        ("To".to_owned(), dial.phone_number.clone()),
        ("From".to_owned(), from),
        ("Twiml".to_owned(), twiml.to_owned()),
    ];
    form.extend(token.map(|token| ("CallToken".to_owned(), token)));
    request(dial, "/Calls.json", form)
}

pub fn end_call(dial: &TwilioDial, sid: &str, status: TwilioCallEnd) -> Request {
    let path = format!("/Calls/{}.json", quote(sid));
    request(dial, &path, vec![("Status".to_owned(), status.as_str().to_owned())])
}

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

pub fn call_sid(call: &Map<String, Value>) -> Result<String, TwilioError> {
    match call.get("sid").and_then(Value::as_str) {
        Some(sid) if !sid.is_empty() => Ok(sid.to_owned()),
        _ => Err(TwilioError::NoCallSid),
    }
}

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

#[cfg(test)]
mod tests {
    #[test]
    fn account_and_call_path_segments_are_escaped() {
        assert_eq!(super::quote("AC12/3"), "AC12%2F3");
    }
}
