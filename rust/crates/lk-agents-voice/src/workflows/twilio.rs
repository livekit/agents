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

/// How a call creation ended, as the dial heard it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Creation {
    Created(String),
    Failed(TwilioError),
    /// The request may have reached Twilio, but its answer was lost.
    Lost,
}

/// What a dial owes a call creation it gives up on.
#[derive(Debug, PartialEq, Eq)]
pub enum Abandon {
    /// Twilio made no call.
    Nothing,
    /// The call is known: cancel it.
    Cancel(String),
    /// Twilio may have made a call the dial cannot name. Only the transfer's own human-agent
    /// room is that call's: deleting it ends the call once answered, and no other transfer's.
    DeleteRoom,
}

/// A created call is cancelled; a refusal made none; a lost answer, a reply without a SID or a
/// server error may follow a call Twilio committed.
pub fn abandon(creation: Creation) -> Abandon {
    match creation {
        Creation::Created(sid) => Abandon::Cancel(sid),
        Creation::Lost | Creation::Failed(TwilioError::NoCallSid) => Abandon::DeleteRoom,
        Creation::Failed(TwilioError::Status { status, .. }) if status >= 500 => {
            Abandon::DeleteRoom
        }
        Creation::Failed(TwilioError::Status { .. }) => Abandon::Nothing,
    }
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

    // Catches: a creation whose answer was lost left ringing, or a refused one answered with a
    // cleanup that could reach another transfer's call.
    #[test]
    fn only_a_creation_that_may_have_made_an_unnamed_call_deletes_the_room() {
        let refused = TwilioError::Status { status: 400, code: Some(21210) };
        let failed = TwilioError::Status { status: 503, code: None };
        assert_eq!(abandon(Creation::Created("CA1".into())), Abandon::Cancel("CA1".into()));
        assert_eq!(abandon(Creation::Lost), Abandon::DeleteRoom);
        assert_eq!(abandon(Creation::Failed(TwilioError::NoCallSid)), Abandon::DeleteRoom);
        assert_eq!(abandon(Creation::Failed(failed)), Abandon::DeleteRoom);
        assert_eq!(abandon(Creation::Failed(refused)), Abandon::Nothing);
    }

    #[test]
    fn twiml_attributes_are_quoted_as_saxutils() {
        assert_eq!(quoteattr("wss://x/y?a=1&b=2"), "\"wss://x/y?a=1&amp;b=2\"");
        assert_eq!(quoteattr("a\"b"), "'a\"b'");
        assert_eq!(quoteattr("a\"b'c"), "\"a&quot;b'c\"");
    }
}
