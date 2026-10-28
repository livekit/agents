//! The outcome table (tools.md "What a tool returns"): the call output a [`ToolOutcome`] records,
//! whether it asks the model for a reply, and the text a verbatim result delivers. One table for
//! direct runs, TextSession and voice; the host only classifies what the body returned or raised.

use lk_agents_abi::llm::tool_executor::{
    DELIVERED, INTERNAL_ERROR_OUTPUT, INVALID_OUTPUT, ToolOutcome, WAITING_PREFIX,
};

/// The output a [`ToolOutcome`] records, and whether it asks the model for a reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolShape {
    pub text: String,
    pub is_error: bool,
    pub reply_required: bool,
}

/// Which coordinator reads the table. The two differ in one row only, a `None` return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// `agent.run()` and TextSession: `None` is a value the model reads and continues from
    /// (tools.md "What a tool returns").
    Text,
    /// A call: a silent tool asks for no follow-up reply, the reference's
    /// `reply_required = fnc_out is not None`. Interim until the owner rules on the `None` row for
    /// calls (pending question, OVERNIGHT.md; settled with E-3).
    Voice,
}

/// `make_tool_output` and the transitions as one table. A verbatim result's delivery is the
/// coordinator's: [`delivered`] is what the model reads once it was said.
pub fn shape(outcome: &ToolOutcome, mode: Mode) -> ToolShape {
    let shape = |text: &str, is_error: bool, reply_required: bool| ToolShape {
        text: text.to_owned(),
        is_error,
        reply_required,
    };
    match outcome {
        ToolOutcome::Value { text, reply_required, .. } => shape(text, false, *reply_required),
        ToolOutcome::Nothing => shape("", false, mode == Mode::Text),
        ToolOutcome::StopResponse | ToolOutcome::Cancelled => shape("", false, false),
        // `ModelRetry` is feedback, as a `ToolError` is: the model reads the reason and calls
        // again. The retry budget is the invocation's.
        ToolOutcome::ToolError(message) | ToolOutcome::ModelRetry(message) => {
            shape(message, true, true)
        }
        ToolOutcome::Internal => shape(INTERNAL_ERROR_OUTPUT, true, true),
        ToolOutcome::Invalid => shape(INVALID_OUTPUT, true, true),
        ToolOutcome::Update { message, .. } => shape(message, false, true),
        ToolOutcome::Completed { text, .. } => shape(text, false, true),
        ToolOutcome::Failed(text) | ToolOutcome::Rejected(text) => shape(text, true, true),
        // The question is asked, not replied to; the person's answer replaces it.
        ToolOutcome::InputRequired { question, .. } => {
            shape(&format!("{WAITING_PREFIX}{question}"), false, false)
        }
        ToolOutcome::Working(message) => {
            shape(message.as_deref().unwrap_or("working"), false, false)
        }
    }
}

/// The text the person gets exactly as written, when the outcome is verbatim.
pub fn verbatim(outcome: &ToolOutcome) -> Option<&str> {
    match outcome {
        ToolOutcome::Value { text, verbatim: true, .. }
        | ToolOutcome::Completed { text, verbatim: true } => Some(text),
        _ => None,
    }
}

/// What the model reads once the verbatim text was said, then a delegate's note for the model.
pub fn delivered(outcome: &ToolOutcome) -> Option<String> {
    let text = verbatim(outcome)?;
    Some(match outcome {
        ToolOutcome::Value { more: Some(more), .. } => format!("{DELIVERED}{text}\n{more}"),
        _ => format!("{DELIVERED}{text}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a `None` return read as asking for no reply, a transition read as silent (its output lost), `Failed`/`Rejected` not errors, an
    // `InputRequired` with no question, `ToolResult(reply_required=False)` losing its text,
    // `ModelRetry` hidden as an internal error, or exception detail reaching the model.
    #[test]
    fn each_outcome_records_its_output_and_asks_for_a_reply() {
        let value = |text: &str, reply_required| ToolOutcome::Value {
            text: text.into(),
            reply_required,
            verbatim: false,
            more: None,
        };
        let ask = ToolOutcome::InputRequired {
            question: "Which day?".into(),
            verbatim: false,
            tool: None,
        };
        let cases: &[(ToolOutcome, (&str, bool, bool))] = &[
            (value("12C", true), ("12C", false, true)),
            (value("sent DTMF 1", false), ("sent DTMF 1", false, false)),
            (ToolOutcome::Nothing, ("", false, true)),
            (ToolOutcome::StopResponse, ("", false, false)),
            (ToolOutcome::ToolError("nope".into()), ("nope", true, true)),
            (ToolOutcome::ModelRetry("use a date".into()), ("use a date", true, true)),
            (ToolOutcome::Internal, (INTERNAL_ERROR_OUTPUT, true, true)),
            (ToolOutcome::Invalid, (INVALID_OUTPUT, true, true)),
            (
                ToolOutcome::Completed { text: "booked".into(), verbatim: false },
                ("booked", false, true),
            ),
            (ToolOutcome::Failed("no slot".into()), ("no slot", true, true)),
            (ToolOutcome::Rejected("not mine".into()), ("not mine", true, true)),
            (ask, ("waiting for the person's answer to: Which day?", false, false)),
            (ToolOutcome::Working(None), ("working", false, false)),
        ];
        for (outcome, (text, is_error, reply)) in cases {
            let got = shape(outcome, Mode::Text);
            assert_eq!(
                (got.text.as_str(), got.is_error, got.reply_required),
                (*text, *is_error, *reply),
                "{outcome:?}"
            );
        }
    }

    // Catches: a verbatim result's model text not saying it was delivered, or the delegate's note
    // dropped.
    #[test]
    fn a_verbatim_result_reads_as_delivered() {
        let v = ToolOutcome::Value {
            text: "Your code is 1234.".into(),
            reply_required: true,
            verbatim: true,
            more: Some("ask if they need more".into()),
        };
        assert_eq!(verbatim(&v), Some("Your code is 1234."));
        let want = "delivered to the person as written: Your code is 1234.\nask if they need more";
        assert_eq!(delivered(&v).as_deref(), Some(want));
        assert_eq!(delivered(&ToolOutcome::Nothing), None);
        assert!(!shape(&ToolOutcome::Nothing, Mode::Voice).reply_required);
    }
}
