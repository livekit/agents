use lk_agents_abi::agent::FunctionToolInfo;
use lk_agents_abi::agent::run::RunError;
use lk_agents_abi::llm::tool_executor::{
    CompletedReturn, EXACT_OUTPUT_PREFIX, INTERNAL_ERROR_OUTPUT, INVALID_OUTPUT,
    InputRequiredReturn, ToolOutcome, ToolReport, ToolResultReturn, ToolReturn, ToolText,
    UNAVAILABLE_OUTPUT, WAITING_PREFIX,
};
use std::collections::BTreeMap;

/// The host's classification of what a tool's body returned or raised.
pub struct ToolReports;

impl lk_agents_abi::llm::tool_executor::ToolReports for ToolReports {
    fn tool_report(value: ToolReturn) -> ToolReport {
        let outcome = match value {
            ToolReturn::Report(report) => return report,
            ToolReturn::ToolError(message) => ToolOutcome::ToolError(message),
            ToolReturn::ModelRetry(message) => ToolOutcome::ModelRetry(message),
            ToolReturn::StopResponse => ToolOutcome::StopResponse,
            ToolReturn::Raised => ToolOutcome::Internal,
            ToolReturn::Completed(CompletedReturn { value, verbatim }) => match value {
                ToolText::None => ToolOutcome::Completed { text: "completed".into(), verbatim },
                ToolText::Text(text) => ToolOutcome::Completed { text, verbatim },
                ToolText::Invalid => ToolOutcome::Invalid,
            },
            ToolReturn::Failed(error) => ToolOutcome::Failed(error),
            ToolReturn::Rejected(reason) => ToolOutcome::Rejected(reason),
            ToolReturn::InputRequired(InputRequiredReturn { question, verbatim }) => {
                ToolOutcome::InputRequired { question, verbatim }
            }
            ToolReturn::Working(message) => ToolOutcome::Working(message),
            ToolReturn::Nothing => ToolOutcome::Nothing,
            ToolReturn::Result(ToolResultReturn { output, reply_required, verbatim, more }) => {
                let text = match output {
                    ToolText::None => String::new(),
                    ToolText::Text(text) => text,
                    ToolText::Invalid => return ToolOutcome::Invalid.into(),
                };
                ToolOutcome::Value { text, reply_required, verbatim, more }
            }
            ToolReturn::Value(ToolText::Text(text)) => {
                ToolOutcome::Value { text, reply_required: true, verbatim: false, more: None }
            }
            ToolReturn::Value(ToolText::None | ToolText::Invalid) => ToolOutcome::Invalid,
        };
        outcome.into()
    }
}

/// A callback failure still settles its admitted body. Cancellation stays typed; local
/// pressure tells the model it may retry; other host diagnostics never become tool output.
pub fn host_failure(error: &lk_agents_abi::errors::HostError) -> ToolOutcome {
    match error.kind {
        lk_agents_abi::errors::HostErrorKind::Cancelled => ToolOutcome::Cancelled,
        lk_agents_abi::errors::HostErrorKind::Unavailable => {
            ToolOutcome::ToolError(UNAVAILABLE_OUTPUT.into())
        }
        _ => ToolOutcome::Internal,
    }
}

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
    /// `agent.run()` and TextSession: `None` is a value the model reads and continues from.
    Text,
    /// A call: a silent tool asks for no follow-up reply (`reply_required = fnc_out is not None`).
    Voice,
}

/// `make_tool_output` and the transitions as one table. A verbatim result's delivery belongs
/// to its coordinator; [`exact_output_note`] formats model context without confirming output.
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
        ToolOutcome::Update { message, reply_required, .. } => {
            shape(message, false, *reply_required)
        }
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

/// Format the supplied exact text and optional model note. This is not a playout receipt and
/// cannot establish perceived history; only the owning output's real delivery evidence can.
pub fn exact_output_note(outcome: &ToolOutcome) -> Option<String> {
    let text = verbatim(outcome)?;
    Some(match outcome {
        ToolOutcome::Value { more: Some(more), .. } => {
            format!("{EXACT_OUTPUT_PREFIX}{text}\n{more}")
        }
        _ => format!("{EXACT_OUTPUT_PREFIX}{text}"),
    })
}

/// The invocation's tool retry budgets: consecutive retried steps per tool, reset by any call of
/// that tool that did not ask for a retry. Shared by AgentRun and a call's reply.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolRetries(BTreeMap<String, u32>);

impl ToolRetries {
    /// One settled step's calls as `(tool, retried)`: the first tool, by name, whose retries
    /// went over its limit: its own `retries=` among `tools`, else the run's `budget`
    /// (`Retries.tools`).
    pub fn step<'a>(
        &mut self,
        calls: impl IntoIterator<Item = (&'a str, bool)>,
        tools: &[FunctionToolInfo],
        budget: u32,
    ) -> Option<RunError> {
        let limit = |name: &str| {
            let tool = tools.iter().find(|t| t.spec.name == name);
            tool.and_then(|t| t.retries).unwrap_or(budget)
        };
        let mut retried: BTreeMap<&str, bool> = BTreeMap::new();
        for (tool, retry) in calls {
            *retried.entry(tool).or_insert(true) &= retry;
        }
        let mut exceeded = None;
        for (tool, retried) in retried {
            let count = self.0.entry(tool.to_owned()).or_default();
            *count = if retried { *count + 1 } else { 0 };
            let limit = limit(tool);
            if *count > limit && exceeded.is_none() {
                exceeded =
                    Some(RunError::ToolRetriesExceeded { tool: tool.to_owned(), retries: limit });
            }
        }
        exceeded
    }
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
        let ask = ToolOutcome::InputRequired { question: "Which day?".into(), verbatim: false };
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

    // Catches: model context claiming delivery before output evidence, dropping business text,
    // or losing the optional follow-up note.
    #[test]
    fn exact_output_context_preserves_the_result_without_claiming_delivery() {
        let v = ToolOutcome::Value {
            text: "Your code is 1234.".into(),
            reply_required: true,
            verbatim: true,
            more: Some("ask if they need more".into()),
        };
        assert_eq!(verbatim(&v), Some("Your code is 1234."));
        let want = "exact text supplied for separate framework delivery: Your code is 1234.\nask if they need more";
        assert_eq!(exact_output_note(&v).as_deref(), Some(want));
        assert_eq!(exact_output_note(&ToolOutcome::Nothing), None);
        assert!(!shape(&ToolOutcome::Nothing, Mode::Voice).reply_required);
    }

    #[test]
    fn exact_output_formatting_never_changes_business_success_reply_policy_or_followup() {
        for reply_required in [false, true] {
            for more in [None, Some("Ask whether they need directions.".to_owned())] {
                let result = ToolOutcome::Value {
                    text: "Reservation R-204 is confirmed.".into(),
                    verbatim: true,
                    reply_required,
                    more: more.clone(),
                };
                let before = result.clone();
                let note = exact_output_note(&result).unwrap();
                assert!(note.starts_with("exact text supplied for separate framework delivery: Reservation R-204 is confirmed."));
                assert_eq!(note.split_once('\n').map(|(_, text)| text), more.as_deref());
                for mode in [Mode::Text, Mode::Voice] {
                    let shape = shape(&result, mode);
                    assert!(
                        !shape.is_error,
                        "delivery uncertainty is not a failed business effect"
                    );
                    assert_eq!(shape.text, "Reservation R-204 is confirmed.");
                    assert_eq!(shape.reply_required, reply_required);
                }
                assert_eq!(result, before);
            }
        }
        let result = ToolOutcome::Completed {
            text: "Reservation R-204 is confirmed.".into(),
            verbatim: true,
        };
        assert_eq!(
            exact_output_note(&result).as_deref(),
            Some(
                "exact text supplied for separate framework delivery: Reservation R-204 is confirmed."
            )
        );
        assert_eq!(
            exact_output_note(&ToolOutcome::Value {
                text: "ordinary result".into(),
                verbatim: false,
                reply_required: true,
                more: None
            }),
            None
        );
    }
}
