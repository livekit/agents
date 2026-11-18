//! What the frontend model reads about remote work. The peer's raw A2A task ID is named wherever
//! the model may target that task (continue, correct, cancel); it is never meant to be spoken.

use serde_json::Value;

/// The longest original request quoted beside a task ID.
const REQUEST_CHARS: usize = 120;

/// Said with every presentation that carries task IDs.
pub const IDS_ARE_NOT_SPOKEN: &str =
    "Task IDs are only for your tool calls: never say them to the person. ";

/// The fact that releases a call delegated to a local agent: it has no remote task to target.
pub const LOCAL_RUNNING: &str =
    "The request is running. Its progress, questions and result arrive separately.";

/// A one-line description of the work: its request text, else its business arguments.
pub fn request_summary(arguments: &Value) -> String {
    let text = match arguments.get("request") {
        Some(Value::String(request)) => request.clone(),
        _ => {
            let mut arguments = arguments.clone();
            if let Some(object) = arguments.as_object_mut() {
                object.remove("task_id");
            }
            arguments.to_string()
        }
    };
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match line.char_indices().nth(REQUEST_CHARS) {
        Some((end, _)) => format!("{}...", &line[..end]),
        None => line,
    }
}

/// The fact that releases a call's step while the remote works. `task_id` is `None` while the
/// peer has not assigned one: the work cannot be targeted until it does.
pub fn running(task_id: Option<&str>, request: &str) -> String {
    match task_id.filter(|id| !id.is_empty()) {
        Some(id) => format!(
            "Remote task {id} is running ({request:?}). Its progress, questions and result \
             arrive separately. To continue, correct or cancel this work, pass task_id \
             {id:?}; never say the ID to the person."
        ),
        None => format!(
            "The remote request ({request:?}) was sent, but the remote agent has not given it a \
             task ID yet, so it cannot be continued or cancelled by ID. Its progress, questions \
             and result arrive separately."
        ),
    }
}

/// Appended to a remote question's recorded output: the task that asked it.
pub fn asked_by(task_id: &str) -> String {
    format!(
        "\nAsked by remote task {task_id}: to answer, correct or cancel it, pass task_id \
         {task_id:?}; never say the ID to the person."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_running_fact_names_the_raw_task_id_and_its_request() {
        let fact = running(Some("AT_export-7"), "Export the March invoices");
        assert!(fact.contains("Remote task AT_export-7 is running"), "{fact}");
        assert!(fact.contains("\"Export the March invoices\""), "{fact}");
        assert!(fact.contains("task_id \"AT_export-7\""), "{fact}");
        assert!(fact.contains("never say the ID"), "{fact}");
    }

    #[test]
    fn before_the_peer_assigns_an_id_the_fact_says_so() {
        for missing in [None, Some("")] {
            let fact = running(missing, "Export");
            assert!(fact.contains("has not given it a task ID yet"), "{fact}");
            assert!(!fact.contains("task_id"), "{fact}");
        }
    }

    #[test]
    fn the_request_summary_is_one_bounded_line_without_routing() {
        assert_eq!(request_summary(&json!({"request": "Book\n a  flight"})), "Book a flight");
        assert_eq!(
            request_summary(&json!({"payer": "acme", "task_id": "t"})),
            r#"{"payer":"acme"}"#
        );
        let long = request_summary(&json!({"request": "é".repeat(300)}));
        assert_eq!(long.chars().count(), REQUEST_CHARS + 3);
    }
}
