//! Run event assertions (testing.md "Programmatic tests"): the cursor `result.expect` walks over
//! the events of one tested invocation, what each assertion matches and the failure it reports.
//! A failure names the expectation and lists the events with the one it concerns marked.

use lk_agents_abi::ChatItem;
use lk_agents_abi::agent::assertions::{AssertionFailure, EventExpectation};
use serde_json::Value;

use crate::approval::python_repr;

/// The observed events and the position `next_event` reads from.
#[derive(Debug, Clone)]
pub struct RunEvents {
    events: Vec<ChatItem>,
    position: usize,
}

impl lk_agents_abi::agent::assertions::RunEvents for RunEvents {
    fn new(events: Vec<ChatItem>) -> Self {
        RunEvents { events, position: 0 }
    }

    fn at(
        &self,
        index: i64,
        expected: Option<EventExpectation>,
    ) -> Result<usize, AssertionFailure> {
        let len = self.events.len();
        let resolved = if index < 0 { index + len as i64 } else { index };
        let Some(index) = usize::try_from(resolved).ok().filter(|i| *i < len) else {
            return Err(
                self.fail(&format!("nth({index}) out of range (total events: {len})"), None)
            );
        };
        match expected {
            Some(expected) => self.check(index, &expected),
            None => Ok(index),
        }
    }

    fn next_event(
        &mut self,
        expected: Option<EventExpectation>,
    ) -> Result<usize, AssertionFailure> {
        loop {
            let index = self.current()?;
            self.position += 1;
            match &expected {
                Some(expected) if !same_kind(&self.events[index], expected) => continue,
                Some(expected) => return self.check(index, expected),
                None => return Ok(index),
            }
        }
    }

    fn skip_next_event_if(&mut self, expected: EventExpectation) -> Option<usize> {
        let index = self.current().ok()?;
        let index = self.check(index, &expected).ok()?;
        self.position += 1;
        Some(index)
    }

    fn skip_next(&mut self, count: usize) -> Result<(), AssertionFailure> {
        let available = self.events.len() - self.position;
        if available < count {
            return Err(self.fail(
                &format!("Tried to skip {count} event(s), but only {available} were available."),
                None,
            ));
        }
        self.position += count;
        Ok(())
    }

    fn no_more_events(&self) -> Result<(), AssertionFailure> {
        match self.events.get(self.position) {
            None => Ok(()),
            Some(event) => Err(self
                .fail(&format!("Expected no more events, but found: {}", event_name(event)), None)),
        }
    }

    fn contains(
        &self,
        start: usize,
        stop: usize,
        expected: EventExpectation,
    ) -> Result<usize, AssertionFailure> {
        let stop = stop.min(self.events.len());
        (start..stop).find(|&i| self.check(i, &expected).is_ok()).ok_or_else(|| {
            let kind = kind_name(&expected);
            self.fail(
                &format!("No {kind} satisfying criteria found in range [{start}:{stop}]"),
                None,
            )
        })
    }

    fn failure(&self, message: String, index: Option<usize>) -> String {
        self.fail(&message, index).0
    }
}

impl RunEvents {
    /// `message`, then every event with the one at `index` (or the current one) marked.
    fn fail(&self, message: &str, index: Option<usize>) -> AssertionFailure {
        let marked = index.unwrap_or(self.position);
        let mut out = format!("{message}\nContext around failure:");
        for (i, event) in self.events.iter().enumerate() {
            let prefix = if i == marked { ">>>" } else { "   " };
            out.push_str(&format!("\n{prefix} [{i}] {}", describe(event)));
        }
        AssertionFailure(out)
    }

    fn current(&self) -> Result<usize, AssertionFailure> {
        if self.position < self.events.len() {
            Ok(self.position)
        } else {
            Err(self.fail("Expected another event, but none left.", None))
        }
    }

    /// `index` when its event meets `expected`, else the first mismatch.
    fn check(&self, index: usize, expected: &EventExpectation) -> Result<usize, AssertionFailure> {
        let event = &self.events[index];
        let fail = |message: String| Err(self.fail(&message, Some(index)));
        match (expected, event) {
            (EventExpectation::Message { role }, ChatItem::Message(message)) => {
                if let Some(role) = role
                    && role != message.role.as_str()
                {
                    return fail(format!(
                        "Expected role '{role}', got '{}'",
                        message.role.as_str()
                    ));
                }
            }
            (EventExpectation::FunctionCall { name, arguments }, ChatItem::FunctionCall(call)) => {
                if let Some(name) = name
                    && *name != call.name
                {
                    return fail(format!("Expected call name '{name}', got '{}'", call.name));
                }
                if let Some(expected) = arguments {
                    let Some(expected) = expected.as_object() else {
                        return fail(format!(
                            "Expected arguments must be an object, got {expected}"
                        ));
                    };
                    let actual = match serde_json::from_str::<Value>(&call.arguments) {
                        Ok(Value::Object(actual)) => actual,
                        _ => {
                            return fail(format!(
                                "The call's arguments are not a JSON object: {}",
                                python_repr(&call.arguments)
                            ));
                        }
                    };
                    for (key, value) in expected {
                        match actual.get(key) {
                            Some(found) if json_eq(found, value) => {}
                            found => {
                                let found = found.map_or("None".to_owned(), Value::to_string);
                                return fail(format!(
                                    "For key '{key}', expected {value}, got {found}"
                                ));
                            }
                        }
                    }
                }
            }
            (
                EventExpectation::FunctionCallOutput { output, is_error },
                ChatItem::FunctionCallOutput(out),
            ) => {
                if let Some(output) = output
                    && *output != out.output
                {
                    return fail(format!("Expected output '{output}', got '{}'", out.output));
                }
                if let Some(is_error) = is_error
                    && *is_error != out.is_error
                {
                    return fail(format!(
                        "Expected is_error={}, got {}",
                        py_bool(*is_error),
                        py_bool(out.is_error)
                    ));
                }
            }
            (expected, _) => return fail(format!("Expected {}", kind_name(expected))),
        }
        Ok(index)
    }
}

fn same_kind(item: &ChatItem, expected: &EventExpectation) -> bool {
    matches!(
        (expected, item),
        (EventExpectation::Message { .. }, ChatItem::Message(_))
            | (EventExpectation::FunctionCall { .. }, ChatItem::FunctionCall(_))
            | (EventExpectation::FunctionCallOutput { .. }, ChatItem::FunctionCallOutput(_))
    )
}

fn kind_name(expected: &EventExpectation) -> &'static str {
    match expected {
        EventExpectation::Message { .. } => "ChatMessageEvent",
        EventExpectation::FunctionCall { .. } => "FunctionCallEvent",
        EventExpectation::FunctionCallOutput { .. } => "FunctionCallOutputEvent",
    }
}

fn event_name(item: &ChatItem) -> &'static str {
    match item {
        ChatItem::Message(_) => "ChatMessageEvent",
        ChatItem::FunctionCall(_) => "FunctionCallEvent",
        ChatItem::FunctionCallOutput(_) => "FunctionCallOutputEvent",
        ChatItem::AgentHandoff(_) => "AgentHandoffEvent",
        ChatItem::AgentConfigUpdate(_) => "AgentConfigUpdate",
    }
}

/// One line of the failure context: the event's kind and the fields an assertion reads.
fn describe(event: &ChatItem) -> String {
    let name = event_name(event);
    match event {
        ChatItem::Message(m) => {
            let text = m.raw_text_content().unwrap_or_default();
            format!("{name}(role={}, content={})", python_repr(m.role.as_str()), python_repr(&text))
        }
        ChatItem::FunctionCall(c) => format!(
            "{name}(name={}, arguments={})",
            python_repr(&c.name),
            python_repr(&c.arguments)
        ),
        ChatItem::FunctionCallOutput(o) => format!(
            "{name}(name={}, output={}, is_error={})",
            python_repr(&o.name),
            python_repr(&o.output),
            py_bool(o.is_error)
        ),
        ChatItem::AgentHandoff(h) => format!(
            "{name}(old_agent={}, new_agent={})",
            h.old_agent_id.as_deref().map_or("None".to_owned(), python_repr),
            python_repr(&h.new_agent_id)
        ),
        ChatItem::AgentConfigUpdate(_) => name.to_owned(),
    }
}

fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

/// Equality as the application's values compare: `1` equals `1.0`, at any depth.
fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_i64(), y.as_i64()) {
            (Some(x), Some(y)) => x == y,
            _ => x.as_f64() == y.as_f64(),
        },
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| json_eq(x, y))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| json_eq(v, w)))
        }
        _ => a == b,
    }
}

#[cfg(test)]
mod tests;
