use crate::workflows::ToolError;
use crate::workflows::builtins::instructions::render;
use crate::workflows::builtins::prompts::task_group as p;
use lk_agents_abi::voice::workflows::{
    RawFunctionDescription, TaskGroupComplete, TaskGroupFail, TaskGroupRun, TaskGroupStepWire,
    TaskGroupSummarize, TaskGroupSummaryFlags,
};
use lk_agents_abi::{ToolFlags, ToolSpec};
use serde_json::{Value, json};
use std::collections::{HashSet, VecDeque};

/// `_FactoryInfo` minus the factory, which cannot cross.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSpec {
    pub id: String,
    /// When the LLM should regress to this task; a prompt artifact per call site.
    pub description: String,
}

impl TaskSpec {
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        TaskSpec { id: id.into(), description: description.into() }
    }
}

/// `_OutOfScopeError`: a message-less `ToolError` subclass caught before the generic one; a
/// control-flow signal, so its own type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfScope {
    pub target_task_ids: Vec<String>,
}

/// The *value* is absent on purpose: the host holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskGroupOutcome {
    /// `res = await self._current_task` returned.
    Value,
    /// `except _OutOfScopeError`.
    Regress(OutOfScope),
    /// `except Exception`.
    Error(ToolError),
}

/// An `outcome` a host named that is none of the three.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown outcome {0:?}; expected \"value\", \"regress\" or \"error\"")]
pub struct UnknownOutcome(pub String);

impl TaskGroupOutcome {
    /// The outcome as a host reports it (absent data is empty).
    pub fn from_wire(
        outcome: &str,
        target_task_ids: Option<Vec<String>>,
        error: Option<String>,
    ) -> Result<Self, UnknownOutcome> {
        Ok(match outcome {
            "value" => TaskGroupOutcome::Value,
            "regress" => TaskGroupOutcome::Regress(OutOfScope {
                target_task_ids: target_task_ids.unwrap_or_default(),
            }),
            "error" => TaskGroupOutcome::Error(ToolError::plain(error.unwrap_or_default())),
            other => return Err(UnknownOutcome(other.to_owned())),
        })
    }
}

/// What ended up in `task_results[task_id]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    /// The task's own value, which the host is holding.
    Value,
    /// The exception stored as a result (`return_exceptions=True`).
    Error,
}

impl RecordKind {
    /// The name that travels beside the id in `TaskGroupComplete::recorded`.
    pub fn as_str(self) -> &'static str {
        match self {
            RecordKind::Value => "value",
            RecordKind::Error => "error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskGroupStep {
    /// Construct the task and await it; `out_of_scope` is appended to its tools first, if any.
    Run { task_id: String, out_of_scope: Option<ToolSpec> },
    /// The stack drained and `summarize_chat_ctx` is set; report via
    /// [`summarized`](TaskGroupSchedule::summarized).
    Summarize,
    /// `self.complete(TaskGroupResult(task_results=...))`.
    Complete(TaskGroupResult),
    /// `self.complete(e)` -- the `return_exceptions=False` arm, and the summariser's failure.
    Fail(ToolError),
}

/// `TaskGroupResult` as the `task_results` keys in insertion order; the host owns the values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TaskGroupResult {
    pub task_results: Vec<(String, RecordKind)>,
}

impl From<TaskGroupStep> for TaskGroupStepWire {
    fn from(step: TaskGroupStep) -> Self {
        match step {
            TaskGroupStep::Run { task_id, out_of_scope } => TaskGroupStepWire::Run(TaskGroupRun {
                task_id,
                out_of_scope: out_of_scope.as_ref().map(RawFunctionDescription::from),
            }),
            TaskGroupStep::Summarize => TaskGroupStepWire::Summarize(TaskGroupSummarize {}),
            TaskGroupStep::Complete(result) => TaskGroupStepWire::Complete(TaskGroupComplete::from(&result)),
            TaskGroupStep::Fail(e) => TaskGroupStepWire::Fail(TaskGroupFail { error: e.message }),
        }
    }
}

impl From<&TaskGroupResult> for TaskGroupComplete {
    fn from(r: &TaskGroupResult) -> Self {
        TaskGroupComplete {
            recorded: r
                .task_results
                .iter()
                .map(|(id, kind)| (id.clone(), kind.as_str().to_owned()))
                .collect(),
        }
    }
}

/// The five `copy()` flags `_summarize` uses, all `false` (the opposite of the defaults) so the
/// summary keeps the tool calls: "keep every item to allow summarization to be more action-aware".
#[lk_agents_abi::host(export(pid = false, name = "task_group_summary_flags"))]
pub const SUMMARY_COPY_FLAGS: TaskGroupSummaryFlags = TaskGroupSummaryFlags {
    exclude_instructions: false,
    exclude_handoff: false,
    exclude_config_update: false,
    exclude_empty_message: false,
    exclude_function_call: false,
};

/// `_build_out_of_scope_tool`'s description, in registration order.
pub fn out_of_scope_description(specs: &[TaskSpec], ids: &[String]) -> String {
    // Built as a string so registration order doesn't depend on serde_json's `preserve_order`.
    let mut repr = String::from("{");
    for (i, spec) in specs.iter().filter(|s| ids.contains(&s.id)).enumerate() {
        if i > 0 {
            repr.push_str(", ");
        }
        repr.push_str(&Value::String(spec.id.clone()).to_string());
        repr.push_str(": ");
        repr.push_str(&Value::String(spec.description.clone()).to_string());
    }
    repr.push('}');
    render(p::OUT_OF_SCOPE_DESC, &[("task_repr", &repr)])
}

/// The whole `out_of_scope` tool, `None` before any visit. Built by hand because the `enum` is a
/// runtime list. Registration order, where the reference's set order varies (and defeats caching).
pub fn out_of_scope_tool(specs: &[TaskSpec], ids: &[String]) -> Option<ToolSpec> {
    if ids.is_empty() {
        return None;
    }
    let enum_values: Vec<Value> =
        specs.iter().filter(|s| ids.contains(&s.id)).map(|s| Value::String(s.id.clone())).collect();
    Some(ToolSpec {
        name: p::OUT_OF_SCOPE_TOOL.to_owned(),
        description: out_of_scope_description(specs, ids),
        json_schema: json!({
            "type": "object",
            "properties": {
                "task_ids": {
                    "type": "array",
                    "description": p::OUT_OF_SCOPE_ARG,
                    "items": { "type": "string", "enum": enum_values },
                }
            },
            "required": ["task_ids"],
            "additionalProperties": false,
        }),
        // `IGNORE_ON_ENTER`: or the model regresses out before the task asked once.
        flags: ToolFlags { ignore_on_enter: true },
    })
}

/// The scheduler. Drive it from the host loop; it never awaits and holds no task objects.
#[derive(Debug, Clone)]
pub struct TaskGroupSchedule {
    specs: Vec<TaskSpec>,
    stack: VecDeque<String>,
    visited: HashSet<String>,
    recorded: Vec<(String, RecordKind)>,
    current: Option<String>,
    summarize_chat_ctx: bool,
    return_exceptions: bool,
}

/// `beta/workflows/task_group.py`'s scheduler as the host drives it: the loop body (factories,
/// awaits, callbacks, summarisation) stays in Python; this decides the stack order, the visited set
/// (written before the await), each outcome's arm, the `out_of_scope` tool and regress validation.
/// Result values never enter Rust. Each step comes back as its `kind`-tagged dict.
#[lk_agents_abi::host(object(pid = false, name = "TaskSchedule", cell = "mutex"))]
impl TaskGroupSchedule {
    /// `TaskGroup.__init__`. Both flags carry the reference's defaults.
    #[host(new(kw_only, defaults(summarize_chat_ctx = true, return_exceptions = false)))]
    pub fn new(summarize_chat_ctx: bool, return_exceptions: bool) -> Self {
        TaskGroupSchedule {
            specs: Vec::new(),
            stack: VecDeque::new(),
            visited: HashSet::new(),
            recorded: Vec::new(),
            current: None,
            summarize_chat_ctx,
            return_exceptions,
        }
    }

    /// `TaskGroup.add`; re-adding replaces the description and keeps the position.
    pub fn add(&mut self, spec: TaskSpec) -> &mut Self {
        match self.specs.iter_mut().find(|s| s.id == spec.id) {
            Some(existing) => *existing = spec,
            None => self.specs.push(spec),
        }
        self
    }

    /// Whether `task_id` was entered (written before the await, so a failed task can be regressed to).
    #[host(method)]
    pub fn visited(&self, task_id: &str) -> bool {
        self.visited.contains(task_id)
    }

    /// `TaskGroup.add`'s bookkeeping. Re-adding an id **replaces** its description and keeps its
    /// position, which is what assigning into an `OrderedDict` does.
    #[host(method(name = "add"))]
    pub fn host_add(&mut self, id: String, description: String) {
        self.add(TaskSpec::new(id, description));
    }

    /// `on_enter`'s loop entry.
    #[host(method(name = "start"))]
    pub fn host_start(&mut self) -> TaskGroupStepWire {
        self.start().into()
    }

    /// The running task finished. `outcome` is `"value"`, `"regress"` or `"error"`.
    #[host(method(name = "advance", kw_only = "target_task_ids"))]
    pub fn host_advance(
        &mut self,
        outcome: &str,
        target_task_ids: Option<Vec<String>>,
        error: Option<String>,
    ) -> Result<TaskGroupStepWire, UnknownOutcome> {
        let outcome = TaskGroupOutcome::from_wire(outcome, target_task_ids, error)?;
        Ok(self.advance(outcome).into())
    }

    /// The summariser finished. `error` is `except Exception as e: self.complete(e)`.
    #[host(method(name = "summarized"))]
    pub fn host_summarized(&mut self, error: Option<String>) -> TaskGroupStepWire {
        self.summarized(error.map(ToolError::plain)).into()
    }

    /// `out_of_scope`'s body: every id must be registered and visited, or a regress would reorder
    /// the group or `KeyError`. Returns the error message, or `None`.
    #[host(method(name = "validate_regress"))]
    pub fn host_validate_regress(&self, task_ids: Vec<String>) -> Option<String> {
        self.validate_regress(&task_ids).err().map(|e| e.message)
    }

    /// `on_enter`'s loop entry: `task_stack = list(self._registered_factories.keys())`.
    pub fn start(&mut self) -> TaskGroupStep {
        self.stack = self.specs.iter().map(|s| s.id.clone()).collect();
        self.next_step()
    }

    /// The running task finished. Panics if none is running (a host bug).
    pub fn advance(&mut self, outcome: TaskGroupOutcome) -> TaskGroupStep {
        let task_id = self.current.take().expect("advance() with no task running");
        match outcome {
            TaskGroupOutcome::Value => {
                self.recorded.push((task_id, RecordKind::Value));
            }
            TaskGroupOutcome::Regress(OutOfScope { target_task_ids }) => {
                // Targets end up first in the order the user asked, then current, then the rest.
                self.stack.push_front(task_id);
                for id in target_task_ids.into_iter().rev() {
                    self.stack.push_front(id);
                }
            }
            TaskGroupOutcome::Error(e) => {
                if !self.return_exceptions {
                    return TaskGroupStep::Fail(e);
                }
                // `continue` without re-pushing: the failed task is *skipped*, not retried.
                self.recorded.push((task_id, RecordKind::Error));
            }
        }
        self.next_step()
    }

    /// The summariser finished. `Some(e)` is `except Exception as e: self.complete(e)`.
    pub fn summarized(&mut self, error: Option<ToolError>) -> TaskGroupStep {
        match error {
            Some(e) => TaskGroupStep::Fail(e),
            None => {
                TaskGroupStep::Complete(TaskGroupResult { task_results: self.recorded.clone() })
            }
        }
    }

    fn next_step(&mut self) -> TaskGroupStep {
        let Some(task_id) = self.stack.pop_front() else {
            return if self.summarize_chat_ctx {
                TaskGroupStep::Summarize
            } else {
                TaskGroupStep::Complete(TaskGroupResult { task_results: self.recorded.clone() })
            };
        };

        // Visited before this task is marked, minus itself: the first task gets no tool.
        let ids: Vec<String> = self
            .specs
            .iter()
            .map(|s| s.id.clone())
            .filter(|id| self.visited.contains(id) && *id != task_id)
            .collect();
        let out_of_scope = out_of_scope_tool(&self.specs, &ids);

        self.visited.insert(task_id.clone());
        self.current = Some(task_id.clone());
        TaskGroupStep::Run { task_id, out_of_scope }
    }

    /// `out_of_scope`'s body: every id must be registered and visited (an unvisited one would
    /// reorder the group).
    pub fn validate_regress(&self, task_ids: &[String]) -> Result<(), ToolError> {
        for id in task_ids {
            if !self.specs.iter().any(|s| &s.id == id) || !self.visited.contains(id) {
                return Err(ToolError::plain(render(p::INVALID_TASK_ID, &[("task_id", id)])));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn three() -> TaskGroupSchedule {
        let mut s = TaskGroupSchedule::new(false, false);
        s.add(TaskSpec::new("a", "first"))
            .add(TaskSpec::new("b", "second"))
            .add(TaskSpec::new("c", "third"));
        s
    }

    fn running(step: &TaskGroupStep) -> (&str, bool) {
        let TaskGroupStep::Run { task_id, out_of_scope } = step else { panic!("expected Run") };
        (task_id, out_of_scope.is_some())
    }

    fn regress(ids: &[&str]) -> TaskGroupOutcome {
        TaskGroupOutcome::Regress(OutOfScope {
            target_task_ids: ids.iter().map(|s| (*s).to_owned()).collect(),
        })
    }

    // Catches: inserting a regression's targets forwards (the user's order reversed).
    #[test]
    fn a_regression_preserves_the_order_the_user_asked_for() {
        let mut s = three();
        // The first task has nothing to regress to, so no tool at all.
        assert_eq!(running(&s.start()), ("a", false));
        assert_eq!(running(&s.advance(TaskGroupOutcome::Value)), ("b", true)); // a done -> b
        assert_eq!(running(&s.advance(TaskGroupOutcome::Value)), ("c", true)); // b done -> c
        assert_eq!(running(&s.advance(regress(&["a", "b"]))).0, "a");
        assert_eq!(running(&s.advance(TaskGroupOutcome::Value)).0, "b");
    }

    // Catches: dropping the `and` from the regress validation.
    #[test]
    fn regressing_to_an_unvisited_task_is_rejected() {
        let mut s = three();
        s.start();
        assert!(s.validate_regress(&["a".to_owned()]).is_ok(), "a is visited");
        assert!(s.validate_regress(&["b".to_owned()]).is_err(), "b has not been reached");
        assert!(s.validate_regress(&["zz".to_owned()]).is_err(), "and zz does not exist");
    }
}
