pub use crate::workflows::builtins::address::{GetAddressConfig, GetAddressTask};
pub use crate::workflows::builtins::credit_card::{
    CARD_CAPTURE_DECLINED, CARD_COLLECTION_RESTART, CARD_ISSUERS, GetCardNumberConfig,
    GetCardNumberResult, GetCardNumberTask, GetCreditCardResult, GetExpirationDateConfig,
    GetExpirationDateResult, GetExpirationDateTask, GetSecurityCodeConfig, GetSecurityCodeResult,
    GetSecurityCodeTask, assemble_credit_card_result, card_capture_declined,
    card_collection_restart, luhn_valid,
};
pub use crate::workflows::builtins::dob::{GetDOBConfig, GetDOBResult, GetDOBTask};
pub use crate::workflows::builtins::dtmf::{
    DtmfAction, DtmfCollector, DtmfEvent, DtmfReply, GetDtmfConfig, GetDtmfResult, GetDtmfTask,
    dtmf_event_to_code, dtmf_result, format_dtmf,
};
pub use crate::workflows::builtins::email_address::{GetEmailConfig, GetEmailTask};
pub use crate::workflows::builtins::instructions::{
    BothInstructionForms, InstructionsArg, Resolved, TemplateValue, WorkflowInstructions, render,
    render_checked, resolve_template,
};
pub use crate::workflows::builtins::name::{
    GetNameConfig, GetNameConfigParts, GetNameResult, GetNameTask, clean_name_arg,
};
pub use crate::workflows::builtins::phone_number::{GetPhoneNumberConfig, GetPhoneNumberTask};
pub use crate::workflows::builtins::task_group::{
    OutOfScope, TaskGroupOutcome, TaskGroupResult, TaskGroupSchedule, TaskGroupStep, TaskSpec,
    out_of_scope_description, out_of_scope_tool,
};
pub use lk_agents_abi::Modality;
pub use lk_agents_voice::generation::Instructions;

pub mod address;
pub mod credit_card;
pub mod dob;
pub mod dtmf;
pub mod email_address;
pub mod instructions;
pub mod name;
pub mod phone_number;
pub mod prompts;
pub mod schema;
pub mod task_group;
pub mod warm_transfer;

/// Config that the reference rejects in `__init__` with a `ValueError`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConfigError {
    #[error("At least one of first_name, middle_name, or last_name must be True")]
    NoNamePartSelected,
    #[error("num_digits must be greater than 0")]
    NonPositiveDigits,
}

// `voice/workflow_tasks.py` is nine thin `AgentTask` subclasses over [`HostWorkflow`]: it owns the
// session effects and the `isinstance` dispatch; every decision is here.

use crate::workflows::{NativeTask, Outcome, TaskDriver, TaskOptions, ToolError, ToolReply, Warning};
use lk_agents_abi::voice::workflows::RawFunctionDescription;
use lk_agents_abi::{CoreObject, ToolSpec};

/// `llm.chat_context.Instructions` as it crosses back: `(common, audio, text)`. A tuple, so there
/// is no second `Instructions` class for users to hold by accident.
pub type InstructionsTriple = (String, Option<String>, Option<String>);

fn triple_of(i: &Instructions) -> InstructionsTriple {
    (i.common.clone(), i.audio.clone(), i.text.clone())
}

/// The workflows' `instructions:` parameter. The host picks the form (an `isinstance` test);
/// [`InstructionsArg::from_host`] decides what it means and refuses both at once.
type Sections = (Option<Instructions>, Option<Instructions>);

/// `date.today()` is local, so the UTC offset crosses: UTC would misjudge a future birth date or
/// this month's card expiry near midnight.
fn today_at(utc_offset_secs: i32) -> lk_agents_abi::utils::dates::TodaySource {
    let clock: std::sync::Arc<dyn lk_agents_abi::Clock> = std::sync::Arc::new(lk_agents_abi::RealClock::new());
    lk_agents_abi::utils::dates::TodaySource::with_utc_offset(clock, utc_offset_secs)
}

fn options(tools: Option<Vec<ToolSpec>>) -> TaskOptions {
    TaskOptions { tools: tools.unwrap_or_default(), ..TaskOptions::default() }
}

/// The [`Warning`]s as the names the host logs them under; the host owns the wording.
fn warning_names(warnings: &[Warning]) -> Vec<String> {
    warnings.iter().map(|w| w.as_str().to_owned()).collect()
}

/// A `workflow` construction refusal (`ValueError`): both instruction forms, or a config the
/// reference rejects in `__init__`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkflowRefused {
    #[error(transparent)]
    Instructions(#[from] BothInstructionForms),
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("{0:?} is not a DTMF key")]
    NotDtmf(String),
}

/// One of the nine shipped workflows, driven from Python.
pub struct HostWorkflow {
    /// A `tokio` mutex: `call` holds it across an `await`.
    driver: tokio::sync::Mutex<Box<dyn TaskDriver<CoreObject>>>,
    instructions: Instructions,
    initial_tools: Vec<ToolSpec>,
    on_enter: String,
    warnings: Vec<String>,
}

impl HostWorkflow {
    /// `task` driven with its value erased to the host object `wrap` makes of it.
    fn wrap<T>(task: T, wrap: fn(T::Output) -> CoreObject, on_enter: &str, warnings: &[Warning]) -> HostWorkflow
    where
        T: crate::workflows::Task + 'static,
        T::Output: Clone,
    {
        let native = NativeTask::new(task, Box::new(move |v| Ok(wrap(v))));
        HostWorkflow {
            instructions: native.instructions(),
            initial_tools: native.declared_tools(),
            on_enter: on_enter.to_owned(),
            warnings: warning_names(warnings),
            driver: tokio::sync::Mutex::new(Box::new(native)),
        }
    }
}

fn value<T: std::any::Any + Send + Sync>(v: T) -> CoreObject {
    CoreObject(std::sync::Arc::new(v))
}

#[lk_agents_abi::host(object(name = "Workflow", cell = "arc"))]
impl HostWorkflow {
    /// `beta/workflows/email_address.py`'s `GetEmailTask`.
    #[host(method(
        kw_only,
        defaults(
            extra_instructions = String::new(),
            require_explicit_ask = false
        )
    ))]
    pub fn email(
        instructions: Option<Instructions>,
        sections: Option<Sections>,
        extra_instructions: String,
        require_confirmation: Option<bool>,
        require_explicit_ask: bool,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<Self, WorkflowRefused> {
        let task = GetEmailTask::new(GetEmailConfig {
            instructions: InstructionsArg::from_host(instructions, sections)?,
            require_confirmation,
            require_explicit_ask,
            extra_instructions,
            options: options(tools),
        });
        let (prompt, warnings) = (task.on_enter_prompt().to_owned(), task.warnings().to_vec());
        Ok(HostWorkflow::wrap(task, value, &prompt, &warnings))
    }

    /// `beta/workflows/address.py`'s `GetAddressTask`.
    #[host(method(
        kw_only,
        defaults(
            extra_instructions = String::new(),
            require_explicit_ask = false
        )
    ))]
    pub fn address(
        instructions: Option<Instructions>,
        sections: Option<Sections>,
        extra_instructions: String,
        require_confirmation: Option<bool>,
        require_explicit_ask: bool,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<Self, WorkflowRefused> {
        let task = GetAddressTask::new(GetAddressConfig {
            instructions: InstructionsArg::from_host(instructions, sections)?,
            require_confirmation,
            require_explicit_ask,
            extra_instructions,
            options: options(tools),
        });
        let (prompt, warnings) = (task.on_enter_prompt().to_owned(), task.warnings().to_vec());
        Ok(HostWorkflow::wrap(task, value, &prompt, &warnings))
    }

    /// `beta/workflows/phone_number.py`'s `GetPhoneNumberTask`. No `instructions=`: the reference
    /// takes `extra_instructions` only.
    #[host(method(
        kw_only,
        defaults(
            extra_instructions = String::new(),
            require_explicit_ask = false
        )
    ))]
    pub fn phone_number(
        extra_instructions: String,
        require_confirmation: Option<bool>,
        require_explicit_ask: bool,
        tools: Option<Vec<ToolSpec>>,
    ) -> Self {
        let task = GetPhoneNumberTask::new(GetPhoneNumberConfig {
            extra_instructions,
            require_confirmation,
            require_explicit_ask,
            options: options(tools),
        });
        let prompt = task.on_enter_prompt().to_owned();
        HostWorkflow::wrap(task, value, &prompt, &[])
    }

    /// `beta/workflows/name.py`'s `GetNameTask`. The flags say which parts to collect; all three
    /// false raises `ValueError`.
    #[host(method(
        kw_only,
        defaults(
            first_name = true,
            last_name = false,
            middle_name = false,
            verify_spelling = false,
            extra_instructions = String::new(),
            require_explicit_ask = false
        )
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn name(
        first_name: bool,
        last_name: bool,
        middle_name: bool,
        name_format: Option<String>,
        verify_spelling: bool,
        extra_instructions: String,
        require_confirmation: Option<bool>,
        require_explicit_ask: bool,
        tools: Option<Vec<ToolSpec>>,
    ) -> Result<Self, WorkflowRefused> {
        let task = GetNameTask::new(GetNameConfig {
            parts: GetNameConfigParts { first_name, last_name, middle_name },
            name_format,
            verify_spelling,
            extra_instructions,
            require_confirmation,
            require_explicit_ask,
            options: options(tools),
        })?;
        // Already a `String`: the prompt interpolates `{name_format}`, which is per-instance.
        let prompt = task.on_enter_prompt();
        Ok(HostWorkflow::wrap(task, value, &prompt, &[]))
    }

    /// `beta/workflows/dob.py`'s `GetDOBTask`.
    #[host(method(
        kw_only,
        defaults(
            extra_instructions = String::new(),
            include_time = false,
            require_explicit_ask = false,
            utc_offset_secs = 0
        )
    ))]
    pub fn dob(
        extra_instructions: String,
        include_time: bool,
        require_confirmation: Option<bool>,
        require_explicit_ask: bool,
        utc_offset_secs: i32,
        tools: Option<Vec<ToolSpec>>,
    ) -> Self {
        let task = GetDOBTask::new(GetDOBConfig {
            extra_instructions,
            include_time,
            require_confirmation,
            require_explicit_ask,
            options: options(tools),
            today: today_at(utc_offset_secs),
        });
        let prompt = task.on_enter_prompt().to_owned();
        HostWorkflow::wrap(task, value, &prompt, &[])
    }

    /// `beta/workflows/credit_card.py`'s `GetCardNumberTask`.
    #[host(method(
        kw_only,
        defaults(require_explicit_ask = false, extra_instructions = String::new())
    ))]
    pub fn card_number(require_confirmation: Option<bool>, require_explicit_ask: bool, extra_instructions: String) -> Self {
        let task = GetCardNumberTask::new(GetCardNumberConfig {
            require_confirmation,
            require_explicit_ask,
            extra_instructions,
            ..Default::default()
        });
        let prompt = task.on_enter_prompt().to_owned();
        HostWorkflow::wrap(task, value, &prompt, &[])
    }

    /// `beta/workflows/credit_card.py`'s `GetSecurityCodeTask`.
    #[host(method(
        kw_only,
        defaults(require_explicit_ask = false, extra_instructions = String::new())
    ))]
    pub fn security_code(require_confirmation: Option<bool>, require_explicit_ask: bool, extra_instructions: String) -> Self {
        let task = GetSecurityCodeTask::new(GetSecurityCodeConfig {
            require_confirmation,
            require_explicit_ask,
            extra_instructions,
            ..Default::default()
        });
        let prompt = task.on_enter_prompt().to_owned();
        HostWorkflow::wrap(task, value, &prompt, &[])
    }

    /// `beta/workflows/credit_card.py`'s `GetExpirationDateTask`.
    #[host(method(
        kw_only,
        defaults(
            require_explicit_ask = false,
            extra_instructions = String::new(),
            utc_offset_secs = 0
        )
    ))]
    pub fn expiration_date(
        require_confirmation: Option<bool>,
        require_explicit_ask: bool,
        extra_instructions: String,
        utc_offset_secs: i32,
    ) -> Self {
        let task = GetExpirationDateTask::new(GetExpirationDateConfig {
            require_confirmation,
            require_explicit_ask,
            extra_instructions,
            today: today_at(utc_offset_secs),
        });
        let prompt = task.on_enter_prompt().to_owned();
        HostWorkflow::wrap(task, value, &prompt, &[])
    }

    /// `beta/workflows/dtmf_inputs.py`'s `GetDtmfTask` -- its *tool* half. The event half is the
    /// driver's (`AgentSession.dtmf_start`).
    #[host(method(
        kw_only = "num_digits",
        defaults(
            ask_for_confirmation = false,
            dtmf_input_timeout = 4.0,
            dtmf_stop_event = "#".to_owned()
        )
    ))]
    pub fn dtmf(
        num_digits: usize,
        ask_for_confirmation: bool,
        dtmf_input_timeout: f64,
        dtmf_stop_event: String,
        extra_instructions: Option<String>,
    ) -> Result<Self, WorkflowRefused> {
        let stop = DtmfEvent::parse(&dtmf_stop_event).ok_or(WorkflowRefused::NotDtmf(dtmf_stop_event))?;
        let task = GetDtmfTask::new(
            GetDtmfConfig {
                num_digits,
                ask_for_confirmation,
                dtmf_input_timeout: std::time::Duration::from_secs_f64(dtmf_input_timeout),
                dtmf_stop_event: stop,
                extra_instructions,
            },
            TaskOptions::default(),
        )?;
        // `GetDtmfTask.on_enter` calls `generate_reply(tool_choice="none")` with no instructions
        // of its own -- the entering turn must speak, never record.
        Ok(HostWorkflow::wrap(task, |v| value(dtmf::DtmfResult(v)), "", &[]))
    }

    /// `Agent.instructions`, as `(common, audio, text)`.
    #[host(getter(name = "instructions"))]
    pub fn host_instructions(&self) -> InstructionsTriple {
        triple_of(&self.instructions)
    }

    /// The `instructions=` argument of `on_enter`'s `generate_reply`. `""` when the workflow's
    /// `on_enter` passes none, which is `GetDtmfTask` and only it.
    #[host(getter)]
    pub fn on_enter_prompt(&self) -> String {
        self.on_enter.clone()
    }

    /// Names of what the reference `logger.warning`s about in `__init__`. The core has no logger,
    /// so these come back as data and `voice/workflow_tasks.py` writes the sentences.
    #[host(getter)]
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    /// `super().__init__(tools=[...])`: the initial, config-derived list. Later changes arrive on
    /// `WorkflowReply.tools`.
    #[host(method)]
    pub fn tools(&self) -> Vec<RawFunctionDescription> {
        self.initial_tools.iter().map(RawFunctionDescription::from).collect()
    }

    /// Run one tool call. `turn` is the speech handle's id: the core keeps one ctx per turn, which
    /// the stale-confirmation check depends on.
    #[host(method(kw_only = "turn", defaults(modality = "audio".to_owned())))]
    pub async fn call(
        &self,
        name: String,
        arguments: serde_json::Value,
        turn: String,
        modality: String,
    ) -> Result<WorkflowReply, lk_agents_abi::UnknownLiteral> {
        let modality: Modality = modality.parse()?;
        let outcome = self.driver.lock().await.call(&turn, modality, &name, arguments).await;
        Ok(WorkflowReply::of(outcome))
    }
}

/// What one tool call produced. Independent fields: one call can reply, install a tool and complete
/// the task at once.
#[lk_agents_abi::class]
pub struct WorkflowReply {
    reply: Option<String>,
    error: Option<ToolError>,
    completed: bool,
    result: Option<CoreObject>,
    failure: Option<ToolError>,
    tools: Option<Vec<ToolSpec>>,
}

impl WorkflowReply {
    fn of(outcome: Outcome<CoreObject>) -> Self {
        let (reply, error) = match outcome.reply {
            ToolReply::Silent => (None, None),
            ToolReply::Text(t) => (Some(t), None),
            ToolReply::Error(e) => (None, Some(e)),
        };
        let (completed, result, failure) = match outcome.settled {
            None => (false, None, None),
            Some(Ok(value)) => (true, Some(value), None),
            Some(Err(e)) => (true, None, Some(e)),
        };
        WorkflowReply { reply, error, completed, result, failure, tools: outcome.tools }
    }
}

/// `(message, class_name, reason)` of an error, each `None` when there is no error at all.
fn error_parts(e: &Option<ToolError>) -> (Option<String>, Option<String>, Option<String>) {
    match e.clone().map(ToolError::into_parts) {
        None => (None, None, None),
        Some((message, class, reason)) => (Some(message), class, reason),
    }
}

#[lk_agents_abi::host(object(pid = false))]
impl WorkflowReply {
    /// The tool body's `return`. `None` is `return None`, deliberately not `""`, which would push an
    /// empty tool message.
    #[host(getter)]
    pub fn reply(&self) -> Option<String> {
        self.reply.clone()
    }

    /// The tool body's `raise ToolError(..)`. The host raises it; the model reads it and retries.
    #[host(getter)]
    pub fn error(&self) -> Option<String> {
        error_parts(&self.error).0
    }

    /// `type(exc).__name__` when the error is a `ToolError` subclass, so the host raises the class
    /// its `except` clauses discriminate on.
    #[host(getter)]
    pub fn error_class(&self) -> Option<String> {
        error_parts(&self.error).1
    }

    /// The subclass' `reason` property.
    #[host(getter)]
    pub fn error_reason(&self) -> Option<String> {
        error_parts(&self.error).2
    }

    /// Whether this call ended the task. Separate from `result`, because a task can complete with
    /// a *failure* and then `result` is legitimately `None`.
    #[host(getter)]
    pub fn completed(&self) -> bool {
        self.completed
    }

    /// `self.complete(value)`'s argument on the success arm.
    #[host(getter)]
    pub fn result(&self) -> Option<CoreObject> {
        self.result.clone()
    }

    /// `self.complete(ToolError(..))`'s argument on the failure arm.
    #[host(getter)]
    pub fn failure(&self) -> Option<String> {
        error_parts(&self.failure).0
    }

    #[host(getter)]
    pub fn failure_class(&self) -> Option<String> {
        error_parts(&self.failure).1
    }

    #[host(getter)]
    pub fn failure_reason(&self) -> Option<String> {
        error_parts(&self.failure).2
    }

    /// The whole new tool list when `await self.update_tools(..)` ran during the call, else `None`.
    /// A list, not a diff: the order reaches the model.
    #[host(getter)]
    pub fn tools(&self) -> Option<Vec<RawFunctionDescription>> {
        self.tools.as_ref().map(|specs| specs.iter().map(RawFunctionDescription::from).collect())
    }

    #[host(method(name = "__repr__"))]
    pub fn host_repr(&self) -> String {
        format!(
            "WorkflowReply(reply={:?}, error={:?}, completed={}, tools_changed={})",
            self.reply,
            error_parts(&self.error).0,
            self.completed,
            self.tools.is_some()
        )
    }
}
