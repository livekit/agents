use crate::workflows::builtins::ConfigError;
use crate::workflows::builtins::address::python_str_repr;
use crate::workflows::builtins::instructions::render;
use crate::workflows::builtins::prompts::name as p;
use crate::workflows::builtins::schema::{Param, nullary, tool};
use crate::workflows::{
    ConfirmSlot, Task, TaskCtx, TaskOptions, ToolError, ToolReply, args, confirmation_required,
    update_tool_flags,
};
use async_trait::async_trait;
use lk_agents_abi::ToolSpec;
use lk_agents_voice::generation::Instructions;
use serde_json::Value;

/// A field is `None` unless configured to be collected.
pub use lk_agents_abi::voice::workflows::GetNameResult;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GetNameConfigParts {
    pub first_name: bool,
    pub last_name: bool,
    pub middle_name: bool,
}

impl Default for GetNameConfigParts {
    /// `first_name: bool = True, last_name: bool = False, middle_name: bool = False`.
    fn default() -> Self {
        GetNameConfigParts { first_name: true, last_name: false, middle_name: false }
    }
}

#[derive(Default)]
pub struct GetNameConfig {
    pub parts: GetNameConfigParts,
    /// `NOT_GIVEN` builds `"{first_name} {middle_name} {last_name}"` from `parts`.
    pub name_format: Option<String>,
    pub verify_spelling: bool,
    pub extra_instructions: String,
    pub require_confirmation: Option<bool>,
    pub require_explicit_ask: bool,
    pub options: TaskOptions,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct NameState {
    first: String,
    middle: String,
    last: String,
}

pub struct GetNameTask {
    instructions: Instructions,
    name_format: String,
    parts: GetNameConfigParts,
    verify_spelling: bool,
    user_tools: Vec<ToolSpec>,
    require_confirmation: Option<bool>,
    require_explicit_ask: bool,
    state: NameState,
    confirm: ConfirmSlot<NameState>,
}

impl GetNameTask {
    pub fn new(cfg: GetNameConfig) -> Result<Self, ConfigError> {
        if !(cfg.parts.first_name || cfg.parts.middle_name || cfg.parts.last_name) {
            return Err(ConfigError::NoNamePartSelected);
        }

        let name_format =
            cfg.name_format.clone().unwrap_or_else(|| default_name_format(&cfg.parts));

        Ok(GetNameTask {
            instructions: build_instructions(&cfg, &name_format),
            name_format,
            parts: cfg.parts,
            verify_spelling: cfg.verify_spelling,
            user_tools: cfg.options.tools,
            require_confirmation: cfg.require_confirmation,
            require_explicit_ask: cfg.require_explicit_ask,
            state: NameState::default(),
            confirm: ConfirmSlot::default(),
        })
    }

    pub fn on_enter_prompt(&self) -> String {
        render(p::ON_ENTER, &[("name_format", &self.name_format)])
    }

    fn result_from(&self, state: &NameState) -> GetNameResult {
        GetNameResult {
            first_name: self.parts.first_name.then(|| state.first.clone()),
            middle_name: self.parts.middle_name.then(|| state.middle.clone()),
            last_name: self.parts.last_name.then(|| state.last.clone()),
        }
    }

    fn update_tool(&self) -> ToolSpec {
        tool(
            p::UPDATE_TOOL,
            p::UPDATE_DESC,
            &[
                Param::opt_string("first_name", p::UPDATE_ARG_FIRST),
                Param::opt_string("middle_name", p::UPDATE_ARG_MIDDLE),
                Param::opt_string("last_name", p::UPDATE_ARG_LAST),
            ],
            update_tool_flags(self.require_explicit_ask),
        )
    }

    async fn update_name(
        &mut self,
        first: Option<String>,
        middle: Option<String>,
        last: Option<String>,
        ctx: &mut TaskCtx<GetNameResult>,
    ) -> ToolReply {
        let first = clean_name_arg(first.as_deref());
        let middle = clean_name_arg(middle.as_deref());
        let last = clean_name_arg(last.as_deref());

        let mut errors: Vec<String> = Vec::new();
        if self.parts.first_name && !is_present(&first) {
            errors.push(p::ERR_MISSING_FIRST.to_owned());
        }
        if self.parts.middle_name && !is_present(&middle) {
            errors.push(p::ERR_MISSING_MIDDLE.to_owned());
        }
        if self.parts.last_name && !is_present(&last) {
            errors.push(p::ERR_MISSING_LAST.to_owned());
        }

        // A real name contains letters: reject digit- or punctuation-only values.
        for (label, value) in [("first", &first), ("middle", &middle), ("last", &last)] {
            if let Some(v) = value
                && !v.trim().is_empty()
                && !v.chars().any(char::is_alphabetic)
            {
                errors.push(render(
                    p::ERR_NO_LETTERS,
                    &[("label", label), ("value", &python_str_repr(v))],
                ));
            }
        }

        if !errors.is_empty() {
            return ToolReply::Error(ToolError::plain(render(
                p::INCOMPLETE,
                &[("errors", &errors.join("; "))],
            )));
        }

        self.state = NameState {
            first: first.map(|v| v.trim().to_owned()).unwrap_or_default(),
            middle: middle.map(|v| v.trim().to_owned()).unwrap_or_default(),
            last: last.map(|v| v.trim().to_owned()).unwrap_or_default(),
        };

        // `safe_render` leaves an unknown placeholder unfilled where the reference raises `KeyError`.
        let full_name = render(
            &self.name_format,
            &[
                ("first_name", &self.state.first),
                ("middle_name", &self.state.middle),
                ("last_name", &self.state.last),
            ],
        )
        .trim()
        .to_owned();

        if !confirmation_required(self.require_confirmation, ctx.modality()) {
            let result = self.result_from(&self.state);
            ctx.complete(result);
            return ToolReply::Silent;
        }

        let revision = ctx.tools().upsert_last(nullary(p::CONFIRM_TOOL, p::CONFIRM_DESC));
        self.confirm.install(revision, self.state.clone());

        if self.verify_spelling {
            return ToolReply::text(render(p::UPDATED_SPELL, &[("full_name", &full_name)]));
        }
        ToolReply::text(render(p::UPDATED, &[("full_name", &full_name)]))
    }

    fn confirm_name(&mut self, ctx: &mut TaskCtx<GetNameResult>) -> ToolReply {
        let Some(captured) = self.confirm.captured_for(ctx.turn_tools_revision()).cloned() else {
            return ToolReply::unknown_tool(p::CONFIRM_TOOL);
        };
        if captured != self.state {
            // stale closure: update_name ran again after this confirm tool was installed
            return ToolReply::text(p::STALE);
        }
        let result = self.result_from(&self.state);
        ctx.complete(result);
        ToolReply::Silent
    }
}

#[async_trait]
impl Task for GetNameTask {
    type Output = GetNameResult;

    fn instructions(&self) -> Instructions {
        self.instructions.clone()
    }

    fn tools(&self) -> Vec<ToolSpec> {
        let mut tools = self.user_tools.clone();
        tools.push(self.update_tool());
        tools.push(tool(
            p::DECLINE_TOOL,
            p::DECLINE_DESC,
            &[Param::string("reason", p::DECLINE_ARG_REASON)],
            update_tool_flags(true),
        ));
        tools
    }

    async fn call(
        &mut self,
        name: &str,
        args_json: Value,
        ctx: &mut TaskCtx<GetNameResult>,
    ) -> ToolReply {
        match name {
            p::UPDATE_TOOL => {
                let first = args::opt_string(&args_json, "first_name");
                let middle = args::opt_string(&args_json, "middle_name");
                let last = args::opt_string(&args_json, "last_name");
                self.update_name(first, middle, last, ctx).await
            }
            p::CONFIRM_TOOL => self.confirm_name(ctx),
            p::DECLINE_TOOL => match args::string(&args_json, name, "reason") {
                Ok(reason) => {
                    ctx.fail(ToolError::plain(render(p::DECLINED, &[("reason", &reason)])));
                    ToolReply::Silent
                }
                Err(e) => ToolReply::Error(e),
            },
            other => ToolReply::unknown_tool(other),
        }
    }
}

fn is_present(v: &Option<String>) -> bool {
    v.as_ref().is_some_and(|s| !s.trim().is_empty())
}

pub fn default_name_format(parts: &GetNameConfigParts) -> String {
    let mut out: Vec<&str> = Vec::new();
    if parts.first_name {
        out.push("{first_name}");
    }
    if parts.middle_name {
        out.push("{middle_name}");
    }
    if parts.last_name {
        out.push("{last_name}");
    }
    out.join(" ")
}

/// `_clean_name_arg`: placeholder strings like "null" become `None`, and any number of quote
/// characters are stripped from both ends (`str.strip("'\"")`).
pub fn clean_name_arg(value: Option<&str>) -> Option<String> {
    let value = value?;
    let value = value.trim().trim_matches(|c| c == '\'' || c == '"');
    if value.is_empty() {
        return None;
    }
    const SENTINELS: [&str; 6] = ["null", "none", "nil", "n/a", "unknown", "unspecified"];
    if SENTINELS.contains(&value.to_lowercase().as_str()) {
        return None;
    }
    Some(value.to_owned())
}

fn build_instructions(cfg: &GetNameConfig, name_format: &str) -> Instructions {
    let spelling = if cfg.verify_spelling { p::SPELLING_INSTRUCTIONS } else { "" };
    let confirmation = p::CONFIRMATION_INSTRUCTION;
    let common = |modality_specific: &str, confirmation_instructions: &str| {
        render(
            p::BASE_INSTRUCTIONS,
            &[
                ("name_format", name_format),
                ("modality_specific", modality_specific),
                ("spelling_instructions", spelling),
                ("confirmation_instructions", confirmation_instructions),
                ("extra_instructions", &cfg.extra_instructions),
            ],
        )
    };
    Instructions {
        common: String::new(),
        audio: Some(common(
            p::AUDIO_SPECIFIC,
            if cfg.require_confirmation == Some(false) { "" } else { confirmation },
        )),
        text: Some(common(
            p::TEXT_SPECIFIC,
            if cfg.require_confirmation == Some(true) { confirmation } else { "" },
        )),
    }
}
