use crate::workflows::builtins::instructions::{
    InstructionsArg, Resolved, TemplateValue, WorkflowInstructions, render,
};
use crate::workflows::builtins::prompts::address as p;
use crate::workflows::builtins::schema::{Param, nullary, tool};
use crate::workflows::{
    ConfirmSlot, Task, TaskCtx, TaskOptions, ToolError, ToolReply, Warning, args,
    confirmation_required, update_tool_flags,
};
use async_trait::async_trait;
use lk_agents_abi::ToolSpec;
use lk_agents_voice::generation::Instructions;
use serde_json::Value;

pub use lk_agents_abi::voice::workflows::GetAddressResult;

#[derive(Default)]
pub struct GetAddressConfig {
    pub instructions: Option<InstructionsArg>,
    pub require_confirmation: Option<bool>,
    pub require_explicit_ask: bool,
    /// Deprecated in the reference.
    pub extra_instructions: String,
    pub options: TaskOptions,
}

pub struct GetAddressTask {
    instructions: Instructions,
    warnings: Vec<Warning>,
    user_tools: Vec<ToolSpec>,
    require_confirmation: Option<bool>,
    require_explicit_ask: bool,
    current_address: String,
    confirm: ConfirmSlot<String>,
}

impl GetAddressTask {
    pub fn new(cfg: GetAddressConfig) -> Self {
        let mut warnings = Vec::new();
        let resolved = build_instructions(&cfg, &mut warnings);
        GetAddressTask {
            instructions: resolved.instructions,
            warnings,
            user_tools: cfg.options.tools,
            require_confirmation: cfg.require_confirmation,
            require_explicit_ask: cfg.require_explicit_ask,
            current_address: String::new(),
            confirm: ConfirmSlot::default(),
        }
    }

    pub fn on_enter_prompt(&self) -> &'static str {
        p::ON_ENTER
    }

    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    fn update_tool(&self) -> ToolSpec {
        tool(
            p::UPDATE_TOOL,
            p::UPDATE_DESC,
            &[
                Param::string("street_address", p::UPDATE_ARG_STREET),
                Param::string("unit_number", p::UPDATE_ARG_UNIT),
                Param::string("locality", p::UPDATE_ARG_LOCALITY),
                Param::string("country", p::UPDATE_ARG_COUNTRY),
            ],
            update_tool_flags(self.require_explicit_ask),
        )
    }

    async fn update_address(
        &mut self,
        street_address: &str,
        unit_number: &str,
        locality: &str,
        country: &str,
        ctx: &mut TaskCtx<GetAddressResult>,
    ) -> ToolReply {
        // Whitespace-only, not empty: a model sending `' '` must not leave a lone space.
        let fields: Vec<&str> = if unit_number.trim().is_empty() {
            vec![street_address, locality, country]
        } else {
            vec![street_address, unit_number, locality, country]
        };
        let address = fields.join(" ");
        self.current_address = address.clone();

        if !confirmation_required(self.require_confirmation, ctx.modality()) {
            ctx.complete(GetAddressResult { address: self.current_address.clone() });
            return ToolReply::Silent;
        }

        // confirm tool is only injected after update_address is called
        let revision = ctx.tools().upsert_last(nullary(p::CONFIRM_TOOL, p::CONFIRM_DESC));
        self.confirm.install(revision, address.clone());

        ToolReply::text(render(
            p::UPDATED,
            &[("address", &address), ("fields", &python_list_repr(&fields))],
        ))
    }

    fn confirm_address(&mut self, ctx: &mut TaskCtx<GetAddressResult>) -> ToolReply {
        let Some(address) = self.confirm.captured_for(ctx.turn_tools_revision()).cloned() else {
            return ToolReply::unknown_tool(p::CONFIRM_TOOL);
        };
        if address != self.current_address {
            // stale closure: update_address ran again after this confirm tool was installed
            return ToolReply::text(p::STALE);
        }
        ctx.complete(GetAddressResult { address });
        ToolReply::Silent
    }
}

#[async_trait]
impl Task for GetAddressTask {
    type Output = GetAddressResult;

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
        ctx: &mut TaskCtx<GetAddressResult>,
    ) -> ToolReply {
        match name {
            p::UPDATE_TOOL => {
                let field = |key| args::string(&args_json, name, key);
                let street = match field("street_address") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                let unit = match field("unit_number") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                let locality = match field("locality") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                let country = match field("country") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                self.update_address(&street, &unit, &locality, &country, ctx).await
            }
            p::CONFIRM_TOOL => self.confirm_address(ctx),
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

fn build_instructions(cfg: &GetAddressConfig, warnings: &mut Vec<Warning>) -> Resolved {
    let workflow = match &cfg.instructions {
        None => WorkflowInstructions {
            persona: Some(TemplateValue::plain(p::PERSONA)),
            extra: Some(TemplateValue::plain(&cfg.extra_instructions)),
        },
        Some(InstructionsArg::Workflow(w)) => {
            if !cfg.extra_instructions.is_empty() {
                warnings.push(Warning::ExtraInstructionsIgnored);
            }
            w.clone()
        }
        Some(InstructionsArg::Fixed(i)) => {
            if !cfg.extra_instructions.is_empty() {
                warnings.push(Warning::ExtraInstructionsIgnored);
            }
            return Resolved { instructions: i.clone(), missing: Vec::new() };
        }
    };

    workflow.resolve(
        p::INSTRUCTIONS_TEMPLATE,
        p::PERSONA,
        &[
            ("_modality_specific", TemplateValue::modal(p::AUDIO_SPECIFIC, p::TEXT_SPECIFIC)),
            // confirmation is enabled by default for audio, disabled by default for text
            (
                "_confirmation",
                TemplateValue::modal(
                    if cfg.require_confirmation == Some(false) {
                        ""
                    } else {
                        p::CONFIRMATION_INSTRUCTION
                    },
                    if cfg.require_confirmation == Some(true) {
                        p::CONFIRMATION_INSTRUCTION
                    } else {
                        ""
                    },
                ),
            ),
        ],
    )
}

/// The field list as Python's list repr, which is what the reference's prompt interpolates.
fn python_list_repr(fields: &[&str]) -> String {
    let inner: Vec<String> = fields.iter().map(|f| python_str_repr(f)).collect();
    format!("[{}]", inner.join(", "))
}

/// Python's `repr` for a `str`.
pub fn python_str_repr(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\");
    if s.contains('\'') && !s.contains('"') {
        format!("\"{escaped}\"")
    } else {
        format!("'{}'", escaped.replace('\'', "\\'"))
    }
}
