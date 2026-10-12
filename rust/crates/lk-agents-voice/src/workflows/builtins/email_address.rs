use crate::workflows::builtins::instructions::{
    InstructionsArg, Resolved, TemplateValue, WorkflowInstructions, render,
};
use crate::workflows::builtins::prompts::email as p;
use crate::workflows::builtins::schema::{Param, nullary, tool};
use crate::workflows::{
    ConfirmSlot, Task, TaskCtx, TaskOptions, ToolError, ToolReply, Warning, args,
    confirmation_required, update_tool_flags,
};
use async_trait::async_trait;
use lk_agents_abi::ToolSpec;
use lk_agents_voice::generation::Instructions;
use serde_json::Value;

pub use lk_agents_abi::voice::workflows::GetEmailResult;

/// `GetEmailTask.__init__`'s keyword-only parameters; tri-state ones stay tri-state.
#[derive(Default)]
pub struct GetEmailConfig {
    /// `instructions: WorkflowInstructions | Instructions | str`. `None` is `NOT_GIVEN`.
    pub instructions: Option<InstructionsArg>,
    pub require_confirmation: Option<bool>,
    pub require_explicit_ask: bool,
    /// Deprecated in the reference, and ignored (with a warning) when `instructions` is given.
    pub extra_instructions: String,
    pub options: TaskOptions,
}

pub struct GetEmailTask {
    instructions: Instructions,
    warnings: Vec<Warning>,
    user_tools: Vec<ToolSpec>,
    require_confirmation: Option<bool>,
    require_explicit_ask: bool,
    current_email: String,
    confirm: ConfirmSlot<String>,
}

impl GetEmailTask {
    pub fn new(cfg: GetEmailConfig) -> Self {
        let mut warnings = Vec::new();
        let resolved = build_instructions(&cfg, &mut warnings);
        GetEmailTask {
            instructions: resolved.instructions,
            warnings,
            user_tools: cfg.options.tools,
            require_confirmation: cfg.require_confirmation,
            require_explicit_ask: cfg.require_explicit_ask,
            current_email: String::new(),
            confirm: ConfirmSlot::default(),
        }
    }

    /// `on_enter`'s `generate_reply(instructions=...)` argument.
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
            &[Param::string("email", p::UPDATE_ARG_EMAIL)],
            update_tool_flags(self.require_explicit_ask),
        )
    }

    async fn update_email(&mut self, email: &str, ctx: &mut TaskCtx<GetEmailResult>) -> ToolReply {
        let email = email.trim().to_owned();

        if !is_valid_email(&email) {
            return ToolReply::Error(ToolError::plain(render(p::INVALID, &[("email", &email)])));
        }

        self.current_email = email.clone();
        let separated: String = email.chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");

        if !confirmation_required(self.require_confirmation, ctx.modality()) {
            ctx.complete(GetEmailResult { email_address: self.current_email.clone() });
            return ToolReply::Silent; // no need to continue the conversation
        }

        // confirm tool is only injected after update_email_address is called
        let revision = ctx.tools().upsert_last(nullary(p::CONFIRM_TOOL, p::CONFIRM_DESC));
        self.confirm.install(revision, email.clone());

        ToolReply::text(render(p::UPDATED, &[("email", &email), ("separated", &separated)]))
    }

    fn confirm_email(&mut self, ctx: &mut TaskCtx<GetEmailResult>) -> ToolReply {
        let Some(email) = self.confirm.captured_for(ctx.turn_tools_revision()).cloned() else {
            return ToolReply::unknown_tool(p::CONFIRM_TOOL);
        };
        if email != self.current_email {
            // stale closure: update_email_address ran again after this confirm tool was installed
            return ToolReply::text(p::STALE);
        }
        ctx.complete(GetEmailResult { email_address: email });
        ToolReply::Silent
    }
}

#[async_trait]
impl Task for GetEmailTask {
    type Output = GetEmailResult;

    fn instructions(&self) -> Instructions {
        self.instructions.clone()
    }

    fn tools(&self) -> Vec<ToolSpec> {
        let mut tools = self.user_tools.clone();
        tools.push(self.update_tool());
        tools.push(decline_tool());
        tools
    }

    async fn call(
        &mut self,
        name: &str,
        args_json: Value,
        ctx: &mut TaskCtx<GetEmailResult>,
    ) -> ToolReply {
        match name {
            p::UPDATE_TOOL => match args::string(&args_json, name, "email") {
                Ok(email) => self.update_email(&email, ctx).await,
                Err(e) => ToolReply::Error(e),
            },
            p::CONFIRM_TOOL => self.confirm_email(ctx),
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

fn decline_tool() -> ToolSpec {
    tool(
        p::DECLINE_TOOL,
        p::DECLINE_DESC,
        &[Param::string("reason", p::DECLINE_ARG_REASON)],
        update_tool_flags(true), // `@function_tool(flags=ToolFlag.IGNORE_ON_ENTER)` -- unconditional
    )
}

fn build_instructions(cfg: &GetEmailConfig, warnings: &mut Vec<Warning>) -> Resolved {
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
            // A plain `Instructions` or `str` is used verbatim.
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

/// `EMAIL_REGEX`, hand-written:
/// `^[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$`.
/// Stricter than RFC 5322 on purpose: the input is a speech transcript.
pub fn is_valid_email(email: &str) -> bool {
    let Some((local, domain)) = email.split_once('@') else { return false };
    if domain.contains('@') {
        return false;
    }

    let mut chars = local.chars();
    if !chars.next().is_some_and(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '%' | '+' | '-')) {
        return false;
    }

    let labels: Vec<&str> = domain.split('.').collect();
    // `(?:label\.)+` requires at least one label before the TLD.
    if labels.len() < 2 {
        return false;
    }
    let (tld, rest) = labels.split_last().expect("len >= 2");
    if tld.len() < 2 || !tld.chars().all(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    rest.iter().all(|label| is_domain_label(label))
}

/// A domain label: alphanumeric at both ends, hyphens only in between.
fn is_domain_label(label: &str) -> bool {
    let bytes = label.as_bytes();
    match bytes {
        [] => false,
        [only] => only.is_ascii_alphanumeric(),
        [first, middle @ .., last] => {
            first.is_ascii_alphanumeric()
                && last.is_ascii_alphanumeric()
                && middle.iter().all(|b| b.is_ascii_alphanumeric() || *b == b'-')
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_instructions(arg: InstructionsArg) -> GetEmailTask {
        GetEmailTask::new(GetEmailConfig { instructions: Some(arg), ..Default::default() })
    }

    // Catches: a fixed `Instructions` being run through the template anyway.
    #[test]
    fn a_fixed_instruction_is_used_verbatim() {
        let task = with_instructions(InstructionsArg::Fixed(Instructions::plain("only this")));
        assert_eq!(task.instructions(), Instructions::plain("only this"));
    }

    // Catches: a per-modality persona flattened to a plain string.
    #[test]
    fn a_modal_persona_reaches_both_variants_separately() {
        let task = with_instructions(InstructionsArg::Workflow(WorkflowInstructions {
            persona: Some(TemplateValue::modal("spoken", "typed")),
            extra: None,
        }));
        let Instructions { audio: Some(audio), text: Some(text), .. } = task.instructions() else {
            panic!("a modal persona renders both variants");
        };
        assert!(audio.contains("spoken"));
        assert!(text.contains("typed"));
        assert!(!audio.contains("typed"));
    }
}
