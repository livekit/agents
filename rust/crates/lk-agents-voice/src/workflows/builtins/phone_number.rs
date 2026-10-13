use crate::workflows::builtins::instructions::render;
use crate::workflows::builtins::prompts::phone as p;
use crate::workflows::builtins::schema::{Param, nullary, tool};
use crate::workflows::{
    ConfirmSlot, Task, TaskCtx, TaskOptions, ToolError, ToolReply, args, confirmation_required,
    update_tool_flags,
};
use async_trait::async_trait;
use lk_agents_abi::ToolSpec;
use lk_agents_voice::generation::Instructions;
use serde_json::Value;

pub use lk_agents_abi::voice::workflows::GetPhoneNumberResult;

/// `GetPhoneNumberTask.__init__`. No `instructions` parameter: it formats `_BASE_INSTRUCTIONS`
/// directly, unlike `GetEmailTask` (kept, since unifying changes the rendered prompt).
#[derive(Default)]
pub struct GetPhoneNumberConfig {
    pub extra_instructions: String,
    pub require_confirmation: Option<bool>,
    pub require_explicit_ask: bool,
    pub options: TaskOptions,
}

pub struct GetPhoneNumberTask {
    instructions: Instructions,
    user_tools: Vec<ToolSpec>,
    require_confirmation: Option<bool>,
    require_explicit_ask: bool,
    current_phone_number: String,
    confirm: ConfirmSlot<String>,
}

impl GetPhoneNumberTask {
    pub fn new(cfg: GetPhoneNumberConfig) -> Self {
        GetPhoneNumberTask {
            instructions: build_instructions(&cfg),
            user_tools: cfg.options.tools,
            require_confirmation: cfg.require_confirmation,
            require_explicit_ask: cfg.require_explicit_ask,
            current_phone_number: String::new(),
            confirm: ConfirmSlot::default(),
        }
    }

    pub fn on_enter_prompt(&self) -> &'static str {
        p::ON_ENTER
    }

    fn update_tool(&self) -> ToolSpec {
        tool(
            p::UPDATE_TOOL,
            p::UPDATE_DESC,
            &[Param::string("phone_number", p::UPDATE_ARG_PHONE)],
            update_tool_flags(self.require_explicit_ask),
        )
    }

    async fn update_phone_number(
        &mut self,
        phone_number: &str,
        ctx: &mut TaskCtx<GetPhoneNumberResult>,
    ) -> ToolReply {
        let cleaned = clean_phone_number(phone_number);

        if !is_valid_phone_number(&cleaned) {
            return ToolReply::Error(ToolError::plain(render(
                p::INVALID,
                &[("phone_number", phone_number)],
            )));
        }

        self.current_phone_number = cleaned.clone();

        if !confirmation_required(self.require_confirmation, ctx.modality()) {
            ctx.complete(GetPhoneNumberResult { phone_number: self.current_phone_number.clone() });
            return ToolReply::Silent; // no need to continue the conversation
        }

        let revision = ctx.tools().upsert_last(nullary(p::CONFIRM_TOOL, p::CONFIRM_DESC));
        self.confirm.install(revision, cleaned.clone());

        ToolReply::text(render(p::UPDATED, &[("cleaned", &cleaned)]))
    }

    fn confirm_phone_number(&mut self, ctx: &mut TaskCtx<GetPhoneNumberResult>) -> ToolReply {
        let Some(number) = self.confirm.captured_for(ctx.turn_tools_revision()).cloned() else {
            return ToolReply::unknown_tool(p::CONFIRM_TOOL);
        };
        if number != self.current_phone_number {
            // stale closure: update_phone_number ran again after this confirm tool was installed
            return ToolReply::text(p::STALE);
        }
        ctx.complete(GetPhoneNumberResult { phone_number: number });
        ToolReply::Silent
    }
}

#[async_trait]
impl Task for GetPhoneNumberTask {
    type Output = GetPhoneNumberResult;

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
        ctx: &mut TaskCtx<GetPhoneNumberResult>,
    ) -> ToolReply {
        match name {
            p::UPDATE_TOOL => match args::string(&args_json, name, "phone_number") {
                Ok(number) => self.update_phone_number(&number, ctx).await,
                Err(e) => ToolReply::Error(e),
            },
            p::CONFIRM_TOOL => self.confirm_phone_number(ctx),
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

fn build_instructions(cfg: &GetPhoneNumberConfig) -> Instructions {
    let confirmation = p::CONFIRMATION_INSTRUCTION;
    Instructions {
        common: String::new(),
        audio: Some(render(
            p::BASE_INSTRUCTIONS,
            &[
                ("modality_specific", p::AUDIO_SPECIFIC),
                (
                    "confirmation_instructions",
                    if cfg.require_confirmation == Some(false) { "" } else { confirmation },
                ),
                ("extra_instructions", &cfg.extra_instructions),
            ],
        )),
        text: Some(render(
            p::BASE_INSTRUCTIONS,
            &[
                ("modality_specific", p::TEXT_SPECIFIC),
                (
                    "confirmation_instructions",
                    if cfg.require_confirmation == Some(true) { confirmation } else { "" },
                ),
                ("extra_instructions", &cfg.extra_instructions),
            ],
        )),
    }
}

/// `re.sub(r"[\s\-().]+", "", phone_number.strip())`.
pub fn clean_phone_number(phone_number: &str) -> String {
    phone_number
        .trim()
        .chars()
        .filter(|c| !(c.is_whitespace() || matches!(c, '-' | '(' | ')' | '.')))
        .collect()
}

/// `PHONE_REGEX = r"^\+?[1-9]\d{6,14}$"`, ASCII digits only (Python's `\d` admits any Unicode
/// digit, which SIP can't take).
pub fn is_valid_phone_number(cleaned: &str) -> bool {
    let digits = cleaned.strip_prefix('+').unwrap_or(cleaned);
    let mut chars = digits.chars();
    if !chars.next().is_some_and(|c| ('1'..='9').contains(&c)) {
        return false;
    }
    // `\d{6,14}` after the leading non-zero digit: 7 to 15 digits in total.
    let rest: Vec<char> = chars.collect();
    (6..=14).contains(&rest.len()) && rest.iter().all(char::is_ascii_digit)
}
