use lk_agents_abi::utils::dates::TodaySource;
use crate::workflows::builtins::instructions::render;
use crate::workflows::builtins::name::GetNameResult;
use crate::workflows::builtins::prompts::credit_card as p;
use crate::workflows::builtins::schema::{Param, tool};
use crate::workflows::{
    ConfirmSlot, Task, TaskCtx, ToolError, ToolReply, args, confirmation_required,
    update_tool_flags,
};
use async_trait::async_trait;
use lk_agents_abi::ToolSpec;
use lk_agents_voice::generation::Instructions;
use serde_json::Value;

/// `CARD_ISSUERS_LOOKUP`, keyed on the first digit; anything else is `"Other"` (user-visible,
/// not widened).
pub const CARD_ISSUERS: [(char, &str); 4] =
    [('3', "American Express"), ('4', "Visa"), ('5', "Mastercard"), ('6', "Discover")];

pub fn issuer_for(card_number: &str) -> &'static str {
    let Some(first) = card_number.chars().next() else { return "Other" };
    CARD_ISSUERS.iter().find(|(d, _)| *d == first).map(|(_, n)| *n).unwrap_or("Other")
}

/// The four collected values (the ABI's). `GetExpirationDateResult.date` is mapped by
/// [`assemble_credit_card_result`].
pub use lk_agents_abi::voice::workflows::{
    GetCardNumberResult, GetCreditCardResult, GetExpirationDateResult, GetSecurityCodeResult,
};

/// The pure half of `GetCreditCardTask.on_enter`'s success arm. A `None` name part is written as
/// the literal `None`, as Python's f-string does.
pub fn assemble_credit_card_result(
    cardholder: &GetNameResult,
    card_number: &GetCardNumberResult,
    security_code: &GetSecurityCodeResult,
    expiration_date: &GetExpirationDateResult,
) -> GetCreditCardResult {
    let first = cardholder.first_name.as_deref().unwrap_or("None");
    let last = cardholder.last_name.as_deref().unwrap_or("None");
    GetCreditCardResult {
        cardholder_name: format!("{first} {last}"),
        issuer: card_number.issuer.clone(),
        card_number: card_number.card_number.clone(),
        security_code: security_code.security_code.clone(),
        expiration_date: expiration_date.date.clone(),
    }
}

/// The Luhn checksum, ASCII digits only (the reference's `isdigit()` admits `²`, which then crashes `int()`).
#[lk_agents_abi::host(export(pid = false))]
pub fn luhn_valid(card_number: &str) -> bool {
    if card_number.is_empty() || !card_number.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let total: u32 = card_number
        .chars()
        .rev()
        .enumerate()
        .map(|(index, c)| {
            let digit = c.to_digit(10).expect("checked ascii above");
            if index % 2 == 1 {
                let doubled = digit * 2;
                if doubled > 9 { doubled - 9 } else { doubled }
            } else {
                digit
            }
        })
        .sum();
    total.is_multiple_of(10)
}

/// Shared by the three sub-tasks; none takes `tools=`, so the built-ins are the whole list.
#[derive(Default)]
pub struct CardSubtaskConfig {
    pub require_confirmation: Option<bool>,
    pub require_explicit_ask: bool,
    pub extra_instructions: String,
}

pub type GetCardNumberConfig = CardSubtaskConfig;
pub type GetSecurityCodeConfig = CardSubtaskConfig;

/// [`GetExpirationDateTask`] additionally needs `date.today()`.
#[derive(Default)]
pub struct GetExpirationDateConfig {
    pub require_confirmation: Option<bool>,
    pub require_explicit_ask: bool,
    pub extra_instructions: String,
    pub today: TodaySource,
}

fn card_instructions(
    base: &str,
    audio_specific: &str,
    text_specific: &str,
    confirmation: &str,
    require_confirmation: Option<bool>,
    extra_instructions: &str,
) -> Instructions {
    // Appended after the format: the card templates have no `{extra_instructions}` hole.
    let extra_suffix = if extra_instructions.is_empty() {
        String::new()
    } else {
        format!("\n{extra_instructions}")
    };
    let build = |modality_specific: &str, confirmation_instructions: &str| {
        render(
            base,
            &[
                ("modality_specific", modality_specific),
                ("confirmation_instructions", confirmation_instructions),
            ],
        ) + &extra_suffix
    };
    Instructions {
        common: String::new(),
        audio: Some(build(
            audio_specific,
            if require_confirmation == Some(false) { "" } else { confirmation },
        )),
        text: Some(build(
            text_specific,
            if require_confirmation == Some(true) { confirmation } else { "" },
        )),
    }
}

fn shared_card_tools() -> [ToolSpec; 2] {
    [
        tool(
            p::DECLINE_TOOL,
            p::DECLINE_DESC,
            &[Param::string("reason", p::DECLINE_ARG_REASON)],
            update_tool_flags(true),
        ),
        tool(
            p::RESTART_TOOL,
            p::RESTART_DESC,
            &[Param::string("reason", p::RESTART_ARG_REASON)],
            update_tool_flags(true),
        ),
    ]
}

/// `class CardCaptureDeclinedError(ToolError)`. The orchestrator `except`s the two subclasses
/// separately, so they are [`ToolErrorKind::Subclass`](crate::workflows::ToolErrorKind) identities.
pub const CARD_CAPTURE_DECLINED: &str = "CardCaptureDeclinedError";

/// `class CardCollectionRestartError(ToolError)`: the orchestrator's `continue` arm.
pub const CARD_COLLECTION_RESTART: &str = "CardCollectionRestartError";

/// The class names of the two `ToolError` subclasses, as `(declined, restart)`: the strings
/// `WorkflowReply.error_class` carries, so a rename cannot desync the two sides.
#[lk_agents_abi::host(export(pid = false))]
pub fn card_error_class_names() -> (&'static str, &'static str) {
    (CARD_CAPTURE_DECLINED, CARD_COLLECTION_RESTART)
}

/// `_assemble_credit_card_result(...)`: the pure half of `GetCreditCardTask`, whose loop stays in
/// Python because `TaskGroup` needs Python task factories.
///
/// The cardholder name is `f"{first} {last}"` with a missing part written as `None`, as the
/// reference does.
#[lk_agents_abi::host(export(name = "assemble_credit_card", pid = false))]
pub fn assemble_credit_card(
    cardholder: &GetNameResult,
    card_number: &GetCardNumberResult,
    security_code: &GetSecurityCodeResult,
    expiration_date: &GetExpirationDateResult,
) -> GetCreditCardResult {
    assemble_credit_card_result(cardholder, card_number, security_code, expiration_date)
}

/// `CardCaptureDeclinedError.__init__` formats the message; `.reason` stays reachable.
pub fn card_capture_declined(reason: String) -> ToolError {
    ToolError::subclass(CARD_CAPTURE_DECLINED, render(p::DECLINED, &[("reason", &reason)]), reason)
}

/// `CardCollectionRestartError.__init__`.
pub fn card_collection_restart(reason: String) -> ToolError {
    ToolError::subclass(
        CARD_COLLECTION_RESTART,
        render(p::RESTARTING, &[("reason", &reason)]),
        reason,
    )
}

/// `decline_card_capture` / `restart_card_collection`, module-level tools shared by the three
/// sub-tasks; `None` for any other name.
fn handle_shared_tool<T>(name: &str, args_json: &Value, ctx: &mut TaskCtx<T>) -> Option<ToolReply> {
    let build: fn(String) -> ToolError = match name {
        p::DECLINE_TOOL => card_capture_declined,
        p::RESTART_TOOL => card_collection_restart,
        // Every other tool name is one of the sub-task's own.
        other => {
            let _ = other;
            return None;
        }
    };
    Some(match args::string(args_json, name, "reason") {
        Ok(reason) => {
            ctx.fail(build(reason));
            ToolReply::Silent
        }
        Err(e) => ToolReply::Error(e),
    })
}

// ---------------------------------------------------------------------------- card number

pub struct GetCardNumberTask {
    instructions: Instructions,
    require_confirmation: Option<bool>,
    require_explicit_ask: bool,
    card_number: String,
    confirm: ConfirmSlot<String>,
}

impl GetCardNumberTask {
    pub fn new(cfg: GetCardNumberConfig) -> Self {
        GetCardNumberTask {
            instructions: card_instructions(
                p::CARD_NUMBER_BASE,
                p::CARD_NUMBER_AUDIO,
                p::CARD_NUMBER_TEXT,
                p::CARD_NUMBER_CONFIRMATION,
                cfg.require_confirmation,
                &cfg.extra_instructions,
            ),
            require_confirmation: cfg.require_confirmation,
            require_explicit_ask: cfg.require_explicit_ask,
            card_number: String::new(),
            confirm: ConfirmSlot::default(),
        }
    }

    pub fn on_enter_prompt(&self) -> &'static str {
        p::CARD_NUMBER_ON_ENTER
    }

    async fn update_card_number(
        &mut self,
        card_number: &str,
        ctx: &mut TaskCtx<GetCardNumberResult>,
    ) -> ToolReply {
        let card_number = digits_only(card_number);

        // A reply, not a `ToolError`: the model is told what to ask next and the task stays alive.
        if card_number.len() < 13 || card_number.len() > 19 {
            return ToolReply::text(p::CARD_NUMBER_BAD_LENGTH);
        }

        self.card_number = card_number.clone();

        if !confirmation_required(self.require_confirmation, ctx.modality()) {
            // Luhn only on the no-confirmation path; `confirm_card_number` checks it otherwise.
            if !luhn_valid(&self.card_number) {
                return ToolReply::text(p::CARD_NUMBER_INVALID);
            }
            ctx.complete(GetCardNumberResult {
                issuer: issuer_for(&self.card_number).to_owned(),
                card_number: self.card_number.clone(),
            });
            return ToolReply::Silent;
        }

        let confirm_tool = tool(
            p::CONFIRM_CARD_NUMBER_TOOL,
            p::CONFIRM_CARD_NUMBER_DESC,
            &[Param::string("repeated_card_number", p::CONFIRM_CARD_NUMBER_ARG)],
            Default::default(),
        );
        let revision = ctx.tools().upsert_last(confirm_tool);
        self.confirm.install(revision, card_number);

        ToolReply::text(p::CARD_NUMBER_UPDATED)
    }

    fn confirm_card_number(
        &mut self,
        repeated: &str,
        ctx: &mut TaskCtx<GetCardNumberResult>,
    ) -> ToolReply {
        let Some(captured) = self.confirm.captured_for(ctx.turn_tools_revision()).cloned() else {
            return ToolReply::unknown_tool(p::CONFIRM_CARD_NUMBER_TOOL);
        };
        if digits_only(repeated) != captured {
            return ToolReply::text(p::CARD_NUMBER_MISMATCH);
        }
        if !luhn_valid(&captured) {
            return ToolReply::text(p::CARD_NUMBER_INVALID);
        }
        ctx.complete(GetCardNumberResult {
            issuer: issuer_for(&captured).to_owned(),
            card_number: captured,
        });
        ToolReply::Silent
    }
}

#[async_trait]
impl Task for GetCardNumberTask {
    type Output = GetCardNumberResult;

    fn instructions(&self) -> Instructions {
        self.instructions.clone()
    }

    fn tools(&self) -> Vec<ToolSpec> {
        let mut tools = vec![tool(
            p::UPDATE_CARD_NUMBER_TOOL,
            p::UPDATE_CARD_NUMBER_DESC,
            &[Param::string("card_number", p::UPDATE_CARD_NUMBER_ARG)],
            update_tool_flags(self.require_explicit_ask),
        )];
        tools.extend(shared_card_tools());
        tools
    }

    async fn call(
        &mut self,
        name: &str,
        args_json: Value,
        ctx: &mut TaskCtx<GetCardNumberResult>,
    ) -> ToolReply {
        if let Some(reply) = handle_shared_tool(name, &args_json, ctx) {
            return reply;
        }
        match name {
            p::UPDATE_CARD_NUMBER_TOOL => match args::string(&args_json, name, "card_number") {
                Ok(v) => self.update_card_number(&v, ctx).await,
                Err(e) => ToolReply::Error(e),
            },
            p::CONFIRM_CARD_NUMBER_TOOL => {
                match args::string(&args_json, name, "repeated_card_number") {
                    Ok(v) => self.confirm_card_number(&v, ctx),
                    Err(e) => ToolReply::Error(e),
                }
            }
            other => ToolReply::unknown_tool(other),
        }
    }
}

// ---------------------------------------------------------------------------- security code

pub struct GetSecurityCodeTask {
    instructions: Instructions,
    require_confirmation: Option<bool>,
    require_explicit_ask: bool,
    security_code: String,
    confirm: ConfirmSlot<String>,
}

impl GetSecurityCodeTask {
    pub fn new(cfg: GetSecurityCodeConfig) -> Self {
        GetSecurityCodeTask {
            instructions: card_instructions(
                p::SECURITY_CODE_BASE,
                p::SECURITY_CODE_AUDIO,
                p::SECURITY_CODE_TEXT,
                p::SECURITY_CODE_CONFIRMATION,
                cfg.require_confirmation,
                &cfg.extra_instructions,
            ),
            require_confirmation: cfg.require_confirmation,
            require_explicit_ask: cfg.require_explicit_ask,
            security_code: String::new(),
            confirm: ConfirmSlot::default(),
        }
    }

    pub fn on_enter_prompt(&self) -> &'static str {
        p::SECURITY_CODE_ON_ENTER
    }

    async fn update_security_code(
        &mut self,
        security_code: &str,
        ctx: &mut TaskCtx<GetSecurityCodeResult>,
    ) -> ToolReply {
        // `.strip()` then `.isdigit()`: interior spaces are rejected, unlike the card number.
        let stripped = security_code.trim();
        if !(3..=4).contains(&stripped.chars().count())
            || stripped.is_empty()
            || !stripped.chars().all(|c| c.is_ascii_digit())
        {
            return ToolReply::text(p::SECURITY_CODE_BAD_LENGTH);
        }

        self.security_code = stripped.to_owned();

        if !confirmation_required(self.require_confirmation, ctx.modality()) {
            ctx.complete(GetSecurityCodeResult { security_code: self.security_code.clone() });
            return ToolReply::Silent;
        }

        let confirm_tool = tool(
            p::CONFIRM_SECURITY_CODE_TOOL,
            p::CONFIRM_SECURITY_CODE_DESC,
            &[Param::string("repeated_security_code", p::CONFIRM_SECURITY_CODE_ARG)],
            Default::default(),
        );
        let revision = ctx.tools().upsert_last(confirm_tool);
        self.confirm.install(revision, stripped.to_owned());

        ToolReply::text(p::SECURITY_CODE_UPDATED)
    }

    fn confirm_security_code(
        &mut self,
        repeated: &str,
        ctx: &mut TaskCtx<GetSecurityCodeResult>,
    ) -> ToolReply {
        let Some(captured) = self.confirm.captured_for(ctx.turn_tools_revision()).cloned() else {
            return ToolReply::unknown_tool(p::CONFIRM_SECURITY_CODE_TOOL);
        };
        if repeated.trim() != captured {
            return ToolReply::text(p::SECURITY_CODE_MISMATCH);
        }
        ctx.complete(GetSecurityCodeResult { security_code: captured });
        ToolReply::Silent
    }
}

#[async_trait]
impl Task for GetSecurityCodeTask {
    type Output = GetSecurityCodeResult;

    fn instructions(&self) -> Instructions {
        self.instructions.clone()
    }

    fn tools(&self) -> Vec<ToolSpec> {
        let mut tools = vec![tool(
            p::UPDATE_SECURITY_CODE_TOOL,
            p::UPDATE_SECURITY_CODE_DESC,
            &[Param::string("security_code", p::UPDATE_SECURITY_CODE_ARG)],
            update_tool_flags(self.require_explicit_ask),
        )];
        tools.extend(shared_card_tools());
        tools
    }

    async fn call(
        &mut self,
        name: &str,
        args_json: Value,
        ctx: &mut TaskCtx<GetSecurityCodeResult>,
    ) -> ToolReply {
        if let Some(reply) = handle_shared_tool(name, &args_json, ctx) {
            return reply;
        }
        match name {
            p::UPDATE_SECURITY_CODE_TOOL => match args::string(&args_json, name, "security_code") {
                Ok(v) => self.update_security_code(&v, ctx).await,
                Err(e) => ToolReply::Error(e),
            },
            p::CONFIRM_SECURITY_CODE_TOOL => {
                match args::string(&args_json, name, "repeated_security_code") {
                    Ok(v) => self.confirm_security_code(&v, ctx),
                    Err(e) => ToolReply::Error(e),
                }
            }
            other => ToolReply::unknown_tool(other),
        }
    }
}

// ---------------------------------------------------------------------------- expiration date

pub struct GetExpirationDateTask {
    instructions: Instructions,
    require_confirmation: Option<bool>,
    require_explicit_ask: bool,
    today: TodaySource,
    expiration_date: String,
    confirm: ConfirmSlot<(i64, i64, String)>,
}

impl GetExpirationDateTask {
    pub fn new(cfg: GetExpirationDateConfig) -> Self {
        GetExpirationDateTask {
            instructions: card_instructions(
                p::EXPIRATION_BASE,
                p::EXPIRATION_AUDIO,
                p::EXPIRATION_TEXT,
                p::EXPIRATION_CONFIRMATION,
                cfg.require_confirmation,
                &cfg.extra_instructions,
            ),
            require_confirmation: cfg.require_confirmation,
            require_explicit_ask: cfg.require_explicit_ask,
            today: cfg.today,
            expiration_date: String::new(),
            confirm: ConfirmSlot::default(),
        }
    }

    pub fn on_enter_prompt(&self) -> &'static str {
        p::EXPIRATION_ON_ENTER
    }

    /// `_is_expired`: strict on `(year, month)`, so a card expiring this month is not expired.
    pub fn is_expired(&self, month: i64, year: i64) -> bool {
        let today = self.today.today();
        let full_year = 2000 + year;
        (full_year, month) < (i64::from(today.year()), i64::from(today.month()))
    }

    async fn update_expiration_date(
        &mut self,
        month: i64,
        year: i64,
        ctx: &mut TaskCtx<GetExpirationDateResult>,
    ) -> ToolReply {
        if !(1..=12).contains(&month) {
            return ToolReply::text(p::EXPIRATION_BAD_MONTH);
        }
        if !(0..=99).contains(&year) {
            return ToolReply::text(p::EXPIRATION_BAD_YEAR);
        }
        if self.is_expired(month, year) {
            return ToolReply::text(p::EXPIRATION_EXPIRED);
        }

        self.expiration_date = format!("{month:02}/{year:02}");

        if !confirmation_required(self.require_confirmation, ctx.modality()) {
            ctx.complete(GetExpirationDateResult { date: self.expiration_date.clone() });
            return ToolReply::Silent;
        }

        let confirm_tool = tool(
            p::CONFIRM_EXPIRATION_TOOL,
            p::CONFIRM_EXPIRATION_DESC,
            &[
                Param::integer("repeated_expiration_month", p::CONFIRM_EXPIRATION_ARG_MONTH),
                Param::integer("repeated_expiration_year", p::CONFIRM_EXPIRATION_ARG_YEAR),
            ],
            Default::default(),
        );
        let revision = ctx.tools().upsert_last(confirm_tool);
        // Captured after the assignment above, so the string matches the month/year.
        self.confirm.install(revision, (month, year, self.expiration_date.clone()));

        ToolReply::text(p::EXPIRATION_UPDATED)
    }

    fn confirm_expiration_date(
        &mut self,
        repeated_month: i64,
        repeated_year: i64,
        ctx: &mut TaskCtx<GetExpirationDateResult>,
    ) -> ToolReply {
        let Some((month, year, date)) =
            self.confirm.captured_for(ctx.turn_tools_revision()).cloned()
        else {
            return ToolReply::unknown_tool(p::CONFIRM_EXPIRATION_TOOL);
        };
        if repeated_month != month || repeated_year != year {
            return ToolReply::text(p::EXPIRATION_MISMATCH);
        }
        ctx.complete(GetExpirationDateResult { date });
        ToolReply::Silent
    }
}

#[async_trait]
impl Task for GetExpirationDateTask {
    type Output = GetExpirationDateResult;

    fn instructions(&self) -> Instructions {
        self.instructions.clone()
    }

    fn tools(&self) -> Vec<ToolSpec> {
        let mut tools = vec![tool(
            p::UPDATE_EXPIRATION_TOOL,
            p::UPDATE_EXPIRATION_DESC,
            &[
                Param::integer("expiration_month", p::UPDATE_EXPIRATION_ARG_MONTH),
                Param::integer("expiration_year", p::UPDATE_EXPIRATION_ARG_YEAR),
            ],
            update_tool_flags(self.require_explicit_ask),
        )];
        tools.extend(shared_card_tools());
        tools
    }

    async fn call(
        &mut self,
        name: &str,
        args_json: Value,
        ctx: &mut TaskCtx<GetExpirationDateResult>,
    ) -> ToolReply {
        if let Some(reply) = handle_shared_tool(name, &args_json, ctx) {
            return reply;
        }
        match name {
            p::UPDATE_EXPIRATION_TOOL => {
                let month = match args::integer(&args_json, name, "expiration_month") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                let year = match args::integer(&args_json, name, "expiration_year") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                self.update_expiration_date(month, year, ctx).await
            }
            p::CONFIRM_EXPIRATION_TOOL => {
                let month = match args::integer(&args_json, name, "repeated_expiration_month") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                let year = match args::integer(&args_json, name, "repeated_expiration_year") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                self.confirm_expiration_date(month, year, ctx)
            }
            other => ToolReply::unknown_tool(other),
        }
    }
}

/// `"".join([d for d in card_number if d.isdigit()])`, ASCII-only. See [`luhn_valid`].
pub fn digits_only(s: &str) -> String {
    s.chars().filter(char::is_ascii_digit).collect()
}
