use lk_agents_abi::utils::dates::{Date, Time, TodaySource};
use crate::workflows::builtins::instructions::render;
use crate::workflows::builtins::prompts::dob as p;
use crate::workflows::builtins::schema::{Param, nullary, tool};
use crate::workflows::{
    ConfirmSlot, Task, TaskCtx, TaskOptions, ToolError, ToolReply, args, confirmation_required,
    update_tool_flags,
};
use async_trait::async_trait;
use lk_agents_abi::ToolSpec;
use lk_agents_voice::generation::Instructions;
use serde_json::Value;

pub use lk_agents_abi::voice::workflows::GetDOBResult;

#[derive(Default)]
pub struct GetDOBConfig {
    pub extra_instructions: String,
    /// Adds the `update_time` tool and its instructions paragraph, together.
    pub include_time: bool,
    pub require_confirmation: Option<bool>,
    pub require_explicit_ask: bool,
    pub options: TaskOptions,
    /// `date.today()`. See [`TodaySource`] for why this is injected.
    pub today: TodaySource,
}

/// What the `confirm_dob` closure captured: `dob` and `self._current_time`.
type Captured = (Option<Date>, Option<Time>);

pub struct GetDOBTask {
    instructions: Instructions,
    include_time: bool,
    user_tools: Vec<ToolSpec>,
    require_confirmation: Option<bool>,
    require_explicit_ask: bool,
    today: TodaySource,
    current_dob: Option<Date>,
    current_time: Option<Time>,
    confirm: ConfirmSlot<Captured>,
}

impl GetDOBTask {
    pub fn new(cfg: GetDOBConfig) -> Self {
        GetDOBTask {
            instructions: build_instructions(&cfg),
            include_time: cfg.include_time,
            user_tools: cfg.options.tools,
            require_confirmation: cfg.require_confirmation,
            require_explicit_ask: cfg.require_explicit_ask,
            today: cfg.today,
            current_dob: None,
            current_time: None,
            confirm: ConfirmSlot::default(),
        }
    }

    pub fn on_enter_prompt(&self) -> &'static str {
        if self.include_time { p::ON_ENTER_WITH_TIME } else { p::ON_ENTER }
    }

    async fn update_dob(
        &mut self,
        year: i64,
        month: i64,
        day: i64,
        ctx: &mut TaskCtx<GetDOBResult>,
    ) -> ToolReply {
        let today = self.today.today();
        let year = normalize_two_digit_year(year, today);

        let dob = match Date::new(year, month, day) {
            Ok(d) => d,
            Err(e) => {
                return ToolReply::Error(ToolError::plain(render(
                    p::INVALID_DATE,
                    &[("error", &e.to_string())],
                )));
            }
        };

        if dob > today {
            return ToolReply::Error(ToolError::plain(render(
                p::FUTURE,
                &[("formatted", &dob.format_long())],
            )));
        }

        self.current_dob = Some(dob);

        if !confirmation_required(self.require_confirmation, ctx.modality()) {
            ctx.complete(GetDOBResult { date_of_birth: dob, time_of_birth: self.current_time });
            return ToolReply::Silent;
        }

        let revision = ctx.tools().upsert_last(nullary(p::CONFIRM_TOOL, p::CONFIRM_DESC));
        self.confirm.install(revision, (Some(dob), self.current_time));

        let mut response = render(p::DOB_UPDATED, &[("formatted_date", &dob.format_long())]);
        if let Some(t) = self.current_time {
            response
                .push_str(&render(p::DOB_UPDATED_AT_TIME, &[("formatted_time", &t.format_12h())]));
        }
        response.push_str(p::CONFIRM_TAIL);
        ToolReply::Text(response)
    }

    async fn update_time(
        &mut self,
        hour: i64,
        minute: i64,
        ctx: &mut TaskCtx<GetDOBResult>,
    ) -> ToolReply {
        let birth_time = match Time::new(hour, minute) {
            Ok(t) => t,
            Err(e) => {
                return ToolReply::Error(ToolError::plain(render(
                    p::INVALID_TIME,
                    &[("error", &e.to_string())],
                )));
            }
        };

        self.current_time = Some(birth_time);
        let confirmation = confirmation_required(self.require_confirmation, ctx.modality());

        if !confirmation && let Some(dob) = self.current_dob {
            ctx.complete(GetDOBResult { date_of_birth: dob, time_of_birth: self.current_time });
            return ToolReply::Silent;
        }

        if confirmation {
            let revision = ctx.tools().upsert_last(nullary(p::CONFIRM_TOOL, p::CONFIRM_DESC));
            self.confirm.install(revision, (self.current_dob, self.current_time));
        }

        let formatted_time = birth_time.format_12h();
        let mut response = match self.current_dob {
            Some(dob) => render(
                p::DATE_AND_TIME_UPDATED,
                &[("formatted_date", &dob.format_long()), ("formatted_time", &formatted_time)],
            ),
            None => render(p::TIME_UPDATED, &[("formatted_time", &formatted_time)]),
        };

        if confirmation {
            response.push_str(p::CONFIRM_TIME_TAIL);
        } else {
            // Only reachable with no date yet.
            response.push_str(p::NO_DATE_YET_TAIL);
        }
        ToolReply::Text(response)
    }

    fn confirm_dob(&mut self, ctx: &mut TaskCtx<GetDOBResult>) -> ToolReply {
        let Some(&(captured_dob, captured_time)) =
            self.confirm.captured_for(ctx.turn_tools_revision())
        else {
            return ToolReply::unknown_tool(p::CONFIRM_TOOL);
        };

        if captured_dob != self.current_dob || captured_time != self.current_time {
            // stale closure: update_dob/update_time ran again after this confirm tool was installed
            return ToolReply::text(p::STALE);
        }

        let Some(dob) = self.current_dob else {
            return ToolReply::text(p::NO_DATE_YET);
        };

        ctx.complete(GetDOBResult { date_of_birth: dob, time_of_birth: self.current_time });
        ToolReply::Silent
    }
}

#[async_trait]
impl Task for GetDOBTask {
    type Output = GetDOBResult;

    fn instructions(&self) -> Instructions {
        self.instructions.clone()
    }

    fn tools(&self) -> Vec<ToolSpec> {
        let mut tools = self.user_tools.clone();
        tools.push(tool(
            p::UPDATE_TOOL,
            p::UPDATE_DESC,
            &[
                Param::integer("year", p::UPDATE_ARG_YEAR),
                Param::integer("month", p::UPDATE_ARG_MONTH),
                Param::integer("day", p::UPDATE_ARG_DAY),
            ],
            update_tool_flags(self.require_explicit_ask),
        ));
        tools.push(tool(
            p::DECLINE_TOOL,
            p::DECLINE_DESC,
            &[Param::string("reason", p::DECLINE_ARG_REASON)],
            update_tool_flags(true),
        ));
        // Appended last, and without IGNORE_ON_ENTER even when the update tool has it.
        if self.include_time {
            tools.push(tool(
                p::UPDATE_TIME_TOOL,
                p::UPDATE_TIME_DESC,
                &[
                    Param::integer("hour", p::UPDATE_TIME_ARG_HOUR),
                    Param::integer("minute", p::UPDATE_TIME_ARG_MINUTE),
                ],
                Default::default(),
            ));
        }
        tools
    }

    async fn call(
        &mut self,
        name: &str,
        args_json: Value,
        ctx: &mut TaskCtx<GetDOBResult>,
    ) -> ToolReply {
        match name {
            p::UPDATE_TOOL => {
                let year = match args::integer(&args_json, name, "year") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                let month = match args::integer(&args_json, name, "month") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                let day = match args::integer(&args_json, name, "day") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                self.update_dob(year, month, day, ctx).await
            }
            p::UPDATE_TIME_TOOL if self.include_time => {
                let hour = match args::integer(&args_json, name, "hour") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                let minute = match args::integer(&args_json, name, "minute") {
                    Ok(v) => v,
                    Err(e) => return ToolReply::Error(e),
                };
                self.update_time(hour, minute, ctx).await
            }
            p::CONFIRM_TOOL => self.confirm_dob(ctx),
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

/// The century pivot: two-digit years (`0 <= year < 100`) map near today, pivoting on
/// `today.year % 100`, which moves every year.
pub fn normalize_two_digit_year(year: i64, today: Date) -> i64 {
    if (0..100).contains(&year) {
        let current_yy = i64::from(today.year() % 100);
        year + if year <= current_yy { 2000 } else { 1900 }
    } else {
        year
    }
}

fn build_instructions(cfg: &GetDOBConfig) -> Instructions {
    let time_instructions = if cfg.include_time { p::TIME_INSTRUCTIONS } else { "" };
    let confirmation = p::CONFIRMATION_INSTRUCTION;
    let common = |modality_specific: &str, confirmation_instructions: &str| {
        render(
            p::BASE_INSTRUCTIONS,
            &[
                ("modality_specific", modality_specific),
                ("time_instructions", time_instructions),
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
