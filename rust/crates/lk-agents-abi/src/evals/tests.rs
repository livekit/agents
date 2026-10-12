//! Calculators, tested as such: ordered inputs in, exact strings and booleans out. Every await is
//! bounded by [`within`]; the concurrency test uses a barrier that deadlocks if the fan-out is
//! sequential.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use lk_agents_abi::{ChatRequest, Closed, Llm, LlmStream, StreamSource, ToolChoice, Ts, VirtualClock};
use lk_agents_abi::TokioSpawner;
use crate::errors::{ApiError, ProviderError};
use lk_agents_abi::{
    AgentConfigUpdate, AgentHandoffItem, ChatContext, ChatItem, ChatMessage, ChoiceDelta,
    FunctionCall, FunctionCallOutput, FunctionToolCall, LlmChunk, Role,
};

use crate::evals::*;

// --- bounded-await helpers ---------------------------------------------------------------------

const LIMIT: Duration = Duration::from_secs(10);

/// Await `fut`, failing with `what` if it does not finish inside [`LIMIT`].
async fn within<F: Future>(what: &str, fut: F) -> F::Output {
    match tokio::time::timeout(LIMIT, fut).await {
        Ok(v) => v,
        Err(_) => panic!("timed out after {LIMIT:?} waiting for {what}"),
    }
}

// --- fixtures ----------------------------------------------------------------------------------

const T0: SystemTime = SystemTime::UNIX_EPOCH;

fn msg(id: &str, role: Role, text: &str) -> ChatItem {
    ChatMessage::text(id, role, text, T0).into()
}

fn interrupted(id: &str, role: Role, text: &str) -> ChatItem {
    let mut m = ChatMessage::text(id, role, text, T0);
    m.interrupted = true;
    m.into()
}

fn call(id: &str, name: &str, arguments: &str) -> ChatItem {
    FunctionCall {
        id: id.into(),
        call_id: format!("{id}_call"),
        arguments: arguments.into(),
        name: name.into(),
        created_at: T0,
        extra: Default::default(),
        group_id: None,
    }
    .into()
}

fn output(id: &str, out: &str, is_error: bool) -> ChatItem {
    FunctionCallOutput::new(id, format!("{id}_call"), out, is_error, T0).into()
}

fn handoff(id: &str, old: Option<&str>, new: &str) -> ChatItem {
    AgentHandoffItem {
        id: id.into(),
        old_agent_id: old.map(str::to_string),
        new_agent_id: new.into(),
        created_at: T0,
    }
    .into()
}

fn config(id: &str, instructions: Option<&str>, added: Option<&[&str]>) -> ChatItem {
    AgentConfigUpdate {
        id: id.into(),
        instructions: instructions.map(str::to_string),
        tools_added: added.map(|v| v.iter().map(|s| (*s).to_string()).collect()),
        tools_removed: None,
        created_at: T0,
        tools: None,
    }
    .into()
}

fn judgment(verdict: Verdict) -> JudgmentResult {
    JudgmentResult::new(verdict, "because")
}

fn result_of(verdicts: &[(&str, Verdict)]) -> EvaluationResult {
    verdicts.iter().map(|(n, v)| ((*n).to_string(), judgment(*v))).collect()
}

/// An `Llm` that answers with one scripted `submit_verdict` tool call and records the request.
struct FakeLlm {
    /// `None`: answer with prose and no tool call, as providers do under load.
    arguments: Option<String>,
    seen: std::sync::Mutex<Option<ChatRequest>>,
    closed: Arc<AtomicUsize>,
    fail: bool,
}

impl FakeLlm {
    fn verdict(arguments: &str) -> Arc<FakeLlm> {
        Arc::new(FakeLlm {
            arguments: Some(arguments.to_string()),
            seen: std::sync::Mutex::new(None),
            closed: Arc::new(AtomicUsize::new(0)),
            fail: false,
        })
    }

    fn silent() -> Arc<FakeLlm> {
        Arc::new(FakeLlm {
            arguments: None,
            seen: std::sync::Mutex::new(None),
            closed: Arc::new(AtomicUsize::new(0)),
            fail: false,
        })
    }

    fn broken() -> Arc<FakeLlm> {
        Arc::new(FakeLlm {
            arguments: None,
            seen: std::sync::Mutex::new(None),
            closed: Arc::new(AtomicUsize::new(0)),
            fail: true,
        })
    }

    fn request(&self) -> ChatRequest {
        self.seen.lock().unwrap().clone().expect("chat() was called")
    }
}

#[async_trait]
impl Llm for FakeLlm {
    fn label(&self) -> String {
        "test.FakeLlm".to_string()
    }

    async fn chat(&self, req: &ChatRequest) -> Result<LlmStream, ProviderError> {
        *self.seen.lock().unwrap() = Some(req.clone());
        if self.fail {
            return Err(ApiError::new("gateway said no").into());
        }
        Ok(Box::new(FakeStream { arguments: self.arguments.clone(), closed: self.closed.clone() }))
    }
}

struct FakeStream {
    arguments: Option<String>,
    closed: Arc<AtomicUsize>,
}

#[async_trait]
impl StreamSource<LlmChunk> for FakeStream {
    async fn next(&mut self) -> Result<Option<LlmChunk>, ProviderError> {
        let Some(arguments) = self.arguments.take() else { return Ok(None) };
        Ok(Some(LlmChunk {
            id: "c1".into(),
            delta: Some(ChoiceDelta {
                role: Some(Role::Assistant),
                tool_calls: vec![FunctionToolCall {
                    name: "submit_verdict".into(),
                    arguments,
                    call_id: "call_1".into(),
                    extra: None,
                }],
                ..Default::default()
            }),
            ..Default::default()
        }))
    }

    async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
        self.closed.fetch_add(1, Ordering::SeqCst);
        Closed::Clean
    }
}

fn judge_llm(llm: Arc<FakeLlm>, model: &str) -> Arc<JudgeLlm> {
    Arc::new(JudgeLlm::new(llm, model, None, Arc::new(VirtualClock::new())))
}

fn req(chat_ctx: ChatContext, llm: Option<Arc<JudgeLlm>>) -> EvaluateRequest {
    EvaluateRequest { chat_ctx: Arc::new(chat_ctx), reference: None, llm }
}

// --- transcript --------------------------------------------------------------------------------

// Catches: an item kind silently dropped from the judge transcript (all five kinds pinned), the
// `[interrupted]` marker lost, or the initial agent assignment (`old_agent_id=None`) skipped.
#[test]
fn every_item_kind_reaches_the_transcript_in_order() {
    let ctx = ChatContext::new(vec![
        msg("m1", Role::User, "book me a table"),
        call("f1", "find_table", r#"{"party":2}"#),
        output("o1", "table 7 at 8pm", false),
        msg("m2", Role::Assistant, "table 7, eight o'clock"),
        handoff("h1", Some("host"), "concierge"),
        output("o2", "no such table", true),
        config("c1", Some("Be brief."), Some(&["cancel"])),
        interrupted("m3", Role::Assistant, "your balance is"),
        handoff("h0", None, "greeter"),
    ]);

    assert_eq!(
        transcript::format_chat_ctx(&ctx),
        "user: book me a table\n\
         [function call: find_table({\"party\":2})]\n\
         [function output: table 7 at 8pm]\n\
         assistant: table 7, eight o'clock\n\
         [agent handoff: host -> concierge]\n\
         [function error: no such table]\n\
         [agent config: instructions='Be brief.', tools_added=['cancel']]\n\
         assistant: your balance is [interrupted]\n\
         [agent handoff: None -> greeter]"
    );
}

// Catches: `has_handoffs` counting the initial assignment.
#[test]
fn has_handoffs_ignores_the_initial_assignment_but_not_a_real_one() {
    let initial = ChatContext::new(vec![handoff("h0", None, "greeter")]);
    assert!(!transcript::has_handoffs(&initial));

    let real = ChatContext::new(vec![
        handoff("h0", None, "greeter"),
        handoff("h1", Some("greeter"), "billing"),
    ]);
    assert!(transcript::has_handoffs(&real));
}

// Catches: reading `raw_text_content` instead of `text_content` (assistant markup only).
#[test]
fn assistant_expressive_markup_is_stripped_and_user_text_is_not() {
    let ctx = ChatContext::new(vec![
        msg("m1", Role::User, "say <expr type=\"laugh\"/> please"),
        msg("m2", Role::Assistant, "sure<expr type=\"laugh\"/> here you go"),
    ]);
    let out = transcript::format_chat_ctx(&ctx);
    assert_eq!(out.lines().next().unwrap(), "user: say <expr type=\"laugh\"/> please");
    assert!(
        !out.lines().nth(1).unwrap().contains("<expr"),
        "assistant markup must be stripped: {out}"
    );
}

// Catches: dropping `!r` from the instructions (a newline would split the line).
#[test]
fn multiline_instructions_are_python_repr_escaped_onto_one_line() {
    let ctx = ChatContext::new(vec![config("c1", Some("Be brief.\nDon't guess."), None)]);
    let out = transcript::format_chat_ctx(&ctx);
    assert_eq!(out, r#"[agent config: instructions="Be brief.\nDon't guess."]"#);
    assert_eq!(out.lines().count(), 1);
}

// Catches: the three `if item.X:` guards read as presence, or the unconditional append suppressed.
#[test]
fn empty_config_fields_are_omitted_but_the_line_is_not() {
    let ctx = ChatContext::new(vec![config("c1", Some(""), Some(&[])), config("c2", None, None)]);
    assert_eq!(transcript::format_chat_ctx(&ctx), "[agent config: ]\n[agent config: ]");
}

// Catches: `latest_instructions` searching forwards, or accepting a system message.
#[test]
fn latest_instructions_reads_backwards_and_only_from_config_updates() {
    let ctx = ChatContext::new(vec![
        msg("s0", Role::System, "you are a system prompt, not instructions"),
        config("c1", Some("first"), None),
        config("c2", Some(""), None),
        config("c3", Some("second"), None),
        config("c4", None, None),
    ]);
    assert_eq!(transcript::latest_instructions(&ctx), Some("second"));

    let only_system = ChatContext::new(vec![msg("s0", Role::System, "be nice")]);
    assert_eq!(transcript::latest_instructions(&only_system), None);
}

// Catches: forgetting `reference.copy(exclude_instructions=True)`.
#[test]
fn the_reference_transcript_drops_instruction_messages() {
    let reference = ChatContext::new(vec![
        msg("s0", Role::System, "you are the gold agent"),
        msg("d0", Role::Developer, "developer note"),
        msg("m1", Role::User, "hi"),
        msg("m2", Role::Assistant, "hello"),
    ]);
    assert_eq!(reference_transcript(&reference), "user: hi\nassistant: hello");
}

// --- verdict parsing ---------------------------------------------------------------------------

// Catches: a malformed `submit_verdict` payload coerced into a verdict (five real-world shapes).
#[test]
fn a_malformed_submit_verdict_payload_is_an_error_not_a_verdict() {
    let good = parse_verdict_arguments(r#"{"verdict": "maybe", "reasoning": "unclear"}"#)
        .expect("a well-formed payload");
    assert_eq!(good.verdict, Verdict::Maybe);
    assert_eq!(good.reasoning, "unclear");
    // `instructions` is filled in by the judge afterwards, never by the parse.
    assert_eq!(good.instructions, "");

    for bad in [
        r#"{"verdict": "pass", "reasoning": "#,
        r#"{"verdict": "PASS", "reasoning": "ok"}"#,
        r#"{"verdict": true, "reasoning": "ok"}"#,
        r#"{"verdict": "pass"}"#,
        r#"{"reasoning": "ok"}"#,
        r#"["pass", "ok"]"#,
    ] {
        assert!(
            matches!(parse_verdict_arguments(bad), Err(JudgeError::BadArguments { .. })),
            "{bad} must not parse"
        );
    }
}

// Catches: `temperature` sent to a gpt-5 judge (a 400), or dropped for everyone.
#[test]
fn temperature_is_pinned_to_zero_except_on_gpt_5() {
    let four_o = JudgeLlm::new(FakeLlm::silent(), "openai/gpt-4o-mini", None, Arc::new(VirtualClock::new()));
    assert_eq!(judgment_extra(&four_o).get("temperature"), Some(&serde_json::json!(0.0)));

    for model in ["openai/gpt-5", "openai/gpt-5.1", "gpt-5-mini"] {
        let five = JudgeLlm::new(FakeLlm::silent(), model, None, Arc::new(VirtualClock::new()));
        assert!(judgment_extra(&five).is_empty(), "{model} must not be sent a temperature");
    }
}

// --- the judgment call -------------------------------------------------------------------------

// Catches: a stream that is dropped instead of closed (leaks the host task).
#[tokio::test]
async fn the_judgment_stream_is_always_closed() {
    let llm = FakeLlm::verdict(r#"{"verdict": "fail", "reasoning": "no"}"#);
    let judge = judge_llm(llm.clone(), "openai/gpt-4o-mini");
    within("the judgment", evaluate_with_llm(&judge, "criteria")).await.unwrap();
    assert_eq!(llm.closed.load(Ordering::SeqCst), 1);

    // ...including when the model answered with no tool call at all.
    let silent = FakeLlm::silent();
    let judge = judge_llm(silent.clone(), "openai/gpt-4o-mini");
    let err = within("the silent judgment", evaluate_with_llm(&judge, "criteria")).await;
    assert!(matches!(err, Err(JudgeError::NoVerdict)));
    assert_eq!(silent.closed.load(Ordering::SeqCst), 1);
}

// Catches: `_HandoffJudge` losing its short circuit (a call per session, and `NoLlm` for a pass).
#[tokio::test]
async fn the_handoff_judge_passes_without_an_llm_when_nothing_was_handed_off() {
    let ctx = ChatContext::new(vec![handoff("h0", None, "greeter"), msg("m1", Role::User, "hi")]);
    let judge = handoff_judge(None);
    let out = within("the handoff judge", judge.evaluate(&req(ctx, None))).await.unwrap();
    assert_eq!(out.verdict, Verdict::Pass);
    assert_eq!(out.reasoning, prompts::HANDOFF_NO_HANDOFFS_REASONING);
}

// Catches: a missing LLM swallowed into a verdict, or the error losing the judge's name.
#[tokio::test]
async fn a_judge_with_no_llm_anywhere_reports_which_judge_it_was() {
    let ctx = ChatContext::new(vec![msg("m1", Role::User, "hi")]);
    let err = within("the accuracy judge", accuracy_judge(None).evaluate(&req(ctx, None)))
        .await
        .expect_err("no LLM was supplied");
    assert!(matches!(&err, JudgeError::NoLlm { name } if name == "accuracy"));
    assert!(err.to_string().starts_with("No LLM provided for judge 'accuracy'."));
}

// Catches: flipping the group-LLM vs judge-LLM precedence (`llm or self._llm`: the group's wins).
#[tokio::test]
async fn the_groups_llm_wins_over_the_judges_own() {
    let own = FakeLlm::verdict(r#"{"verdict": "fail", "reasoning": "own"}"#);
    let group_llm = FakeLlm::verdict(r#"{"verdict": "pass", "reasoning": "group"}"#);
    let judge = accuracy_judge(Some(judge_llm(own.clone(), "openai/gpt-4o-mini")));

    let ctx = ChatContext::new(vec![msg("m1", Role::User, "hi")]);
    let out = within(
        "the accuracy judge",
        judge.evaluate(&req(ctx, Some(judge_llm(group_llm.clone(), "openai/gpt-4o-mini")))),
    )
    .await
    .unwrap();

    assert_eq!(out.reasoning, "group");
    assert!(own.seen.lock().unwrap().is_none(), "the judge's own LLM must not be called");
}

// Catches: the criteria not reaching the stored judgment; also pins the prompt's shape.
#[tokio::test]
async fn an_llm_judge_echoes_its_criteria_and_frames_the_transcript() {
    let llm = FakeLlm::verdict(r#"{"verdict": "pass", "reasoning": "grounded"}"#);
    let judge = conciseness_judge(None);
    let ctx = ChatContext::new(vec![msg("m1", Role::User, "hi")]);
    let request = req(ctx, Some(judge_llm(llm, "openai/gpt-4o-mini")));

    assert_eq!(
        judge.prompt(&request),
        format!(
            "Criteria: {}\n\nConversation:\nuser: hi\n\n{}",
            prompts::CONCISENESS.instructions,
            prompts::EVALUATE_TRAILER
        )
    );

    let out = within("the conciseness judge", judge.evaluate(&request)).await.unwrap();
    assert_eq!(out.instructions, prompts::CONCISENESS.instructions);
}

// Catches: the task-completion judge not embedding the agent's instructions.
#[tokio::test]
async fn the_task_completion_judge_embeds_the_latest_agent_instructions() {
    let judge = task_completion_judge(None);
    let with_instructions = req(
        ChatContext::new(vec![
            config("c1", Some("Book tables."), None),
            msg("m1", Role::User, "hi"),
        ]),
        None,
    );
    let (prompt, found) = judge.prompt(&with_instructions);
    assert!(found);
    assert!(prompt.contains("Agent Instructions:\nBook tables."), "{prompt}");

    let without = req(ChatContext::new(vec![msg("m1", Role::User, "hi")]), None);
    let (prompt, found) = judge.prompt(&without);
    assert!(!found, "the caller must be told the goal is unknown");
    assert!(!prompt.contains("Agent Instructions:"), "{prompt}");
    assert!(prompt.starts_with(prompts::TASK_COMPLETION_CRITERIA));
}

// --- the four aggregates -----------------------------------------------------------------------

// Catches: the four aggregates implemented as variations of one another (2 pass, 1 fail, 1 maybe).
#[test]
fn the_four_aggregates_disagree_correctly_on_a_mixed_set() {
    let mixed = result_of(&[
        ("accuracy", Verdict::Pass),
        ("relevancy", Verdict::Pass),
        ("safety", Verdict::Fail),
        ("tool_use", Verdict::Maybe),
    ]);

    assert_eq!(mixed.score(), 2.5 / 4.0);
    assert!(!mixed.all_passed(), "a fail and a maybe are both not-passed");
    assert!(mixed.any_passed());
    assert!(!mixed.majority_passed(), "2 of 4 is not strictly more than half");
    assert!(!mixed.none_failed(), "safety failed");
}

// Catches: "fixing" the empty case: `score` fails closed at 0.0, the booleans are vacuously true.
#[test]
fn an_empty_result_scores_zero_while_the_vacuous_booleans_hold() {
    let empty = EvaluationResult::new();
    assert!(empty.is_empty());
    assert_eq!(empty.score(), 0.0);
    assert!(empty.all_passed());
    assert!(empty.majority_passed());
    assert!(empty.none_failed());
    assert!(!empty.any_passed(), "`any` is the one that is not vacuously true");
}

// Catches: `majority_passed` at an odd count (2 of 3 yes, 1 of 3 no).
#[test]
fn majority_passed_is_strict() {
    let two_of_three =
        result_of(&[("a", Verdict::Pass), ("b", Verdict::Pass), ("c", Verdict::Fail)]);
    assert!(two_of_three.majority_passed());

    let one_of_three =
        result_of(&[("a", Verdict::Pass), ("b", Verdict::Maybe), ("c", Verdict::Maybe)]);
    assert!(!one_of_three.majority_passed(), "maybes do not count towards the half");
}

// Catches: the result map losing `dict` semantics (a re-insert replaces in place).
#[test]
fn re_inserting_a_name_replaces_in_place() {
    let mut result = result_of(&[("a", Verdict::Pass), ("b", Verdict::Fail)]);
    result.insert("a", judgment(Verdict::Fail));

    assert_eq!(result.len(), 2);
    assert_eq!(result.judgments().map(|(n, _)| n).collect::<Vec<_>>(), vec!["a", "b"]);
    assert_eq!(result.get("a").unwrap().verdict, Verdict::Fail);
}

// --- the group ---------------------------------------------------------------------------------

/// A judge with a fixed answer, used to pin the group's collection order.
struct Fixed {
    name: &'static str,
    verdict: Verdict,
}

#[async_trait]
impl Evaluator for Fixed {
    fn name(&self) -> String {
        self.name.to_string()
    }
    async fn evaluate(&self, _req: &EvaluateRequest) -> Result<JudgmentResult, JudgeError> {
        Ok(judgment(self.verdict))
    }
}

/// A judge that cannot answer until another judge has also started; deadlocks if sequential.
struct Barriered {
    name: &'static str,
    barrier: Arc<tokio::sync::Barrier>,
}

#[async_trait]
impl Evaluator for Barriered {
    fn name(&self) -> String {
        self.name.to_string()
    }
    async fn evaluate(&self, _req: &EvaluateRequest) -> Result<JudgmentResult, JudgeError> {
        self.barrier.wait().await;
        Ok(judgment(Verdict::Pass))
    }
}

fn group(judges: Vec<Arc<dyn Evaluator>>) -> JudgeGroup {
    JudgeGroup::new(
        judge_llm(FakeLlm::silent(), "openai/gpt-4o-mini"),
        judges,
        Arc::new(TokioSpawner::current()),
    )
}

// Catches: results collected in completion order instead of judge order.
#[tokio::test]
async fn the_group_collects_in_judge_order_not_completion_order() {
    let judges: Vec<Arc<dyn Evaluator>> = vec![
        Arc::new(Fixed { name: "accuracy", verdict: Verdict::Pass }),
        Arc::new(Fixed { name: "safety", verdict: Verdict::Fail }),
        Arc::new(Fixed { name: "tool_use", verdict: Verdict::Maybe }),
    ];
    let run =
        within("the group", group(judges).evaluate(Arc::new(ChatContext::empty()), None)).await;

    assert!(run.failures.is_empty());
    assert_eq!(
        run.result.judgments().map(|(n, j)| (n, j.verdict)).collect::<Vec<_>>(),
        vec![("accuracy", Verdict::Pass), ("safety", Verdict::Fail), ("tool_use", Verdict::Maybe)]
    );
}

// Catches: the fan-out becoming sequential.
#[tokio::test]
async fn judges_run_concurrently() {
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let judges: Vec<Arc<dyn Evaluator>> = vec![
        Arc::new(Barriered { name: "first", barrier: barrier.clone() }),
        Arc::new(Barriered { name: "second", barrier }),
    ];
    let run = within(
        "two judges that can only finish together",
        group(judges).evaluate(Arc::new(ChatContext::empty()), None),
    )
    .await;
    assert_eq!(run.result.len(), 2);
}

// Catches: the group not handing its own LLM to the judges.
#[tokio::test]
async fn the_group_supplies_its_llm_to_judges_constructed_without_one() {
    let llm = FakeLlm::verdict(r#"{"verdict": "pass", "reasoning": "ok"}"#);
    let judges: Vec<Arc<dyn Evaluator>> = vec![Arc::new(accuracy_judge(None))];
    let group = JudgeGroup::new(
        judge_llm(llm.clone(), "openai/gpt-4o-mini"),
        judges,
        Arc::new(TokioSpawner::current()),
    );

    let run = within(
        "the group",
        group.evaluate(Arc::new(ChatContext::new(vec![msg("m1", Role::User, "hi")])), None),
    )
    .await;

    assert!(run.failures.is_empty(), "{:?}", run.failures);
    assert_eq!(run.result.get("accuracy").unwrap().verdict, Verdict::Pass);
    assert!(llm.request().tools_host.is_none(), "a native judge LLM needs no host tool list");
}

// Catches: a provider failure converted into a verdict instead of reaching the caller intact.
#[tokio::test]
async fn a_provider_failure_is_not_a_verdict() {
    let judge = judge_llm(FakeLlm::broken(), "openai/gpt-4o-mini");
    let err = within("the broken judgment", evaluate_with_llm(&judge, "criteria"))
        .await
        .expect_err("the gateway refused");
    assert!(matches!(err, JudgeError::Provider(_)), "{err:?}");
}

// --- the intent judge (`ChatMessageAssert.judge`) ----------------------------------------------

// Catches: the intent judge losing its forced `check_intent` call, the default connection
// options, or the stream close.
#[tokio::test]
async fn the_intent_judge_forces_check_intent_with_the_default_options() {
    let llm = FakeLlm::verdict(r#"{"success": true, "reason": "greets"}"#);
    let judge = judge_llm(llm.clone(), "openai/gpt-4o-mini");
    let v = within("the intent judge", judge_intent(&judge, "greets the user", "hello!"))
        .await
        .unwrap();
    assert!(v.success);
    assert_eq!(v.reason, "greets");
    let seen = llm.request();
    assert_eq!(seen.tool_choice, Some(ToolChoice::Required));
    assert_eq!(seen.tools[0].name, "check_intent");
    assert_eq!(seen.conn_options.timeout, lk_agents_abi::ApiConnectOptions::default().timeout);
    assert_eq!(llm.closed.load(Ordering::SeqCst), 1);
}

// Catches: preconditions checked after the call, or a silent model reported as a failed intent.
#[tokio::test]
async fn the_intent_judge_refuses_empty_inputs_and_reports_a_missing_verdict() {
    let silent = FakeLlm::silent();
    let judge = judge_llm(silent.clone(), "openai/gpt-4o-mini");
    assert!(matches!(judge_intent(&judge, "x", "").await, Err(JudgeError::EmptyMessage)));
    assert!(matches!(judge_intent(&judge, "", "hello").await, Err(JudgeError::EmptyIntent)));
    assert_eq!(silent.closed.load(Ordering::SeqCst), 0, "no request for an empty input");
    let err = within("the silent judge", judge_intent(&judge, "x", "hello")).await;
    assert!(matches!(err, Err(JudgeError::NoArguments)));
}

// Catches: strict parsing where the reference is lax (json-repair, pydantic's lax bool).
#[test]
fn intent_arguments_parse_as_pydantic_validates_them() {
    assert_eq!(parse_intent_arguments(r#"{"success": "yes", "reason": "r"}"#).unwrap(), (true, "r".into()));
    assert_eq!(parse_intent_arguments(r#"{"success": 0, "reason": "r"}"#).unwrap(), (false, "r".into()));
    assert!(parse_intent_arguments(r#"{"success": "maybe", "reason": "r"}"#).is_err());
    assert!(parse_intent_arguments(r#"{"reason": "r"}"#).is_err());
}
