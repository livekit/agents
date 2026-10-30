use super::*;

fn tokens(input: u64, output: u64, cache_read: Option<u64>) -> Measurement {
    let tokens = TokenUsage {
        input_tokens: input,
        output_tokens: output,
        cache_read_tokens: cache_read,
        ..TokenUsage::default()
    };
    Measurement { tokens, ..Measurement::default() }
}

// usage.md "Implementation boundaries": cumulative reports replace, they never add up.
#[test]
fn repeated_cumulative_reports_count_once() {
    let mut l = Ledger::default();
    l.start(1, ModelKind::Llm, "openai", "gpt");
    l.report(1, tokens(10, 2, None));
    l.report(1, tokens(10, 5, None));
    l.finish(1, true);
    let u = {
        l.settle();
        l.snapshot()
    };
    assert_eq!((u.input_tokens(), u.output_tokens(), u.requests), (10, 5, 1));
    assert!(u.complete);
}

// usage.md "Counting and missing measurements": a failed request keeps its partial tokens and
// stays unmeasured after settlement.
#[test]
fn a_failed_request_keeps_its_partial_tokens_and_stays_unmeasured() {
    let mut l = Ledger::default();
    l.start(1, ModelKind::Llm, "openai", "gpt");
    l.report(1, tokens(7, 3, None));
    assert_eq!(l.snapshot().unmeasured_requests, 1, "awaiting its measurement");
    l.finish(1, false);
    let u = {
        l.settle();
        l.snapshot()
    };
    assert_eq!((u.total_tokens(), u.unmeasured_requests), (10, 1));
    assert!(u.settled && !u.complete);
}

// usage.md "TokenUsage": an aggregated breakdown is known only if every contribution supplied it.
#[test]
fn a_breakdown_one_request_lacks_is_unknown_in_the_aggregate() {
    let mut l = Ledger::default();
    for (id, cached) in [(1, Some(4)), (2, None)] {
        l.start(id, ModelKind::Llm, "openai", "gpt");
        l.report(id, tokens(10, 1, cached));
        l.finish(id, true);
    }
    let u = {
        l.settle();
        l.snapshot()
    };
    assert_eq!((u.tokens.input_tokens, u.tokens.cache_read_tokens), (20, None));
}

// usage.md "Scope and delegation": a turn counts every invocation it absorbed exactly once, and an
// invocation's request still in flight stays unmeasured.
#[test]
fn absorbed_invocations_count_once_and_close_their_pending_requests() {
    let mut turn = Ledger::default();
    for (id, done) in [(1, true), (2, false)] {
        let mut invocation = Ledger::default();
        invocation.start(id, ModelKind::Llm, "openai", "gpt");
        invocation.report(id, tokens(5, 5, None));
        if done {
            invocation.finish(id, true);
        }
        invocation.tool_started();
        turn.absorb(&invocation);
    }
    let u = {
        turn.settle();
        turn.snapshot()
    };
    assert_eq!(u.models.len(), 1, "one (type, provider, model) bucket");
    assert_eq!((u.requests, u.unmeasured_requests, u.tool_calls, u.total_tokens()), (2, 1, 2, 20));
}

// usage.md "ModelUsage": empty identities are `None`, and a text and a realtime LLM of the same
// identity share one entry.
#[test]
fn identities_are_normalized_and_shared_across_llm_sources() {
    let mut l = Ledger::default();
    l.start(1, ModelKind::Llm, "", "");
    l.finish(1, false);
    let mut rt = lk_agents_abi::metrics::testkit::realtime();
    rt.metadata = Some(Metadata { model_name: Some(String::new()), model_provider: None });
    l.observe(&AgentMetrics::RealtimeModel(rt));
    let u = {
        l.settle();
        l.snapshot()
    };
    let [ModelUsage::Llm(llm)] = u.models.as_slice() else { panic!("{:?}", u.models) };
    assert_eq!((llm.provider.as_deref(), llm.model.as_deref(), llm.requests), (None, None, 2));
}
