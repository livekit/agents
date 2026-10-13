//! A streamed say does not complete normally while the case still holds its stream open.
//!
//! Nothing records which speech a `say` became, so the link is rebuilt from the two resources the
//! call opens together: the stream the case feeds, and the call that carries the speech the
//! driver minted for it. Both take the call's own stamp, so both name the same `Say`.
use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::{AgentText, Input, Ref, SayInput},
    record::{Answer, Detail, Handle, Kind, Recording},
};

pub struct NoPrematureCompletion;

impl Rule for NoPrematureCompletion {
    const CONTRACT: Contract = Contract {
        id: "no_premature_completion",
        guarantee: "A streamed say does not complete normally while its input is still open.",
        examples: &[completed_after_end],
        counterexamples: &[completed_with_open_stream],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        for call in host.api_calls() {
            let Some((_, Answer::Speech(id))) = call.returned() else { continue };
            let Input::Say { input: SayInput::TextStream | SayInput::Audio, .. } = call.input()
            else {
                continue;
            };
            // The same `say` opened the stream the case feeds: one input, one stamp, both resources.
            let Some(stream) = call.stream() else { continue };
            let speech = host.speech(*id);
            if !speech.is_known() {
                continue;
            }

            // The speech is fed from a stream the case owns. Until the case ends it, more input may
            // come, so a normal completion while it was open dropped whatever came next. Judged on
            // the timeline -- the stream's close against the speech's completion -- not on whether the
            // stream happens to be closed by the time the rule reads the record.
            let normal = speech.completed_normally();
            let ended_first = host
                .stream_ended_at(stream)
                .is_some_and(|closed| speech.done_at().is_none_or(|done| closed <= done));
            check.expect(
                &speech,
                !(normal && !ended_first),
                format!("completed normally while stream {stream} was still open"),
            );
        }
    }
}

fn streamed_say() -> Input {
    Input::Say {
        text: AgentText("hello".into()),
        input: SayInput::TextStream,
        allow: None,
        to_ctx: true,
    }
}

/// The stream was ended, then the say completed.
fn completed_after_end() -> Recording {
    Authored::new()
        .sent_opening(
            streamed_say(),
            &[(Kind::Stream, Detail::Stream { audio: false }), (Kind::Answer, Detail::None)],
        )
        .result(0, Answer::Speech(1))
        .sent(Input::FeedEnd { stream: Ref::new(0) })
        .closed(Kind::Stream, 0)
        .speech(1, Handle::finished())
        .done()
}

/// A streamed say completed while its stream was never ended.
fn completed_with_open_stream() -> Recording {
    Authored::new()
        .sent_opening(
            streamed_say(),
            &[(Kind::Stream, Detail::Stream { audio: false }), (Kind::Answer, Detail::None)],
        )
        .result(0, Answer::Speech(1))
        .speech(1, Handle::finished())
        .done()
}
