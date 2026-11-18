use lk_agents_abi::llm::realtime::RealtimeProvider;
use lk_agents_fuzz::{
    drive::{RealtimeSetup, Run, Setup, simulate},
    fakes::RealtimeDelays,
    input::{Input, Millis, Ref},
};

async fn advance(run: &mut Run, millis: u64) {
    crate::playout::apply_with_playout(run, &Input::Advance { ms: Millis(millis) }).await;
}

async fn reply(run: &mut Run) {
    run.apply(&Input::GenerateReply { user_input: None, instructions: None, allow: None }).await;
}

fn setup(provider: RealtimeProvider, delays: RealtimeDelays) -> Setup {
    Setup {
        realtime: Some(RealtimeSetup {
            provider,
            turn_detection: false,
            tool_calls: Some(0),
            delays,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// A late response and its expired request cannot steal the successor's handle or waiter.
#[test]
fn a_timed_out_reply_does_not_complete_or_fail_the_next_speech_handle() {
    for provider in [
        RealtimeProvider::Openai,
        RealtimeProvider::Xai,
        RealtimeProvider::Inference,
        RealtimeProvider::VoiceLive,
    ] {
        simulate(async {
            let mut run = Run::start(
                &setup(
                    provider,
                    RealtimeDelays {
                        first_created: Some(11_000),
                        created: Some(100),
                        ..Default::default()
                    },
                ),
                0,
                Default::default(),
            )
            .await;
            reply(&mut run).await;
            let first = run.speech_handle(Ref::new(0)).unwrap();
            advance(&mut run, 1_000).await;
            reply(&mut run).await;
            let second = run.speech_handle(Ref::new(1)).unwrap();
            assert_ne!(first.speech_id(), second.speech_id());
            advance(&mut run, 9_100).await;
            assert!(first.done(), "{provider}: the first request's bound settles its handle");
            assert!(first.exception().unwrap().is_some());
            assert!(!second.done(), "{provider}: queued reply does not inherit the first deadline");
            assert!(first.chat_items().is_empty());
            advance(&mut run, 15_000).await;
            assert!(second.done(), "{provider}: the conversation continues");
            assert!(second.exception().unwrap().is_none(), "{provider}: {:?}", second.exception());
            assert_eq!(second.chat_items().len(), 1);
            assert!(
                first.chat_items().is_empty(),
                "late output must not migrate to the expired handle"
            );
            first.wait_for_playout().await;
            second.wait_for_playout().await;
            reply(&mut run).await;
            let third = run.speech_handle(Ref::new(2)).unwrap();
            advance(&mut run, 15_000).await;
            assert!(third.done() && third.exception().unwrap().is_none());
            assert_ne!(second.chat_items()[0].id(), third.chat_items()[0].id());
            run.apply(&Input::Close { drain: true }).await;
            run.apply(&Input::AwaitExit {}).await;
            let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
            assert!(failures.is_empty(), "{provider}: {failures:#?}");
        });
    }
}

/// Completion clears only that request's timer. A later reply has a full independent deadline.
#[test]
fn the_previous_responses_old_deadline_cannot_expire_a_new_request() {
    simulate(async {
        let mut run = Run::start(
            &setup(
                RealtimeProvider::Openai,
                RealtimeDelays {
                    first_created: Some(100),
                    created: Some(8_000),
                    ..Default::default()
                },
            ),
            0,
            Default::default(),
        )
        .await;
        reply(&mut run).await;
        let first = run.speech_handle(Ref::new(0)).unwrap();
        advance(&mut run, 9_000).await;
        assert!(first.done() && first.exception().unwrap().is_none());
        reply(&mut run).await;
        let second = run.speech_handle(Ref::new(1)).unwrap();
        advance(&mut run, 2_000).await;
        assert!(
            !second.done(),
            "the old request's 10s mark passed without timing out the new request"
        );
        advance(&mut run, 15_000).await;
        assert!(second.done() && second.exception().unwrap().is_none());
        assert_eq!(first.chat_items().len(), 1);
        assert_eq!(second.chat_items().len(), 1);
        assert_ne!(first.chat_items()[0].id(), second.chat_items()[0].id());
        run.apply(&Input::Close { drain: true }).await;
        run.apply(&Input::AwaitExit {}).await;
    });
}

#[test]
fn cancelling_one_handle_or_its_waiter_cannot_cancel_the_queued_reply() {
    use std::{
        future::Future,
        task::{Context, Poll, Waker},
    };
    for interrupt in [false, true] {
        simulate(async {
            let mut run = Run::start(
                &setup(
                    RealtimeProvider::Openai,
                    RealtimeDelays {
                        first_created: Some(1_000),
                        created: Some(100),
                        ..Default::default()
                    },
                ),
                0,
                Default::default(),
            )
            .await;
            reply(&mut run).await;
            let first = run.speech_handle(Ref::new(0)).unwrap();
            reply(&mut run).await;
            let second = run.speech_handle(Ref::new(1)).unwrap();
            // Cancelling an await is distinct from asking to interrupt its speech.
            let mut waiter = Box::pin(first.wait_for_playout());
            assert!(matches!(
                waiter.as_mut().poll(&mut Context::from_waker(Waker::noop())),
                Poll::Pending
            ));
            drop(waiter);
            assert!(!first.interrupted());
            if interrupt {
                run.apply(&Input::InterruptSpeech { speech: Ref::new(0), force: true }).await;
            }
            advance(&mut run, 20_000).await;
            assert!(first.done() && second.done());
            assert_eq!(first.interrupted(), interrupt);
            assert!(!second.interrupted());
            assert!(second.exception().unwrap().is_none());
            assert_eq!(second.chat_items().len(), 1);
            first.wait_for_playout().await;
            second.wait_for_playout().await;
            run.apply(&Input::Close { drain: true }).await;
            run.apply(&Input::AwaitExit {}).await;
            let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
            assert!(failures.is_empty(), "{failures:#?}");
        });
    }
}

#[test]
fn tool_follow_up_responses_stay_on_the_original_handle_while_another_reply_is_queued() {
    use lk_agents_abi::ChatItem;
    use lk_agents_fuzz::{
        input::{AgentText, ToolAnswer},
        record::Kind,
    };
    simulate(async {
        let mut configuration = setup(RealtimeProvider::Openai, Default::default());
        configuration.tools = vec!["lookup".into()];
        configuration.max_tool_steps = Some(0);
        configuration.realtime.as_mut().unwrap().tool_calls = Some(100);
        let mut run = Run::start(&configuration, 0, Default::default()).await;
        reply(&mut run).await;
        let first = run.speech_handle(Ref::new(0)).unwrap();
        reply(&mut run).await;
        let second = run.speech_handle(Ref::new(1)).unwrap();
        let mut answered = std::collections::BTreeSet::new();
        for _ in 0..3 {
            advance(&mut run, 15_000).await;
            for tool in run.record.live().of(Kind::Tool) {
                if answered.insert(tool) {
                    run.apply(&Input::ToolAnswered {
                        tool: Ref::new(tool),
                        answer: ToolAnswer::Value,
                        text: AgentText("ok".into()),
                    })
                    .await;
                }
            }
        }
        assert_eq!(answered.len(), 2);
        advance(&mut run, 15_000).await;
        assert!(first.done() && second.done());
        assert!(first.exception().unwrap().is_none() && second.exception().unwrap().is_none());
        for handle in [&first, &second] {
            let items = handle.chat_items();
            assert_eq!(
                items.iter().filter(|i| matches!(i, ChatItem::Message(_))).count(),
                2,
                "{items:#?}"
            );
            let calls: Vec<_> = items
                .iter()
                .filter_map(|i| match i {
                    ChatItem::FunctionCall(c) => Some(c),
                    _ => None,
                })
                .collect();
            assert_eq!(calls.len(), 1);
            assert!(items.iter().any(
                |i| matches!(i, ChatItem::FunctionCallOutput(o) if o.call_id == calls[0].call_id)
            ));
        }
        let first_ids: std::collections::BTreeSet<_> =
            first.chat_items().into_iter().map(|i| i.id().to_owned()).collect();
        assert!(
            second.chat_items().iter().all(|item| !first_ids.contains(item.id())),
            "responses and tools cannot migrate between handles"
        );
        run.apply(&Input::Close { drain: true }).await;
        run.apply(&Input::AwaitExit {}).await;
        let (failures, _) = lk_agents_fuzz::check::check_all(&run.record);
        assert!(failures.is_empty(), "{failures:#?}");
    });
}
