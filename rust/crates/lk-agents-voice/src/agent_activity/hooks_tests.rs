//! Hook terminal ownership and the deferred boundary needed by no-host close.

use super::*;
use crate::agent_session::Close;
use crate::agent_session::effects::VoiceEffect;
use crate::trace::Step;

fn driver(hosted: bool) -> VoiceDriver {
    let mut driver = crate::agent_session::tests::driver();
    driver.journalled = true;
    driver.activity.hook_bound = hosted.then_some(Duration::from_secs(1));
    driver
}

fn exit(driver: &mut VoiceDriver) -> HookId {
    let hook = driver.activity.claim_exit().hook().unwrap();
    driver.issue_claimed_exits();
    hook
}

fn finished(driver: &mut VoiceDriver, hook: HookId) -> Vec<HookEnd> {
    driver
        .take_outbox()
        .into_iter()
        .filter_map(|planned| match planned {
            VoiceEffect::Journal(step) => match *step {
                Step::Handoff(HandoffStep::HookFinished { hook: id, how, .. }) if id == hook => {
                    Some(how)
                }
                _ => None,
            },
            _ => None,
        })
        .collect()
}

#[test]
fn no_host_close_installs_its_wait_before_common_completion_dispatch() {
    let mut driver = driver(false);
    driver.begin_closing(Ts::ZERO, Close::default());
    let hook = driver.activity.exiting_hook().expect("close claimed the exit");
    assert!(matches!(
        driver.session.close_phase(),
        Some(crate::agent_session::ClosePhase::Exiting { hook: id }) if id == hook
    ));
    assert!(driver.activity.hooks.pending.contains_key(&hook));
    assert!(
        driver
            .pending_callbacks()
            .iter()
            .any(|s| matches!(s, VoiceCallback::HookFinished(id) if *id == hook))
    );
    driver.settle_internal();
    let effects = driver.take_outbox();
    assert!(driver.stopped(), "local completion must not strand close in its exit phase");
    assert!(!effects.iter().any(|e| matches!(e, VoiceEffect::Host(VoiceHostEvent::Hook(_)))));
    let ends: Vec<_> = effects
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::Journal(step) => match &**step {
                Step::Handoff(HandoffStep::HookFinished { hook: id, how, .. }) if *id == hook => {
                    Some(*how)
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(ends, [HookEnd::NoHost]);
}

#[test]
fn exit_reply_and_timeout_claim_only_one_deferred_completion() {
    for timeout_first in [false, true] {
        let mut driver = driver(true);
        let hook = exit(&mut driver);
        if timeout_first {
            driver.hook_timed_out(Ts::ZERO, hook);
            driver.on_hook_done(Ts::ZERO, hook, HookOutcome::Returned);
        } else {
            driver.on_hook_done(Ts::ZERO, hook, HookOutcome::Returned);
            driver.hook_timed_out(Ts::ZERO, hook);
        }
        assert_eq!(
            driver
                .pending_callbacks()
                .iter()
                .filter(|s| matches!(s, VoiceCallback::HookFinished(id) if *id == hook))
                .count(),
            1
        );
        assert!(driver.activity.hooks.pending.contains_key(&hook));
        driver.dispatch_hook_completion(Ts::ZERO, hook);
        driver.dispatch_hook_completion(Ts::ZERO, hook);
        driver.on_hook_done(Ts::ZERO, hook, HookOutcome::Returned);
        assert!(driver.activity.exiting_hook().is_none());
        assert_eq!(
            finished(&mut driver, hook),
            [if timeout_first { HookEnd::TimedOut } else { HookEnd::Returned { edits: 0 } }]
        );
    }
}

#[test]
fn abandonment_preserves_accepted_cause_but_suppresses_queued_exit_continuation() {
    let mut driver = driver(true);
    let hook = exit(&mut driver);
    driver.on_hook_done(Ts::ZERO, hook, HookOutcome::Returned);
    driver.abandon_hooks();
    driver.dispatch_hook_completion(Ts::ZERO, hook);
    driver.hook_timed_out(Ts::ZERO, hook);
    assert_eq!(driver.activity.exiting_hook(), Some(hook), "no exit continuation was delivered");
    assert_eq!(finished(&mut driver, hook), [HookEnd::Returned { edits: 0 }]);
}

#[test]
fn abandonment_before_result_is_the_single_terminal_cause() {
    let mut driver = driver(true);
    let hook = exit(&mut driver);
    driver.abandon_hooks();
    driver.on_hook_done(Ts::ZERO, hook, HookOutcome::Returned);
    driver.hook_timed_out(Ts::ZERO, hook);
    driver.dispatch_hook_completion(Ts::ZERO, hook);
    assert_eq!(finished(&mut driver, hook), [HookEnd::Abandoned]);
}

#[test]
fn no_host_and_host_enter_share_accounting_without_an_activation_barrier() {
    for hosted in [false, true] {
        let mut driver = driver(hosted);
        driver.issue_on_enter(Ts::ZERO);
        let hook = HookId(1);
        if hosted {
            assert!(driver.activity.hooks.pending.contains_key(&hook));
            driver.on_hook_done(Ts::ZERO, hook, HookOutcome::Returned);
        }
        assert!(!driver.activity.hooks.pending.contains_key(&hook));
        assert!(
            !driver.pending_callbacks().iter().any(|s| matches!(s, VoiceCallback::HookFinished(_)))
        );
        assert_eq!(
            finished(&mut driver, hook),
            [if hosted { HookEnd::Returned { edits: 0 } } else { HookEnd::NoHost }]
        );
    }
}

#[test]
fn non_exit_completion_preserves_inline_order_ahead_of_queued_callbacks() {
    let mut driver = driver(true);
    driver.issue_on_enter(Ts::ZERO);
    driver.defer_callback(VoiceCallback::AgentStays);
    driver.on_hook_done(Ts::ZERO, HookId(1), HookOutcome::Returned);
    assert!(driver.activity.hooks.pending.is_empty());
    assert!(matches!(driver.pending_callbacks().front(), Some(VoiceCallback::AgentStays)));
    assert_eq!(
        driver.pending_callbacks().len(),
        1,
        "no new deferred completion changes inline ordering"
    );
}

#[test]
fn no_host_filler_source_refuses_once_instead_of_rescheduling_a_decline() {
    use crate::agent_activity::scheduling::SpeechId;
    use crate::agent_session::FillerSource;
    use crate::filler_scheduler::{FillerId, FillerScheduler};
    use crate::tool_executor::FillerRun;
    use crate::trace::{FillerEnd, GenerationStep};

    let mut driver = driver(false);
    let hook = driver.activity.hooks.mint();
    let filler = FillerId(1);
    driver.activity.tools.fillers.insert(
        filler,
        FillerRun {
            sched: FillerScheduler::new(0.0, Some(0.0), None).unwrap(),
            call_id: "call".into(),
            speech: SpeechId(1),
            source: FillerSource::Host(None),
            asking: Some(hook),
            armed: None,
            fired_at: None,
        },
    );
    driver.issue_hook(
        Ts::ZERO,
        HookCall::FillerSource { id: hook, filler, step: 0, source: None },
        Some(Duration::from_secs(1)),
    );
    driver.on_hook_done(Ts::ZERO, hook, HookOutcome::Returned);
    assert!(!driver.activity.tools.fillers.contains_key(&filler));
    assert!(driver.activity.hooks.pending.is_empty());
    let stops: Vec<_> = driver
        .take_outbox()
        .into_iter()
        .filter_map(|planned| match planned {
            VoiceEffect::Journal(step) => match *step {
                Step::Generation(GenerationStep::FillerStopped { filler: id, cause })
                    if id == filler =>
                {
                    Some(cause)
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(stops, [FillerEnd::Refused]);
}
