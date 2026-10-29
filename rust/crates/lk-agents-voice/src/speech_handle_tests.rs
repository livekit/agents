use super::*;
use crate::agent_session::Tx;
use lk_agents_abi::voice::speech_handle::SpeechHandle as HostSpeech;

fn created(allow: bool) -> SpeechHandle {
    SpeechHandle::new(SpeechId(1), allow, Tx::disconnected())
}

// A queued or refused setter must not change the accepted speech view.
#[test]
fn attached_options_wait_for_core_acceptance() {
    let speech = created(true);
    HostSpeech::set_allow_interruptions(&speech, false).unwrap();
    assert!(HostSpeech::allow_interruptions(&speech), "an undelivered setter changed the view");
    HostSpeech::hold_interruptions(&speech).unwrap();
    assert!(HostSpeech::allow_interruptions(&speech), "an undelivered hold changed the view");
}

// Catches: a hold that is not counted, so one holder's release lifts another's, or a release that
// restores the value at the last hold instead of the first holder's.
#[test]
fn holds_are_counted_and_the_first_holders_value_comes_back() {
    let speech = created(true);
    speech.hold_interruptions().unwrap();
    speech.hold_interruptions().unwrap();
    assert!(!speech.allow_interruptions());
    speech.release_interruptions();
    assert!(!speech.allow_interruptions(), "one release lifted a hold another holder still has");
    speech.release_interruptions();
    assert!(speech.allow_interruptions());

    // A set during a hold is undone by the last release, as in the reference.
    let speech = created(false);
    speech.hold_interruptions().unwrap();
    speech.set_allow_interruptions(true).unwrap();
    speech.release_interruptions();
    assert!(!speech.allow_interruptions());
}

// Catches: a hold on an interrupted speech counted anyway, so a later release would restore
// interruptions the reference's raise never took away; and a release after a forced interrupt
// re-enabling interruptions on a speech that no longer allows them.
#[test]
fn an_interrupted_speech_refuses_a_first_hold_and_keeps_its_value() {
    let speech = created(true);
    assert_eq!(speech.interrupt(false), Ok(true));
    assert_eq!(speech.hold_interruptions(), Err(SpeechError::AlreadyInterrupted));
    assert!(speech.allow_interruptions());

    let speech = created(false);
    speech.hold_interruptions().unwrap();
    assert_eq!(speech.interrupt(true), Ok(true));
    speech.release_interruptions();
    assert!(!speech.allow_interruptions());
    // Interrupted already: a second interrupt, and the watchdog once fired, are no-ops.
    assert_eq!(speech.interrupt(false), Ok(false));
}

// Catches: `wait_for_generation` answering a step that was never authorized, or not telling an
// absent generation from an out-of-range step.
#[tokio::test]
async fn generation_waits_name_the_missing_step() {
    let speech = created(true);
    assert_eq!(speech.wait_for_generation(0).await, Err(SpeechError::NoActiveGeneration));
    assert_eq!(speech.wait_for_latest_generation().await, Err(SpeechError::NoActiveGeneration));
    speech.authorize_generation();
    assert_eq!(
        speech.wait_for_generation(3).await,
        Err(SpeechError::NoSuchGeneration { step: 3, len: 1 })
    );
    speech.mark_generation_done().unwrap();
    speech.wait_for_generation(0).await.unwrap();
    HostSpeech::wait_for_generation(&speech, Some(0)).await.unwrap();
    HostSpeech::wait_for_generation(&speech, None).await.unwrap();
}

// Catches: a host wait on a handle no loop will ever create hanging instead of raising, and the
// two reasons (constructed directly, session ended) being confused.
#[tokio::test]
async fn host_waits_on_a_speech_never_created_raise_why() {
    let detached = SpeechHandle::detached("mine".into(), true);
    assert_eq!(HostSpeech::id(&detached), "mine");
    // Nothing to play: `wait_for_playout` returns rather than hang or raise.
    assert_eq!(HostSpeech::wait_for_playout(&detached, None).await, Ok(()));
    assert_eq!(
        HostSpeech::wait_until_created(&detached).await,
        Err(CreationError::Consumed.into())
    );
    assert_eq!(HostSpeech::hold_interruptions(&detached), Err(SpeechRefused::Detached.into()));
    assert_eq!(HostSpeech::release_interruptions(&detached), Err(SpeechRefused::Detached.into()));
    // A detached handle's setter is its own, with no loop to tell.
    HostSpeech::set_allow_interruptions(&detached, false).unwrap();
    assert!(!detached.allow_interruptions());
    assert!(matches!(HostSpeech::exception(&detached), Ok(None)));

    // A setter before the loop created the speech is the value it is created with.
    let pending = SpeechHandle::pending(Tx::disconnected(), SpeechId(7), true);
    HostSpeech::set_allow_interruptions(&*pending, false).unwrap();
    assert!(pending.allow_interruptions(), "a queued setter is not yet accepted");
    pending.create(SpeechId(7), true);
    assert!(pending.allow_interruptions(), "creation uses its accepted options");
    pending.set_allow_interruptions(false).unwrap();
    assert!(!pending.allow_interruptions());

    let lost = SpeechHandle::pending(Tx::disconnected(), SpeechId(8), true);
    HostSpeech::interrupt(&*lost, false).unwrap();
    assert!(HostSpeech::interrupted(&*lost));
    lost.answered(false);
    assert_eq!(HostSpeech::wait_for_interrupted(&*lost).await, Err(CreationError::Ended.into()));
    assert_eq!(
        HostSpeech::wait_for_generation(&*lost, None).await,
        Err(CreationError::Ended.into())
    );
    tokio::task::yield_now().await;
}

// Catches: the attached host calls refusing on a created speech, or a hold taken through the host
// not reaching the speech.
#[tokio::test]
async fn host_calls_on_a_created_speech_reach_it() {
    let speech = created(true);
    HostSpeech::hold_interruptions(&speech).unwrap();
    assert!(speech.allow_interruptions(), "the hold is queued");
    speech.hold_interruptions().unwrap();
    assert!(!speech.allow_interruptions());
    HostSpeech::release_interruptions(&speech).unwrap();
    assert!(!speech.allow_interruptions(), "the release is queued");
    speech.release_interruptions();
    assert!(speech.allow_interruptions());
    assert_eq!(
        HostSpeech::exception(&speech),
        Err(SpeechRefused::Speech(SpeechError::NotDoneYet).into())
    );

    speech.interrupt(false).unwrap();
    HostSpeech::wait_for_interrupted(&speech).await.unwrap();
    assert_eq!(
        HostSpeech::set_allow_interruptions(&speech, false),
        Err(SpeechError::AlreadyInterrupted.into())
    );
    speech.mark_done(None);
    HostSpeech::wait_for_playout(&speech, None).await.unwrap();
    assert!(matches!(HostSpeech::exception(&speech), Ok(None)));
}

// Catches: two handles of one speech comparing unequal, or two pending handles comparing equal.
#[test]
fn a_speech_is_itself_across_handles() {
    assert_eq!(created(true), created(false));
    let (a, b) =
        (SpeechHandle::detached(String::new(), true), SpeechHandle::detached(String::new(), true));
    assert_ne!(a, b);
    assert_eq!(a, a);
}

// Catches: a host `interrupt()` that passes the handle's check but that the loop refuses leaving
// the request behind (the speech read `interrupted` forever and was reported interrupted when it
// played out), the check reading a stale `allow_interruptions` right after the setter, and a
// handle with no id until the loop created it.
#[tokio::test]
async fn a_refused_host_interrupt_leaves_the_speech_uninterrupted() {
    let pending = SpeechHandle::pending(Tx::disconnected(), SpeechId(3), false);
    assert_eq!(HostSpeech::id(&*pending), "speech_3", "the id exists as the call returns");
    assert!(HostSpeech::interrupt(&*pending, false).is_err(), "disallowed at the call");

    let speech = Arc::new(created(true));
    HostSpeech::set_allow_interruptions(&*speech, false).unwrap();
    assert!(HostSpeech::allow_interruptions(&*speech), "the setter is queued");
    speech.set_allow_interruptions(false).unwrap();
    assert!(!HostSpeech::allow_interruptions(&*speech), "the accepted setter is visible");
    assert!(HostSpeech::interrupt(&*speech, false).is_err());

    let speech = Arc::new(created(true));
    HostSpeech::interrupt(&*speech, false).unwrap();
    assert!(HostSpeech::interrupted(&*speech));
    speech.interrupt_refused();
    assert!(!HostSpeech::interrupted(&*speech));
    speech.mark_done(None);
    assert_eq!(speech.outcome(), crate::events::SpeechOutcome::Completed);
}
