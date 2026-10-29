use super::*;
use lk_agents_abi::errors::ApiError;

// Catches: a hold that is not counted, so one holder's release lifts another's, or a release that
// restores the value at the last hold instead of the first holder's.
#[test]
fn holds_are_counted_and_the_first_holders_value_comes_back() {
    let mut speech = SpeechState::new(true);
    speech.hold_interruptions().unwrap();
    speech.hold_interruptions().unwrap();
    assert!(!speech.allow_interruptions());
    speech.release_interruptions();
    assert!(!speech.allow_interruptions(), "one release lifted a hold another holder still has");
    speech.release_interruptions();
    assert!(speech.allow_interruptions());

    // A set during a hold is undone by the last release, as in the reference.
    let mut speech = SpeechState::new(false);
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
    let mut speech = SpeechState::new(true);
    assert_eq!(speech.interrupt(false), Ok(true));
    assert_eq!(speech.hold_interruptions(), Err(SpeechError::AlreadyInterrupted));
    assert!(speech.allow_interruptions());

    let mut speech = SpeechState::new(false);
    speech.hold_interruptions().unwrap();
    assert_eq!(speech.interrupt(true), Ok(true));
    speech.release_interruptions();
    assert!(!speech.allow_interruptions());
    // Interrupted already: a second interrupt, and the watchdog once fired, are no-ops.
    assert_eq!(speech.interrupt(false), Ok(false));
    assert!(speech.take_interruption_timeout());
    assert!(!speech.take_interruption_timeout());
}

// Catches: a generation lookup answering a step that was never authorized, or not telling an
// absent generation from an out-of-range step.
#[test]
fn generation_lookups_name_the_missing_step() {
    let mut speech = SpeechState::new(true);
    assert_eq!(speech.view().generation(0), Err(SpeechError::NoActiveGeneration));
    assert_eq!(speech.view().latest_generation(), Err(SpeechError::NoActiveGeneration));
    assert_eq!(speech.mark_generation_done(), Err(SpeechError::NoGenerationToMark));
    speech.authorize_generation();
    assert_eq!(speech.view().generation(3), Err(SpeechError::NoSuchGeneration { step: 3, len: 1 }));
    speech.mark_generation_done().unwrap();
    assert_eq!(speech.view().generations, [true]);
}

// Catches: a later completion replacing the error the speech failed with.
#[test]
fn a_failed_speech_keeps_its_first_error() {
    let mut speech = SpeechState::new(true);
    assert_eq!(speech.exception(), Err(SpeechError::NotDoneYet));
    let first = ProviderError::Local(ApiError::timeout(Some("generation timed out".into())));
    speech.mark_done(Some(first.clone()));
    speech.mark_done(None);
    speech.mark_done(Some(ProviderError::Local(ApiError::new("late failure"))));
    assert_eq!(speech.exception(), Ok(Some(first)));
    assert_eq!(speech.outcome(), crate::events::SpeechOutcome::Failed);
}

// Catches: a change the runtime never hears of, or the view published before the creation it
// belongs to.
#[test]
fn every_change_is_published_after_the_creation() {
    let mut speech = SpeechState::new(true);
    speech.mark_scheduled();
    let updates = speech.take_updates();
    assert!(matches!(updates[..], [SpeechUpdate::Created, SpeechUpdate::View(ref v)] if v.scheduled));
    assert!(speech.take_updates().is_empty(), "nothing changed since");
    speech.interrupt_refused();
    assert!(matches!(speech.take_updates()[..], [SpeechUpdate::InterruptRefused]));
}
