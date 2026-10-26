//! AMD's clock starts when the callee answers in the room: a ringing SIP call is not timed, and a
//! callee whose audio never comes, or who hangs up first, is `participant_missing`.
use lk_agents_abi::voice::amd::{AmdCategory, AmdReason};
use lk_agents_fuzz::drive::{AmdAnswer, AmdSetup, RoomSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AmdInput, Input, Millis, RoomIn};
use lk_agents_voice::events::SessionEvent;

/// AMD's verdict after `inputs`, if any.
fn verdict(answer: AmdAnswer, inputs: &[Input]) -> Option<(AmdCategory, AmdReason)> {
    let setup = Setup {
        amd: Some(AmdSetup { answer, ..AmdSetup::default() }),
        room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false }),
        ..Setup::default()
    };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in inputs {
            run.apply(input).await;
        }
        run
    });
    run.record.host_view().events().into_iter().find_map(|(_, ev)| match ev {
        SessionEvent::AmdResult { verdict } => Some((verdict.category, verdict.reason)),
        _ => None,
    })
}

fn advance(ms: u64) -> Input {
    Input::Advance { ms: Millis(ms) }
}

const NO_SPEECH: Option<(AmdCategory, AmdReason)> =
    Some((AmdCategory::Uncertain, AmdReason::NoSpeechTimeout));
const MISSING: Option<(AmdCategory, AmdReason)> =
    Some((AmdCategory::Uncertain, AmdReason::ParticipantMissing));

// Catches: listening before the callee's audio is there, or the no-speech timer not armed once
// it is: their track opens listening at once.
#[test]
fn the_callees_audio_starts_listening() {
    assert_eq!(verdict(AmdAnswer::Track, &[advance(9_900)]), None);
    assert_eq!(verdict(AmdAnswer::Track, &[advance(10_100)]), NO_SPEECH);
}

// Catches: a wait for the track with no bound (`_TRACK_PUBLICATION_TIMEOUT`, 5 s), or the
// callee joining in time not heard.
#[test]
fn a_callee_whose_audio_never_comes_is_missing() {
    assert_eq!(verdict(AmdAnswer::Missing, &[advance(4_900)]), None);
    assert_eq!(verdict(AmdAnswer::Missing, &[advance(5_100)]), MISSING);
    let joined = [advance(1_000), Input::Room { event: RoomIn::Linked }, advance(10_100)];
    assert_eq!(verdict(AmdAnswer::Missing, &joined), NO_SPEECH);
}

// Catches: a SIP call timed while it rings (its track is there before the callee answers), or
// its answer not opening listening.
#[test]
fn a_ringing_sip_call_is_not_timed() {
    assert_eq!(verdict(AmdAnswer::Sip, &[advance(30_000)]), None);
    let answered = [advance(30_000), Input::Amd { input: AmdInput::Answer }, advance(10_100)];
    assert_eq!(verdict(AmdAnswer::Sip, &answered), NO_SPEECH);
}

// Catches: a SIP wait that outlives the call it waits on.
#[test]
fn a_sip_call_that_ends_unanswered_is_missing() {
    let hung_up = [advance(1_000), Input::Room { event: RoomIn::Unlinked }, advance(20)];
    assert_eq!(verdict(AmdAnswer::Sip, &hung_up), MISSING);
}
