use super::*;
use lk_agents_abi::voice::speech_handle::SpeechHandle as HostSpeech;
use lk_agents_voice::speech_handle::SpeechState;

/// The core's speech and the session's registry its updates reach.
struct Speech {
    state: SpeechState,
    speeches: Speeches,
    id: SpeechId,
}

impl Speech {
    fn created(allow: bool) -> Self {
        let mut speech = Speech {
            state: SpeechState::new(allow),
            speeches: Speeches::default(),
            id: SpeechId(1),
        };
        speech.publish();
        speech
    }

    /// Over the handle a host call returned for `id`.
    fn over(pending: &Arc<SpeechHandle>, id: SpeechId, allow: bool) -> Self {
        let speech = Speech { state: SpeechState::new(allow), speeches: Speeches::default(), id };
        speech.speeches.register(Arc::clone(pending));
        speech
    }

    /// What the core accepted reaches the handle.
    fn publish(&mut self) {
        for update in self.state.take_updates() {
            self.speeches.apply(self.id, update, &Tx::disconnected());
        }
    }

    fn handle(&self) -> Arc<SpeechHandle> {
        self.speeches.get(self.id).expect("published")
    }
}

// Catches: a queued setter or hold changing the speech's view before the core accepted it.
#[test]
fn attached_options_wait_for_core_acceptance() {
    let mut speech = Speech::created(true);
    let handle = speech.handle();
    HostSpeech::set_allow_interruptions(&*handle, false).unwrap();
    assert!(HostSpeech::allow_interruptions(&*handle), "an undelivered setter changed the view");
    HostSpeech::hold_interruptions(&*handle).unwrap();
    assert!(HostSpeech::allow_interruptions(&*handle), "an undelivered hold changed the view");
    speech.state.hold_interruptions().unwrap();
    speech.publish();
    assert!(!HostSpeech::allow_interruptions(&*handle), "the accepted hold is visible");
}

// Catches: `wait_for_generation` answering a step that was never authorized, or not telling an
// absent generation from an out-of-range step.
#[tokio::test]
async fn generation_waits_name_the_missing_step() {
    let mut speech = Speech::created(true);
    let handle = speech.handle();
    assert_eq!(handle.wait_for_generation(0).await, Err(SpeechError::NoActiveGeneration));
    assert_eq!(handle.wait_for_latest_generation().await, Err(SpeechError::NoActiveGeneration));
    speech.state.authorize_generation();
    speech.publish();
    assert!(handle.step_authorized(1) && !handle.step_authorized(2));
    assert_eq!(
        handle.wait_for_generation(3).await,
        Err(SpeechError::NoSuchGeneration { step: 3, len: 1 })
    );
    speech.state.mark_generation_done().unwrap();
    speech.publish();
    handle.wait_for_generation(0).await.unwrap();
    HostSpeech::wait_for_generation(&*handle, Some(0)).await.unwrap();
    HostSpeech::wait_for_generation(&*handle, None).await.unwrap();
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
        Err(refused(CreationError::Consumed))
    );
    assert_eq!(HostSpeech::hold_interruptions(&detached), Err(refused(SpeechRefused::Detached)));
    assert_eq!(HostSpeech::release_interruptions(&detached), Err(refused(SpeechRefused::Detached)));
    // A detached handle's setter is its own, with no loop to tell.
    HostSpeech::set_allow_interruptions(&detached, false).unwrap();
    assert!(!detached.allow_interruptions());
    assert!(matches!(HostSpeech::exception(&detached), Ok(None)));

    // A setter before the loop created the speech is the value it is created with.
    let pending = SpeechHandle::pending(Requests::disconnected(), SpeechId(7), true);
    HostSpeech::set_allow_interruptions(&*pending, false).unwrap();
    assert!(pending.allow_interruptions(), "a queued setter is not yet accepted");
    let mut speech = Speech::over(&pending, SpeechId(7), true);
    speech.publish();
    assert_eq!(pending.created(), Some(SpeechId(7)), "created over the host's handle");
    speech.state.set_allow_interruptions(false).unwrap();
    speech.publish();
    assert!(!pending.allow_interruptions());

    let lost = SpeechHandle::pending(Requests::disconnected(), SpeechId(8), true);
    HostSpeech::interrupt(&*lost, false).unwrap();
    assert!(HostSpeech::interrupted(&*lost));
    lost.answered(false);
    assert_eq!(HostSpeech::wait_for_interrupted(&*lost).await, Err(refused(CreationError::Ended)));
    assert_eq!(
        HostSpeech::wait_for_generation(&*lost, None).await,
        Err(refused(CreationError::Ended))
    );
}

// Catches: two handles of one speech comparing unequal, or two pending handles comparing equal.
#[test]
fn a_speech_is_itself_across_handles() {
    assert_eq!(*Speech::created(true).handle(), *Speech::created(false).handle());
    let (a, b) =
        (SpeechHandle::detached(String::new(), true), SpeechHandle::detached(String::new(), true));
    assert_ne!(a, b);
    assert_eq!(a, a);
}

// Catches: a host `interrupt()` the loop refused leaving the request behind (the speech read
// `interrupted` forever), the check reading a stale `allow_interruptions` right after the setter,
// a handle with no id until the loop created it, a host view going back to `False` once the
// speech ended (1.x's `_interrupt_fut` is sticky), and a request a speech created done never
// heard reading as an interruption the loop never made.
#[test]
fn the_host_reads_the_loops_verdict_on_its_interrupt() {
    let pending = SpeechHandle::pending(Requests::disconnected(), SpeechId(3), false);
    assert_eq!(HostSpeech::id(&*pending), "speech_3", "the id exists as the call returns");
    assert!(HostSpeech::interrupt(&*pending, false).is_err(), "disallowed at the call");

    let mut speech = Speech::created(true);
    let handle = speech.handle();
    HostSpeech::set_allow_interruptions(&*handle, false).unwrap();
    assert!(HostSpeech::allow_interruptions(&*handle), "the setter is queued");
    speech.state.set_allow_interruptions(false).unwrap();
    speech.publish();
    assert!(!HostSpeech::allow_interruptions(&*handle), "the accepted setter is visible");
    assert!(HostSpeech::interrupt(&*handle, false).is_err());

    let mut speech = Speech::created(true);
    let handle = speech.handle();
    HostSpeech::interrupt(&*handle, false).unwrap();
    assert!(HostSpeech::interrupted(&*handle), "requested, from this tick");
    speech.state.interrupt_refused();
    speech.publish();
    assert!(!HostSpeech::interrupted(&*handle), "the refusal dropped the request");

    // Asked before the speech's end: the host's view stays interrupted, whatever comes next.
    let mut speech = Speech::created(true);
    let handle = speech.handle();
    HostSpeech::interrupt(&*handle, false).unwrap();
    speech.state.mark_done(None);
    speech.publish();
    assert!(HostSpeech::interrupted(&*handle), "the host's interrupted went back to False");

    // Asked of a speech the loop created done (refused, or answered as the session closed): it
    // never ran, so the request lands nowhere and the host reads the core's verdict.
    let pending = SpeechHandle::pending(Requests::disconnected(), SpeechId(4), true);
    HostSpeech::interrupt(&*pending, false).unwrap();
    let mut speech = Speech::over(&pending, SpeechId(4), true);
    speech.state.mark_done(None);
    speech.publish();
    pending.answered(true);
    assert!(!HostSpeech::interrupted(&*pending), "a request the speech never heard was recorded");
}
