use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_voice::speech_handle::{SpeechError, SpeechHandle};

#[tokio::test]
async fn a_failed_speech_finishes_its_waiters_and_keeps_its_first_error() {
    let speech = SpeechHandle::detached("speech".into(), true);
    assert_eq!(speech.exception(), Err(SpeechError::NotDoneYet));
    let first = ProviderError::Local(ApiError::timeout(Some("generation timed out".into())));
    speech.mark_done(Some(first.clone()));
    speech.wait_for_playout().await;
    speech.mark_done(None);
    speech.mark_done(Some(ProviderError::Local(ApiError::new("late failure"))));
    assert_eq!(speech.exception(), Ok(Some(first)));
}
