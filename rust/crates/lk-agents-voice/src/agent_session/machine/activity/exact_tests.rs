use super::*;
use crate::agent_session::effects::VoiceEffect;
use crate::agent_session::machine::activity::scheduling::SpeechId;
use crate::agent_session::{Answer, RequestId};
use crate::speech_handle::SpeechUpdate;
use lk_agents_abi::resources::{ResourceId, TtsRef};

#[test]
fn exact_realtime_output_without_tts_is_refused_before_generation_or_delivered_history() {
    let mut driver = crate::agent_session::tests::realtime_driver();
    driver.take_outbox();
    let before: Vec<_> = driver.session.chat_ctx.history().cloned().collect();
    let call = SpeechCall { id: RequestId(31), speech: SpeechId(91) };
    driver.say_exact(Ts::ZERO, "Reference ABC.".into(), Some(call.clone()));
    let effects = driver.settle_step();
    assert!(
        !effects.iter().any(|effect| matches!(
            effect,
            VoiceEffect::Realtime { .. } | VoiceEffect::StartSpeech { .. }
        )),
        "a generative request already breaks the exact-output promise"
    );
    let view = effects
        .iter()
        .find_map(|effect| match effect {
            VoiceEffect::Speech(id, SpeechUpdate::View(view)) if *id == call.speech => Some(view),
            _ => None,
        })
        .expect("the existing receipt settles with an explicit refusal");
    let error = view.error().unwrap();
    assert_eq!(error.message(), EXACT_TTS_REQUIRED);
    assert!(!error.cancelled(), "output refusal cannot cancel a remote business task");
    assert!(view.done());
    assert!(
        effects.iter().any(
            |effect| matches!(effect, VoiceEffect::Answer(id, Answer::Speech) if *id == call.id)
        )
    );
    assert_eq!(driver.session.chat_ctx.history().cloned().collect::<Vec<_>>(), before);
}

#[test]
fn exact_realtime_output_with_a_registered_tts_leg_uses_the_pipeline() {
    let mut driver = crate::agent_session::tests::realtime_driver();
    let tts = ResourceId(41);
    driver.session.tts = Some(TtsRef {
        adapter: false,
        id: tts,
        provider: Default::default(),
        capabilities: Default::default(),
        num_channels: 1,
    });
    driver.take_outbox();
    driver.say_exact(
        Ts::ZERO,
        "Reference ABC.".into(),
        Some(SpeechCall { id: RequestId(32), speech: SpeechId(92) }),
    );
    let effects = driver.settle_step();
    assert!(!effects.iter().any(|effect| matches!(effect, VoiceEffect::Realtime { .. })));
    assert!(effects.iter().any(
        |effect| matches!(effect, VoiceEffect::StartSpeech { tts: Some(id), .. } if *id == tts)
    ));
}

#[test]
fn local_verbatim_results_and_readbacks_refuse_generation_without_inventing_presentation() {
    use crate::tool_executor::Delivery;
    for delivery in [
        Delivery::Say("Reference ABC.".into()),
        Delivery::Ask {
            question: lk_agents_abi::agent::run::RunQuestion {
                question: "Approve charging 20 dollars?".into(),
                verbatim: true,
            },
            call_id: "canonical-charge".into(),
        },
    ] {
        let readback = matches!(&delivery, Delivery::Ask { .. });
        let mut driver = crate::agent_session::tests::realtime_driver();
        driver.take_outbox();
        let before: Vec<_> = driver.session.chat_ctx.history().cloned().collect();
        driver.deliver_outcome(Ts::ZERO, SpeechId(7), delivery);
        let effects = driver.settle_step();
        assert!(!effects.iter().any(|effect| matches!(
            effect,
            VoiceEffect::Realtime { .. }
                | VoiceEffect::StartSpeech { .. }
                | VoiceEffect::InputPresented { .. }
        )));
        assert_eq!(driver.session.chat_ctx.history().cloned().collect::<Vec<_>>(), before);
        if readback {
            assert!(effects.iter().any(|effect| matches!(effect, VoiceEffect::RetainLocalInput { request_id, call_id, source } if request_id == "canonical-charge" && call_id == "canonical-charge" && source.raw_text_content().as_deref() == Some("Approve charging 20 dollars?"))), "refusal keeps the exact original question pending");
        }
    }
}
