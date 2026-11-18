use super::*;
use crate::room_io::FixedRoom;
use crate::telemetry;
use crate::voice::session::tests::{close_session, observing_opened_context, unobserved};
use crate::voice::session::{SessionRunner, VoiceHostEvent};
use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_voice::agent_session::commands::{Command, HookOutcome};
use parking_lot::Mutex;

#[tokio::test(start_paused = true)]
async fn provider_context_uses_the_sessions_recording_clock_and_keeps_its_job() {
    let clock = Arc::new(VirtualClock::new());
    let room = FixedRoom::new(None, Timebase::from_virtual(clock.clone()), None);
    let api = unobserved(room, None);
    let process_clock = Arc::new(VirtualClock::new());
    let tracer = telemetry::Tracer::disabled(process_clock.clone());
    let trace =
        tracer.root().with_job(Some(JobKey("AJ_clock".into()))).with_conversation(Some("AT_clock"));
    let session = telemetry::reserve::<telemetry::AgentSession>(&trace);
    let start = telemetry::reserve::<telemetry::SessionStart>(
        &trace.clone().under::<telemetry::AgentSession>(session.context),
    );
    api.trace_under(
        trace,
        Default::default(),
        telemetry::Opening { session, start, activity: None },
    );
    process_clock.advance(Duration::from_secs(10));
    clock.advance(Duration::from_secs(1));
    let observed = Arc::new(Mutex::new(None));
    api.open(
        observing_opened_context(&api, observed.clone()),
        None,
        |mut runner: SessionRunner| {
            Box::pin(async move {
                while let Some(event) = runner.next_host_event().await {
                    if let VoiceHostEvent::Hook(call) = event {
                        runner
                            .api()
                            .send(Command::HookDone {
                                id: call.id(),
                                outcome: HookOutcome::Returned,
                            })
                            .unwrap();
                    }
                }
            })
        },
    )
    .unwrap()
    .await
    .unwrap();
    let cx = observed.lock().take().expect("the production executor opened its provider legs");
    let expected = api.time().wall().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let actual = u128::from(cx.now_nanos());
    assert_eq!(cx.job(), Some(&JobKey("AJ_clock".into())));
    assert_eq!(cx.conversation(), Some("AT_clock"));
    close_session(&api).await;
    assert_eq!(actual, expected, "provider spans must share the recording's clock");
}
