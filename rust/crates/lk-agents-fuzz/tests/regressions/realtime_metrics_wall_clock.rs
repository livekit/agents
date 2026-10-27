//! A realtime session's metrics are stamped on the wall clock, `time.time()`'s.
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_fuzz::drive::{Case, RealtimeSetup, Run, Setup, run, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, RtEvent};
use lk_agents_voice::events::SessionEvent;

// Catches: the native realtime session stamped its `RealtimeModelMetrics` (each response's, and
// the connection's) and its transcription `STTMetrics` with the seconds its clock had run (a
// `RealClock` counts from the model's creation), and the items the server created with the epoch
// plus that: a response reported at `timestamp=5.1` on a call in 2026. The reference stamps them
// with `time.time()` (`items_carry_their_payload`).
#[test]
fn realtime_metrics_are_stamped_on_the_wall_clock() {
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
        ..Setup::default()
    };
    let inputs = vec![
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::GenerationDone },
        Input::Advance { ms: Millis(3_000) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(500) },
    ];
    let outcome = run(Case { setup: setup.clone(), inputs: inputs.clone(), ..Case::default() });
    let failed: Vec<_> = outcome.failures.iter().map(|f| f.contract.as_str()).collect();
    assert!(!failed.contains(&"items_carry_their_payload"), "{:#?}", outcome.failures);
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let stamps: Vec<f64> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, ev)| match ev {
            SessionEvent::MetricsCollected { metrics, .. } => match &**metrics {
                AgentMetrics::RealtimeModel(m) => Some(m.timestamp),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert!(!stamps.is_empty(), "no realtime metrics");
    assert!(stamps.iter().all(|t| *t > 1.7e9), "{stamps:?}");
}
