#[cfg(unix)]
mod support {
    //! Execute context disposal through an admission pool like a live session's.

    use std::sync::OnceLock;
    use std::time::Duration;

    use lk_agents_inference::realtime::RealtimeMachine;
    use lk_agents_media::pool::{JobClass, MediaPool};

    pub fn thread_cpu() -> Duration {
        let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: the OS writes one initialized timespec through a valid exclusive pointer.
        assert_eq!(unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) }, 0);
        Duration::new(time.tv_sec.try_into().unwrap(), time.tv_nsec.try_into().unwrap())
    }

    #[derive(Default)]
    pub struct RetirementStats {
        pub jobs: usize,
        pub cpu: Duration,
        pub maximum: Duration,
    }

    struct Executor {
        handle: tokio::runtime::Handle,
        media_pool: MediaPool,
    }

    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

    fn executor() -> &'static Executor {
        static EXECUTOR: OnceLock<Executor> = OnceLock::new();
        EXECUTOR.get_or_init(|| Executor {
            handle: RUNTIME
                .get_or_init(|| tokio::runtime::Runtime::new().expect("runtime"))
                .handle()
                .clone(),
            media_pool: MediaPool::new(Default::default()).expect("worker pool"),
        })
    }

    pub fn execute_retirements(
        machine: &mut RealtimeMachine,
        mut controls: impl FnMut(&mut RealtimeMachine),
    ) -> RetirementStats {
        let executor = executor();
        let mut stats = RetirementStats::default();
        while machine.has_context_retirement() {
            // Capacity refusal leaves every destructor-bearing owner in the machine.
            let permit = loop {
                match executor.media_pool.try_admit(JobClass::Formatting) {
                    Ok(permit) => break permit,
                    Err(_) => executor
                        .handle
                        .block_on(executor.media_pool.wait_for_capacity(JobClass::Formatting)),
                }
            };
            let retirement = machine.take_context_retirement().expect("admitted retired owner");
            let core_thread = std::thread::current().id();
            let receipt = executor.media_pool.spawn_admitted(permit, move || {
                assert_ne!(
                    std::thread::current().id(),
                    core_thread,
                    "cleanup ran on the core thread"
                );
                let start = thread_cpu();
                drop(retirement);
                thread_cpu().saturating_sub(start)
            });
            controls(machine);
            // Admission, dispatch, controls and awaiting settlement are outside timed core calls.
            let cpu = executor.handle.block_on(receipt).expect("retirement worker settled");
            stats.jobs += 1;
            stats.cpu += cpu;
            stats.maximum = stats.maximum.max(cpu);
        }
        stats
    }
}

#[cfg(unix)]
#[path = "realtime/budget_tests.rs"]
mod context_step_budget;
#[cfg(unix)]
#[path = "realtime/history_tests.rs"]
mod incremental_context;
mod live {
    //! Real-peer realtime protocol acceptance. Run with `cargo test -p lk-agents-runtime --test realtime_live -- --ignored`.

    use std::sync::Arc;
    use std::time::Duration;

    use lk_agents_abi::llm::realtime::GenerationDone;
    use lk_agents_inference::realtime::openai::{
        AudioDelta, Event, GenerationCreated, OPENAI, OpenAIRealtime,
    };
    use lk_agents_inference::testkit::NoImages;
    use lk_agents_net::clock::RealClock;
    use lk_agents_net::task::TokioSpawner;
    use lk_agents_net::websocket::TungsteniteTransport;

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY and reaches the network"]
    async fn a_real_session_accepts_our_opening_update_and_answers() {
        let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
            eprintln!("OPENAI_API_KEY unset; skipping");
            return;
        };

        let session = <OpenAIRealtime>::new(
            OPENAI,
            api_key,
            Arc::new(TungsteniteTransport),
            Arc::new(TokioSpawner::current()),
            Arc::new(RealClock::new()),
            Arc::new(NoImages),
        )
        .session(false)
        .await;

        let req_id = session.generate_reply(Some("Say the single word: pong.".into()));

        // Bounded, like every other await in this crate's tests. A live server that says nothing is a
        // failed test, not a hung one.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
        let mut saw_audio = false;
        loop {
            let Ok(Some(ev)) = tokio::time::timeout_at(deadline, session.next_event()).await else {
                panic!("no answer within 20 s; saw_audio={saw_audio}");
            };
            match ev {
                // The server rejecting the opening `session.update` arrives here and nowhere else.
                Event::Error(e) => panic!("the server rejected something: {}", e.message),
                Event::GenerationCreated(GenerationCreated { req_id: Some(got), .. }) => {
                    assert_eq!(got, req_id, "the correlation survives a real round trip");
                }
                Event::AudioDelta(AudioDelta { .. }) => saw_audio = true,
                Event::GenerationDone(GenerationDone { status, usage, .. }) => {
                    assert_eq!(status, lk_agents_inference::realtime::GenerationStatus::Completed);
                    assert!(usage.output_tokens > 0, "the usage block is where we think it is");
                    assert!(saw_audio, "an audio session that produced no audio");
                    break;
                }
                _ => {}
            }
        }
        session.aclose().await;
    }
}
#[path = "realtime/server_tests.rs"]
mod server;
#[path = "realtime/sync_tests.rs"]
mod server_sync;
#[path = "realtime/session_tests.rs"]
mod session;
#[path = "realtime/simulation_tests.rs"]
mod sim;
