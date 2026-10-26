//! A `say(audio=)` with no words still plays its frames to the end.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, SayInput};

// Catches: an audio say with empty text produced no segment, so its frames were never read and
// the speech completed at once while the caller still held the stream open (survey seed 245).
// The reference's `_tts_task` forwards `audio` whatever the text.
#[test]
fn a_wordless_audio_say_waits_for_its_frames() {
    let say = Input::Say {
        text: AgentText(String::new()),
        input: SayInput::Audio,
        allow: None,
        to_ctx: true,
    };
    let run = simulate(async {
        let mut run = Run::start(&Setup::default(), 0, Default::default()).await;
        for input in [say, Input::Advance { ms: Millis(100) }] {
            run.apply(&input).await;
        }
        let open = run.record.host_view().speeches().iter().all(|s| !s.is_done());
        assert!(open, "the speech waits for the caller's frames");
        let feed = [
            Input::FeedAudio { stream: Ref::new(0), chunks: Count(2) },
            Input::FeedEnd { stream: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
        ];
        for input in &feed {
            run.apply(input).await;
        }
        run
    });
    let pushed = run.record.host_view().audio_pushed();
    assert!(!pushed.is_empty(), "the frames reached the output");
}
