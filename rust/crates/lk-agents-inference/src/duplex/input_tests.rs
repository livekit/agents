use super::*;
use lk_agents_abi::VirtualClock;
use lk_agents_abi::llm::duplex::{DuplexInput as _, DuplexInputQueue};

/// 10 ms of 16 kHz mono, every sample `tag`.
fn frame(tag: i16) -> AudioFrame {
    AudioFrame::new(tag.to_le_bytes().repeat(160), 16_000, 1, 160)
}

fn input() -> DuplexInput {
    DuplexInput::new(Arc::new(VirtualClock::new()))
}

/// Everything queued now, as `(kind, ms, first tag)`, so assertions read as the timeline.
async fn drain(input: &DuplexInput) -> Vec<(&'static str, u64, i16)> {
    let mut items = Vec::new();
    loop {
        let next = tokio::time::timeout(Duration::from_millis(10), input.recv()).await;
        let Ok(Some(item)) = next else { return items };
        items.push(match item {
            DuplexInputItem::Audio(f) => (
                "audio",
                (f.duration() * 1e3).round() as u64,
                i16::from_le_bytes([f.data[0], f.data[1]]),
            ),
            DuplexInputItem::Gap(g) => ("gap", (g.duration * 1e3).round() as u64, 0),
            DuplexInputItem::Mute(m) => (if m.muted { "mute" } else { "unmute" }, 0, 0),
        });
    }
}

fn audio_ms(items: &[(&str, u64, i16)]) -> u64 {
    items.iter().filter(|i| i.0 == "audio").map(|i| i.1).sum()
}

// Catches: a plugin that pulls slower than the caller speaks answering seconds-old audio.
#[tokio::test]
async fn a_slow_plugin_hears_only_the_newest_audio_after_a_gap() {
    let input = input();
    for tag in 0..200 {
        DuplexInputQueue::push(&input, frame(tag));
    }
    let items = drain(&input).await;
    assert_eq!(items[0].0, "gap");
    assert!(audio_ms(&items) <= 250, "{items:?}");
    assert_eq!(items[0].1 + audio_ms(&items), 2_000, "the gap covers what was dropped");
    let last = items.last().expect("audio");
    assert_eq!(last.2 + last.1 as i16 / 10, 200, "the newest audio survives: {items:?}");
}

// Catches: the pre-connect head held to the live audio's bound, cutting the first sentence.
#[tokio::test]
async fn the_pre_connect_head_is_kept_in_full() {
    let input = input();
    for tag in 0..300 {
        input.head(frame(tag));
    }
    for tag in 300..400 {
        DuplexInputQueue::push(&input, frame(tag));
    }
    let items = drain(&input).await;
    let head: u64 = items.iter().take_while(|i| i.0 == "audio" && i.2 < 300).map(|i| i.1).sum();
    assert_eq!(head, 3_000, "{items:?}");
    assert!(audio_ms(&items) - head <= 250, "the live audio after it stays current");
}

// Catches: a mute overtaking the audio before it, an unmute lagging the audio after it, or a
// flush dropping a mute along with the audio.
#[tokio::test]
async fn mute_and_unmute_stay_where_they_were_asked() {
    let input = input();
    DuplexInputQueue::push(&input, frame(1));
    input.mute(true).unwrap();
    DuplexInputQueue::push(&input, frame(2));
    input.mute(false).unwrap();
    DuplexInputQueue::push(&input, frame(3));
    assert_eq!(
        drain(&input).await,
        [("audio", 10, 1), ("mute", 0, 0), ("audio", 10, 2), ("unmute", 0, 0), ("audio", 10, 3)]
    );

    DuplexInputQueue::push(&input, frame(4));
    input.mute(true).unwrap();
    DuplexInputQueue::push(&input, frame(5));
    input.flush().unwrap();
    DuplexInputQueue::push(&input, frame(6));
    assert_eq!(drain(&input).await, [("mute", 0, 0), ("audio", 10, 6)]);
}
