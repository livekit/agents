//! `strip_thinking_tokens` (`llm/utils.py`). `""` (an empty delta) and `None` (no delta, or pure
//! thinking) are different answers, written back into `delta.content`.

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilterState {
    /// Might be the start of a tag.
    pub buffer: String,
    /// The closing tag captured when the span opened.
    pub end_marker: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stripped {
    pub visible: Option<String>,
    pub state: FilterState,
}

/// Proper prefix: a complete marker was already consumed and would otherwise be held forever.
fn partial_marker_length(content: &str, marker: &str) -> usize {
    let marker_chars = marker.chars().count();
    if marker_chars < 2 {
        return 0;
    }
    let content_chars = content.chars().count();
    let max_chars = marker_chars.saturating_sub(1).min(content_chars);
    let mut best = 0;
    for (n, (byte_idx, _)) in marker.char_indices().enumerate().skip(1) {
        if n > max_chars {
            break;
        }
        if content.ends_with(&marker[..byte_idx]) {
            best = byte_idx;
        }
    }
    best
}

/// [`strip_thinking_tokens`] over the filter's fields: `(visible, buffer, end_marker)`.
/// `content=None` (no field) and `""` (empty field) are distinct, and only `""` can answer `""`:
/// collapsing them would make a content-filtered chunk look like pure reasoning.
#[crate::host(export(pid = false))]
pub fn strip_thinking(
    content: Option<&str>,
    start_tag: &str,
    end_tag: &str,
    buffer: &str,
    end_marker: Option<&str>,
    is_final: bool,
) -> (Option<String>, String, Option<String>) {
    let state = FilterState { buffer: buffer.to_owned(), end_marker: end_marker.map(str::to_owned) };
    let out = strip_thinking_tokens(content, start_tag, end_tag, state, is_final);
    (out.visible, out.state.buffer, out.state.end_marker)
}

/// `is_final` flushes the buffer only when no span is open: an unterminated `<think>` is dropped,
/// never spoken.
pub fn strip_thinking_tokens(
    content: Option<&str>,
    start_tag: &str,
    end_tag: &str,
    state: FilterState,
    is_final: bool,
) -> Stripped {
    let FilterState { mut buffer, mut end_marker } = state;
    if let Some(c) = content {
        buffer.push_str(c);
    }

    let mut visible = String::new();
    while !buffer.is_empty() {
        if let Some(marker) = end_marker.clone() {
            match buffer.find(&marker) {
                Some(idx) => {
                    buffer = buffer[idx + marker.len()..].to_owned();
                    end_marker = None;
                    continue;
                }
                None => {
                    // Inside a span nothing is visible; keep what could still become the close tag.
                    let keep = partial_marker_length(&buffer, &marker);
                    buffer = buffer[buffer.len() - keep..].to_owned();
                    break;
                }
            }
        }

        if let Some(idx) = buffer.find(start_tag) {
            visible.push_str(&buffer[..idx]);
            buffer = buffer[idx + start_tag.len()..].to_owned();
            end_marker = Some(end_tag.to_owned());
            continue;
        }

        let keep = partial_marker_length(&buffer, start_tag);
        visible.push_str(&buffer[..buffer.len() - keep]);
        buffer = buffer[buffer.len() - keep..].to_owned();
        break;
    }

    if is_final {
        if end_marker.is_none() {
            visible.push_str(&buffer);
        }
        buffer.clear();
        end_marker = None;
    }

    // `result or ...`: an empty delta vs no delta.
    let visible = if !visible.is_empty() {
        Some(visible)
    } else if content == Some("") {
        Some(String::new())
    } else {
        None
    };

    Stripped { visible, state: FilterState { buffer, end_marker } }
}

#[cfg(test)]
mod tests {
    use super::*;

    const START: &str = "<think>";
    const END: &str = "</think>";

    fn feed(chunks: &[Option<&str>], is_final_last: bool) -> (Vec<Option<String>>, FilterState) {
        let mut state = FilterState::default();
        let mut out = Vec::new();
        let last = chunks.len().saturating_sub(1);
        for (i, c) in chunks.iter().enumerate() {
            let r = strip_thinking_tokens(*c, START, END, state, is_final_last && i == last);
            out.push(r.visible);
            state = r.state;
        }
        (out, state)
    }

    // Catches: collapsing the empty-delta answer into `None`.
    #[test]
    fn an_empty_delta_answers_empty_and_a_missing_one_answers_none() {
        let (out, _) = feed(&[Some(""), None], false);
        assert_eq!(out, vec![Some(String::new()), None]);
    }

    // Catches: emitting a tag-shaped tail (`"<thi"`).
    #[test]
    fn a_partial_start_tag_is_held_back_until_the_next_chunk_resolves_it() {
        let (out, state) = feed(&[Some("hello <thi")], false);
        assert_eq!(out, vec![Some("hello ".to_owned())]);
        assert_eq!(state.buffer, "<thi");

        // ...and it resolves the other way too: the same tail followed by ordinary text.
        let (out, _) = feed(&[Some("hello <thi"), Some("s is fine")], false);
        assert_eq!(out, vec![Some("hello ".to_owned()), Some("<this is fine".to_owned())]);
    }

    // Catches: treating a complete marker as a partial one (stalls the stream).
    #[test]
    fn a_whole_tag_is_never_reported_as_a_partial_match() {
        assert_eq!(partial_marker_length("<think>", START), 0);
        assert_eq!(partial_marker_length("a<think", START), 6);
    }

    // Catches: leaking the reasoning body across chunks.
    #[test]
    fn a_thinking_span_split_across_chunks_emits_nothing_from_inside_it() {
        let (out, _) = feed(&[Some("a<think>se"), Some("cret</th"), Some("ink>b")], false);
        assert_eq!(out, vec![Some("a".to_owned()), None, Some("b".to_owned())]);
    }

    // Catches: flushing at end of stream while a span is still open.
    #[test]
    fn an_unterminated_thinking_span_is_dropped_at_the_end_not_spoken() {
        let (out, state) = feed(&[Some("hi <think>never closed")], true);
        assert_eq!(out, vec![Some("hi ".to_owned())]);
        assert_eq!(state, FilterState::default());
    }

    // Catches: dropping a held-back partial tag at end of stream outside a span.
    #[test]
    fn a_held_back_tail_is_flushed_by_the_final_chunk() {
        let (out, state) = feed(&[Some("done <thi")], true);
        assert_eq!(out, vec![Some("done <thi".to_owned())]);
        assert_eq!(state.buffer, "");
    }
}

/// An empty list means no text; a present empty part is an empty delta.
#[crate::host(export(name = "flatten_delta_text"))]
pub fn flatten_text_parts(parts: &[String]) -> Option<String> {
    (!parts.is_empty()).then(|| parts.concat())
}
