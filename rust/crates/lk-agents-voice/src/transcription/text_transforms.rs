use crate::transcription::filters::{MarkdownFilter, filter_emoji};

pub trait TextTransform: Send {
    /// Feed a chunk; return the pieces that are now final. May be empty.
    fn push(&mut self, chunk: &str) -> Vec<String>;
    fn flush(&mut self) -> Vec<String>;
}

/// `TextTransforms`: a builtin's name (a callable is refused at the boundary).
#[derive(Clone, Copy)]
pub enum TransformSpec {
    FilterMarkdown,
    FilterEmoji,
}

impl std::fmt::Debug for TransformSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformSpec::FilterMarkdown => f.write_str("filter_markdown"),
            TransformSpec::FilterEmoji => f.write_str("filter_emoji"),
        }
    }
}

impl TransformSpec {
    /// The resolved `tts_text_transforms`, in the reference's order; `None` runs none.
    pub fn of_set(
        set: Option<lk_agents_abi::voice::session_options::TextTransformSet>,
    ) -> Vec<Self> {
        let Some(set) = set else { return Vec::new() };
        [
            (set.filter_markdown, TransformSpec::FilterMarkdown),
            (set.filter_emoji, TransformSpec::FilterEmoji),
        ]
        .into_iter()
        .filter_map(|(on, spec)| on.then_some(spec))
        .collect()
    }

    fn build(&self) -> Box<dyn TextTransform> {
        match self {
            TransformSpec::FilterMarkdown => Box::new(MarkdownTransform(MarkdownFilter::new())),
            TransformSpec::FilterEmoji => Box::new(EmojiTransform),
        }
    }
}

/// [`filter_markdown`](crate::transcription::filters::MarkdownFilter) as a pipeline stage.
#[derive(Debug)]
pub struct MarkdownTransform(MarkdownFilter);

impl TextTransform for MarkdownTransform {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        self.0.push(chunk)
    }
    fn flush(&mut self) -> Vec<String> {
        self.0.flush()
    }
}

/// [`filter_emoji`] as a pipeline stage. Stateless, so it never holds a chunk back.
#[derive(Debug, Default)]
pub struct EmojiTransform;

impl TextTransform for EmojiTransform {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        // Yields the filtered chunk even when empty, as the reference does.
        vec![filter_emoji(chunk)]
    }
    fn flush(&mut self) -> Vec<String> {
        Vec::new()
    }
}

/// `_apply_text_transforms`: the stages, in order.
pub struct Pipeline {
    stages: Vec<Box<dyn TextTransform>>,
}

impl Pipeline {
    /// Resolve and build. Fails on the first unknown name, before any text has been seen.
    pub fn new(specs: &[TransformSpec]) -> Pipeline {
        Pipeline { stages: specs.iter().map(TransformSpec::build).collect() }
    }
}

/// A pipeline is itself a transform, so one can be nested in another.
impl TextTransform for Pipeline {
    fn push(&mut self, chunk: &str) -> Vec<String> {
        let mut current = vec![chunk.to_owned()];
        for stage in &mut self.stages {
            let mut next = Vec::new();
            for piece in &current {
                next.extend(stage.push(piece));
            }
            current = next;
        }
        current
    }

    /// End of stream: each stage flushes after the pieces carried down from above.
    fn flush(&mut self) -> Vec<String> {
        let mut carried: Vec<String> = Vec::new();
        for stage in &mut self.stages {
            let mut next = Vec::new();
            for piece in &carried {
                next.extend(stage.push(piece));
            }
            next.extend(stage.flush());
            carried = next;
        }
        carried
    }
}
